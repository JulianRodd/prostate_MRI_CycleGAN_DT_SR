import itertools
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

import models.utils.model_utils
from loss_functions.GANLoss import GANLoss
from loss_functions.PerceptualLoss import PerceptualLoss
from loss_functions.discriminator_loss import discriminator_loss
from loss_functions.domain_adaptation_loss import DomainAdaptationLoss
from loss_functions.feature_matching_loss import feature_matching_loss
from loss_functions.mmd_loss import (
    compute_patch_mmd_loss,
)
from loss_functions.ssim_loss import SSIMLoss
from visualization.feature_visualizer import (
    visualize_gradient_flow,
    visualize_attention_maps,
    visualize_slice_attention_maps,
)
from utils.image_pool import ImagePool
from .base_model import BaseModel
from .utils.cycle_gan_utils import (
    apply_spectral_norm,
    verify_spectral_norm,
    compute_gradient_penalty,
)


class CycleGANModel(BaseModel):
    """
    CycleGANModel implements a 3D CycleGAN for medical image domain translation.
    This enhanced version includes memory optimization, mixed precision training,
    multiple advanced loss functions, spectral normalization, and feature visualization.
    """

    def name(self) -> str:
        """Return the name of the model"""
        return "CycleGANModel"

    @staticmethod
    def modify_commandline_options(parser, is_train: bool = True):
        """
        Add model-specific commandline options.

        Args:
            parser: Option parser
            is_train: Whether the model is in training mode

        Returns:
            Modified parser with model-specific options
        """
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument("--identity_loss_type_1", type=str, default="l1")
            parser.add_argument("--identity_loss_type_2", type=str, default="None")
            parser.add_argument("--cycle_loss_type_1", type=str, default="l1")
            parser.add_argument("--cycle_loss_type_2", type=str, default="None")
            parser.add_argument("--lambda_cycle_A", type=float, default=2.0)
            parser.add_argument("--lambda_cycle_B", type=float, default=2.0)
            parser.add_argument("--lambda_ganloss_A", type=float, default=1.0)
            parser.add_argument("--lambda_ganloss_B", type=float, default=1.0)
            parser.add_argument("--lambda_vgg", type=float, default=1.0)
            parser.add_argument("--lambda_feature_matching", type=float, default=10.0)
            parser.add_argument("--lambda_identity", type=float, default=0.5)
            parser.add_argument("--lambda_domain_adaptation", type=float, default=1.0)
            parser.add_argument("--lambda_da_contrast", type=float, default=1.0)
            parser.add_argument("--lambda_da_structure", type=float, default=1.0)
            parser.add_argument("--lambda_da_texture", type=float, default=1.0)
            parser.add_argument("--lambda_da_histogram", type=float, default=1.0)
            parser.add_argument("--lambda_da_gradient", type=float, default=1.0)
            parser.add_argument("--lambda_da_ncc", type=float, default=1.0)
            parser.add_argument("--constant_for_A", type=float, default=0.5)
            parser.add_argument(
                "--lambda_mmd",
                type=float,
                default=1.0,
                help="weight for Maximum Mean Discrepancy loss",
            )
            parser.add_argument(
                "--use_r1_penalty",
                action="store_true",
                help="use R1 gradient penalty instead of WGAN-GP",
            )
            parser.add_argument(
                "--lambda_r1",
                type=float,
                default=10.0,
                help="weight for R1 regularization",
            )
            parser.add_argument(
                "--use_gradient_checkpointing",
                action="store_true",
                help="use gradient checkpointing to save memory",
            )
            parser.add_argument(
                "--visualize_features",
                action="store_true",
                help="Enable feature visualization",
            )
            parser.add_argument(
                "--vis_freq",
                type=int,
                default=10,
                help="Frequency of feature visualization (epochs)",
            )
            parser.add_argument(
                "--vis_path",
                type=str,
                default="./visualizations",
                help="Path to save visualizations",
            )
            parser.add_argument(
                "--vis_max_channels",
                type=int,
                default=16,
                help="Maximum number of channels to visualize per layer",
            )
            parser.add_argument(
                "--lambda_mmd_bridge",
                type=float,
                default=10,
                help="Weight for MMD loss on generator bridge features",
            )
            parser.add_argument(
                "--disc_update_freq",
                type=int,
                default=1,
                help="Frequency of discriminator updates",
            )
        return parser

    def initialize(self, opt):
        """
        Initialize the model and set up feature visualization.

        Args:
            opt: Command line options
        """
        BaseModel.initialize(self, opt)
        self._init_tensors()

        # Initialize attributes for training control
        self.current_update_disc = True  # Default to updating discriminator
        self.disc_update_counter = 0  # Initialize counter

        self.using_grad_checkpointing = (
            hasattr(opt, "use_gradient_checkpointing")
            and opt.use_gradient_checkpointing
        )

        self.using_cpu = self.device.type == "cpu"

        self._init_networks(opt)
        self._init_loss_functions(opt)
        self._init_optimizers(opt)

        # Setup visualization tools if enabled
        self.visualize_features = (
            hasattr(opt, "visualize_features") and opt.visualize_features
        )

        if self.visualize_features:
            from visualization.feature_visualizer import (
                FeatureVisualizer,
                visualize_gradient_flow,
                visualize_attention_maps,
            )

            self.feature_visualizer = FeatureVisualizer(
                save_path=opt.vis_path,
                max_channels=opt.vis_max_channels,
                normalize=True,
            )
            self.vis_freq = opt.vis_freq
            self.iter_count = 0

        if (
            hasattr(opt, "debug")
            and opt.debug
            and hasattr(opt, "use_spectral_norm_G")
            and opt.use_spectral_norm_G
        ):
            verify_spectral_norm(self.netG_A, self.netG_B)

    def _init_tensors(self):
        """Initialize all tensors and loss values"""
        self.real_A = self.real_B = self.fake_A = self.fake_B = None
        self.rec_A = self.rec_B = self.idt_A = self.idt_B = None
        self._init_loss_values()
        self._init_names()

    def _init_loss_values(self):
        """Initialize all loss values including domain adaptation components"""
        loss_attrs = [
            "D_A",
            "D_B",
            "G_A",
            "G_B",
            "cycle_A",
            "cycle_B",
            "identity_A",
            "identity_B",
            "vgg_A",
            "vgg_B",
            "G_A_gan",
            "G_B_gan",
            "domain_adaptation_A",
            "domain_adaptation_B",
            "G",
            "feature_matching_A",
            "feature_matching_B",
            "da_contrast_A",
            "da_texture_A",
            "da_structure_A",
            "da_contrast_B",
            "da_texture_B",
            "da_structure_B",
            "mmd_A",
            "mmd_B",
            "mmd_bridge_A",
            "mmd_bridge_B",
        ]

        if (
            hasattr(self, "opt")
            and hasattr(self.opt, "use_r1_penalty")
            and self.opt.use_r1_penalty
        ):
            loss_attrs.extend(["r1_A", "r1_B"])

        for attr in loss_attrs:
            setattr(self, f"loss_{attr}", 0.0)

    def _init_names(self):
        """Initialize model names and dynamically set active loss names based on lambda values"""
        self.loss_names = ["D_A", "D_B", "G"]

        if hasattr(self.opt, "lambda_ganloss_A") and self.opt.lambda_ganloss_A > 0:
            self.loss_names.append("G_A_gan")

        if hasattr(self.opt, "lambda_ganloss_B") and self.opt.lambda_ganloss_B > 0:
            self.loss_names.append("G_B_gan")

        if hasattr(self.opt, "lambda_cycle_A") and self.opt.lambda_cycle_A > 0:
            self.loss_names.extend(["G_A", "cycle_A"])

        if hasattr(self.opt, "lambda_cycle_B") and self.opt.lambda_cycle_B > 0:
            self.loss_names.extend(["G_B", "cycle_B"])

        if hasattr(self.opt, "lambda_identity") and self.opt.lambda_identity > 0:
            self.loss_names.extend(["identity_A", "identity_B"])

        if hasattr(self.opt, "lambda_vgg") and self.opt.lambda_vgg > 0:
            self.loss_names.extend(["vgg_A", "vgg_B"])
        if hasattr(self.opt, "lambda_mmd") and self.opt.lambda_mmd > 0:
            self.loss_names.extend(["mmd_A", "mmd_B"])
        if (
            hasattr(self.opt, "lambda_domain_adaptation")
            and self.opt.lambda_domain_adaptation > 0
        ):
            self.loss_names.extend(["domain_adaptation_A", "domain_adaptation_B"])

            # Add individual domain adaptation components
            da_components = [
                "da_contrast_A",
                "da_texture_A",
                "da_structure_A",
                "da_contrast_B",
                "da_texture_B",
                "da_structure_B",
            ]
            self.loss_names.extend(da_components)

        if hasattr(self.opt, "lambda_mmd_bridge") and self.opt.lambda_mmd_bridge > 0:
            self.loss_names.extend(["mmd_bridge_A", "mmd_bridge_B"])
        if (
            hasattr(self.opt, "lambda_feature_matching")
            and self.opt.lambda_feature_matching > 0
        ):
            self.loss_names.extend(["feature_matching_A", "feature_matching_B"])

        # Remove duplicate loss names
        self.loss_names = list(dict.fromkeys(self.loss_names))

        self.visual_names = [
            "real_A",
            "fake_B",
            "rec_A",
            "idt_A",
            "real_B",
            "fake_A",
            "rec_B",
            "idt_B",
        ]
        self.model_names = (
            ["G_A", "G_B", "D_A", "D_B"] if self.isTrain else ["G_A", "G_B"]
        )

    def _init_networks(self, opt):
        """
        Initialize generator and discriminator networks with appropriate parameters.

        Args:
            opt: Command line options
        """
        self.netG_A = self._create_generator(opt.input_nc, opt.output_nc, opt)
        self.netG_B = self._create_generator(opt.output_nc, opt.input_nc, opt)

        try:
            if (
                hasattr(opt, "use_gradient_checkpointing")
                and opt.use_gradient_checkpointing
            ):
                print("Attempting to enable gradient checkpointing...")
                self.netG_A.apply(self._enable_grad_checkpointing)
                self.netG_B.apply(self._enable_grad_checkpointing)
        except Exception as e:
            print(f"Warning: Failed to enable gradient checkpointing: {e}")
            print("Continuing without gradient checkpointing")

        if self.isTrain:
            self.netD_A = self._create_discriminator(opt.output_nc, opt)
            self.netD_B = self._create_discriminator(opt.input_nc, opt)

            try:
                if (
                    hasattr(opt, "use_gradient_checkpointing")
                    and opt.use_gradient_checkpointing
                ):
                    self.netD_A.apply(self._enable_grad_checkpointing)
                    self.netD_B.apply(self._enable_grad_checkpointing)
            except Exception as e:
                print(
                    f"Warning: Failed to enable gradient checkpointing for discriminators: {e}"
                )

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

    def _enable_grad_checkpointing(self, module):
        """
        Enable gradient checkpointing for a module to save memory.

        Args:
            module: PyTorch module to enable gradient checkpointing for
        """
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable(use_reentrant=False)
            return

        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):

            def create_custom_forward(original_forward):
                def custom_forward(*args, **kwargs):
                    return original_forward(*args, **kwargs)

                return custom_forward

            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward

                def checkpointed_forward(*args, **kwargs):
                    if module.training:
                        return torch.utils.checkpoint.checkpoint(
                            create_custom_forward(module._original_forward),
                            *args,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    else:
                        return module._original_forward(*args, **kwargs)

                module.forward = checkpointed_forward

        for name, child in module.named_children():
            if sum(1 for _ in child.children()) > 1:
                self._enable_grad_checkpointing(child)

    def _create_generator(self, input_nc, output_nc, opt):
        """
        Create generator with optional spectral normalization.

        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            opt: Command line options

        Returns:
            Configured generator network
        """
        try:
            # Set use_stn to False for better memory efficiency
            use_stn = self.opt.use_stn

            netG = models.utils.model_utils.define_generator(
                input_nc,
                output_nc,
                opt.ngf,
                opt.norm,
                not opt.no_dropout,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
                use_stn=use_stn,
                pretrained_path=(
                    opt.pretrained_path_generator
                    if hasattr(opt, "pretrained_path_generator")
                    else None
                ),
                patch_size=opt.patch_size if hasattr(opt, "patch_size") else None,
                use_residual=self.opt.use_residual,
                use_full_attention=self.opt.use_full_attention,
            )

            if hasattr(opt, "use_spectral_norm_G") and opt.use_spectral_norm_G:
                print("Applying spectral normalization to generator...")
                netG = apply_spectral_norm(netG)
                print("Spectral normalization applied")

            return netG

        except Exception as e:
            print(f"Error creating generator: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def _create_discriminator(self, nc, opt):
        """
        Create discriminator with specified parameters.

        Args:
            nc: Number of input channels
            opt: Command line options

        Returns:
            Configured discriminator network
        """
        return models.utils.model_utils.define_discriminator(
            nc,
            opt.ndf,
            opt.n_layers_D,
            opt.norm,
            not opt.use_lsgan,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

    def _init_loss_functions(self, opt):
        """
        Initialize all loss functions used for training.

        Args:
            opt: Command line options
        """
        if self.isTrain:
            self.criterionGAN = GANLoss(
                use_lsgan=opt.use_lsgan,
                use_hinge=opt.use_hinge,
                use_wasserstein=opt.use_wasserstein,
                relativistic=opt.use_relativistic,
                device=self.device,
            )
            self.criterionPerceptual = PerceptualLoss(device=self.device)

            if opt.lambda_domain_adaptation > 0:
                # Use optimized domain adaptation loss with 3 key components
                self.criterionDomain = DomainAdaptationLoss(
                    device=self.device,
                    contrast_weight=opt.lambda_da_contrast,
                    structure_weight=opt.lambda_da_structure,
                    texture_weight=opt.lambda_da_texture,
                    histogram_weight=opt.lambda_da_histogram,
                    gradient_weight=opt.lambda_da_gradient,
                    ncc_weight=opt.lambda_da_ncc,
                    return_components=True,
                )

    def _init_optimizers(self, opt):
        """
        Initialize optimizers for generators and discriminators.

        Args:
            opt: Command line options
        """
        if self.isTrain:

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=(
                    opt.lr * opt.lr_D_Constant
                    if hasattr(opt, "lr_D_Constant")
                    else opt.lr
                ),
                betas=(opt.beta1, 0.999),
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input):
        """
        Handle input assignment with memory optimizations.

        Args:
            input: Input data (tuple, list, or tensor)
        """
        if input is None:
            raise ValueError("Input cannot be None")

        AtoB = self.opt.which_direction == "AtoB"

        if isinstance(input, torch.Tensor):
            self.real_A = input.to(self.device)
            self.real_B = torch.zeros_like(
                self.real_A, dtype=torch.float32, device=self.device
            )
        elif isinstance(input, (list, tuple)) and len(input) >= 2:
            try:
                self.real_A = input[0 if AtoB else 1].to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                self.real_B = input[1 if AtoB else 0].to(
                    self.device, dtype=torch.float32, non_blocking=True
                )

                input = None
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    import gc

                    gc.collect()

                    self.real_A = (
                        input[0 if AtoB else 1]
                        .detach()
                        .to(self.device, dtype=torch.float32)
                    )
                    self.real_B = (
                        input[1 if AtoB else 0]
                        .detach()
                        .to(self.device, dtype=torch.float32)
                    )
                else:
                    raise
        else:
            raise ValueError("Invalid input format")

    def forward(self):
        """
        Memory-optimized forward pass with sequential operations.
        Generates fake images, reconstructions, and identity mappings.
        """
        if self.real_A is None or self.real_B is None:
            raise RuntimeError("Inputs must be set before forward pass")

        with torch.set_grad_enabled(self.isTrain):
            self.fake_B = self.netG_A(self.real_A.float())
            self.fake_A = self.netG_B(self.real_B.float())

            if self.isTrain and (
                self.opt.lambda_cycle_A > 0 or self.opt.lambda_cycle_B > 0
            ):
                self.rec_A = self.netG_B(self.fake_B)
                self.rec_B = self.netG_A(self.fake_A)
            else:
                self.rec_A = None
                self.rec_B = None

            if self.isTrain and self.opt.lambda_identity > 0:
                self.idt_A = self.netG_A(self.real_B.float())
                self.idt_B = self.netG_B(self.real_A.float())
            else:
                self.idt_A = None
                self.idt_B = None

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _compute_generator_losses(self):
        """
        Compute all generator losses with configurable loss types.

        Returns:
            Combined generator loss value
        """
        # Dictionary to collect losses before applying weights
        unweighted_losses = {}

        # Safety check for NaN values
        self.fake_A = torch.nan_to_num(self.fake_A, nan=0.0)
        self.fake_B = torch.nan_to_num(self.fake_B, nan=0.0)

        # Convert to float32 for numerical stability
        fake_B_float = self.fake_B.float()
        fake_A_float = self.fake_A.float()
        real_B_float = self.real_B.float()
        real_A_float = self.real_A.float()

        # Create a reference tensor for gradient connection
        grad_reference = fake_B_float  # We know this tensor has requires_grad=True

        # Helper function to create zero-tensor with grad connection
        def create_zero_loss():
            return grad_reference.sum() * 0

        # Extract bridge features directly if MMD bridge loss is used
        lambda_mmd_bridge = getattr(self.opt, "lambda_mmd_bridge", 0.0)
        if lambda_mmd_bridge > 0:
            # Extract features directly when needed
            bridge_features_A = self._extract_bridge_features(
                self.netG_A, self.real_A.float()
            )
            bridge_features_B = self._extract_bridge_features(
                self.netG_B, self.real_B.float()
            )
        else:
            bridge_features_A = []
            bridge_features_B = []

        # Create a dictionary of loss functions for compute_loss_by_type
        loss_criterions = {
            "ssim": SSIMLoss(device=self.device),
            "perceptual": self.criterionPerceptual,
        }

        # 1. Adversarial losses
        # 1. Adversarial losses
        if hasattr(self.opt, "lambda_ganloss_A") and self.opt.lambda_ganloss_A > 0:
            use_wasserstein = getattr(self.opt, "use_wasserstein", False)

            if use_wasserstein:
                # Wasserstein generator loss - simply maximize critic output
                pred_fake_B = self.netD_A(fake_B_float, use_wasserstein=True)
                pred_fake_A = self.netD_B(fake_A_float, use_wasserstein=True)

                # Negate loss to maximize critic output
                unweighted_losses["ganloss_A"] = -torch.mean(pred_fake_B)
                unweighted_losses["ganloss_B"] = -torch.mean(pred_fake_A)

                # No feature matching for basic Wasserstein
                unweighted_losses["feature_matching_A"] = create_zero_loss()
                unweighted_losses["feature_matching_B"] = create_zero_loss()

                # Set empty feature lists
                real_B_features = []
                real_A_features = []
                fake_B_features = []
                fake_A_features = []
            else:
                # Get discriminator outputs with feature extraction
                pred_fake_B, fake_B_features = self.netD_A(
                    fake_B_float, get_features=True
                )
                pred_real_B, real_B_features = self.netD_A(
                    real_B_float, get_features=True
                )

                pred_fake_A, fake_A_features = self.netD_B(
                    fake_A_float, get_features=True
                )
                pred_real_A, real_A_features = self.netD_B(
                    real_A_float, get_features=True
                )

                # Relativistic GAN formulation
                avg_pred_real_B = torch.mean(pred_real_B)
                avg_pred_real_A = torch.mean(pred_real_A)

                rel_pred_fake_B = pred_fake_B - avg_pred_real_B
                rel_pred_fake_A = pred_fake_A - avg_pred_real_A

                # Calculate GAN losses
                unweighted_losses["ganloss_A"] = self.criterionGAN(
                    rel_pred_fake_B, True
                )
                unweighted_losses["ganloss_B"] = self.criterionGAN(
                    rel_pred_fake_A, True
                )

                # Calculate feature matching losses separately
                if (
                    hasattr(self.opt, "lambda_feature_matching")
                    and self.opt.lambda_feature_matching > 0
                ):
                    unweighted_losses["feature_matching_A"] = feature_matching_loss(
                        real_B_features, fake_B_features
                    )
                    unweighted_losses["feature_matching_B"] = feature_matching_loss(
                        real_A_features, fake_A_features
                    )
                else:
                    unweighted_losses["feature_matching_A"] = create_zero_loss()
                    unweighted_losses["feature_matching_B"] = create_zero_loss()

        # 2. Cycle consistency losses
        if hasattr(self.opt, "lambda_cycle_A") and self.opt.lambda_cycle_A > 0:
            # Ensure dimensions match
            if self.rec_A.shape != self.real_A.shape:
                self.rec_A = F.interpolate(
                    self.rec_A,
                    size=self.real_A.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                )

            if self.rec_B.shape != self.real_B.shape:
                self.rec_B = F.interpolate(
                    self.rec_B,
                    size=self.real_B.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                )

            # Use the configured loss types for cycle consistency
            from loss_functions.combined_loss import compute_combined_loss

            # Calculate cycle losses based on specified types
            unweighted_losses["cycle_A"] = compute_combined_loss(
                self.rec_A,
                self.real_A,
                self.opt.cycle_loss_type_1,
                self.opt.cycle_loss_type_2,
                self.device,
                loss_criterions,
            )

            unweighted_losses["cycle_B"] = compute_combined_loss(
                self.rec_B,
                self.real_B,
                self.opt.cycle_loss_type_1,
                self.opt.cycle_loss_type_2,
                self.device,
                loss_criterions,
            )
        else:
            unweighted_losses["cycle_A"] = create_zero_loss()
            unweighted_losses["cycle_B"] = create_zero_loss()

        # 3. Identity losses
        if (
            hasattr(self.opt, "lambda_identity")
            and self.opt.lambda_identity > 0
            and self.idt_A is not None
        ):
            # Ensure dimensions match
            if self.idt_A.shape != self.real_B.shape:
                self.idt_A = F.interpolate(
                    self.idt_A,
                    size=self.real_B.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                )

            if self.idt_B.shape != self.real_A.shape:
                self.idt_B = F.interpolate(
                    self.idt_B,
                    size=self.real_A.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                )

            # Use the configured loss types for identity preservation
            from loss_functions.combined_loss import compute_combined_loss

            # Calculate identity losses based on specified types
            unweighted_losses["identity_A"] = compute_combined_loss(
                self.idt_A,
                self.real_B,
                self.opt.identity_loss_type_1,
                self.opt.identity_loss_type_2,
                self.device,
                loss_criterions,
            )

            unweighted_losses["identity_B"] = compute_combined_loss(
                self.idt_B,
                self.real_A,
                self.opt.identity_loss_type_1,
                self.opt.identity_loss_type_2,
                self.device,
                loss_criterions,
            )
        else:
            unweighted_losses["identity_A"] = create_zero_loss()
            unweighted_losses["identity_B"] = create_zero_loss()

        # 4. VGG perceptual losses - calculate separately for each direction
        if hasattr(self.opt, "lambda_vgg") and self.opt.lambda_vgg > 0:
            # Ensure dimensions match
            fake_B_for_vgg = self.fake_B
            if fake_B_for_vgg.shape != self.real_A.shape:
                fake_B_for_vgg = F.interpolate(
                    fake_B_for_vgg,
                    size=self.real_A.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                )

            fake_A_for_vgg = self.fake_A
            if fake_A_for_vgg.shape != self.real_B.shape:
                fake_A_for_vgg = F.interpolate(
                    fake_A_for_vgg,
                    size=self.real_B.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                )

            # Calculate separate VGG losses
            unweighted_losses["vgg_A"] = self.criterionPerceptual(
                fake_B_for_vgg, self.real_A
            )
            unweighted_losses["vgg_B"] = self.criterionPerceptual(
                fake_A_for_vgg, self.real_B
            )
        else:
            unweighted_losses["vgg_A"] = create_zero_loss()
            unweighted_losses["vgg_B"] = create_zero_loss()

        # 5. Domain adaptation losses with component tracking
        if (
            hasattr(self.opt, "lambda_domain_adaptation")
            and self.opt.lambda_domain_adaptation > 0
        ):
            try:
                # Get loss and components for A domain
                domain_loss_A, comp_A = self.criterionDomain(self.fake_A, self.real_A)
                # Get loss and components for B domain
                domain_loss_B, comp_B = self.criterionDomain(self.fake_B, self.real_B)

                # Set individual components for logging
                self.loss_da_contrast_A = comp_A.get("contrast", 0.0)
                self.loss_da_texture_A = comp_A.get("texture", 0.0)
                self.loss_da_structure_A = comp_A.get("structure", 0.0)

                self.loss_da_contrast_B = comp_B.get("contrast", 0.0)
                self.loss_da_texture_B = comp_B.get("texture", 0.0)
                self.loss_da_structure_B = comp_B.get("structure", 0.0)

                # Safety check - clamp domain losses
                domain_loss_A = torch.clamp(domain_loss_A, 0.0, 100.0)
                domain_loss_B = torch.clamp(domain_loss_B, 0.0, 100.0)

                # Store domain losses
                unweighted_losses["domain_adaptation_A"] = domain_loss_A
                unweighted_losses["domain_adaptation_B"] = domain_loss_B

            except Exception as e:
                unweighted_losses["domain_adaptation_A"] = create_zero_loss()
                unweighted_losses["domain_adaptation_B"] = create_zero_loss()

                # Zero out component values
                self.loss_da_contrast_A = 0.0
                self.loss_da_texture_A = 0.0
                self.loss_da_structure_A = 0.0
                self.loss_da_contrast_B = 0.0
                self.loss_da_texture_B = 0.0
                self.loss_da_structure_B = 0.0
        else:
            unweighted_losses["domain_adaptation_A"] = create_zero_loss()
            unweighted_losses["domain_adaptation_B"] = create_zero_loss()

            # Zero out component values
            self.loss_da_contrast_A = 0.0
            self.loss_da_texture_A = 0.0
            self.loss_da_structure_A = 0.0
            self.loss_da_contrast_B = 0.0
            self.loss_da_texture_B = 0.0
            self.loss_da_structure_B = 0.0

        # 6. MMD losses on discriminator features
        if hasattr(self.opt, "lambda_mmd") and self.opt.lambda_mmd > 0:
            try:
                # Extract discriminator features
                if "fake_A_features" in locals() and "real_B_features" in locals():
                    unweighted_losses["mmd_A"] = compute_patch_mmd_loss(
                        fake_A_features,  # Features from fake A (generated from B)
                        real_B_features,  # Features from real B
                        device=self.device,
                        weight=1.0,
                    )
                else:
                    unweighted_losses["mmd_A"] = create_zero_loss()

                if "fake_B_features" in locals() and "real_A_features" in locals():
                    unweighted_losses["mmd_B"] = compute_patch_mmd_loss(
                        fake_B_features,  # Features from fake B (generated from A)
                        real_A_features,  # Features from real A
                        device=self.device,
                        weight=1.0,
                    )
                else:
                    unweighted_losses["mmd_B"] = create_zero_loss()

            except Exception as e:
                unweighted_losses["mmd_A"] = create_zero_loss()
                unweighted_losses["mmd_B"] = create_zero_loss()
        else:
            unweighted_losses["mmd_A"] = create_zero_loss()
            unweighted_losses["mmd_B"] = create_zero_loss()

        # 7. MMD Bridge Losses
        unweighted_losses["mmd_bridge_A"] = create_zero_loss()
        unweighted_losses["mmd_bridge_B"] = create_zero_loss()

        if lambda_mmd_bridge > 0:
            try:
                # Explicitly use full float32 precision for bridge computation on GPU
                with (
                    torch.cuda.amp.autocast(enabled=False)
                    if torch.cuda.is_available()
                    else nullcontext()
                ):
                    # Extract features with GPU optimization
                    bridge_features_A = self._extract_bridge_features(
                        self.netG_A, self.real_A.float()
                    )
                    bridge_features_B = self._extract_bridge_features(
                        self.netG_B, self.real_B.float()
                    )

                # Safety for GPU-specific issues
                if self.device.type == "cuda":
                    # Simple structure for MMD computation that avoids dimension issues
                    bridge_A_flat = (
                        bridge_features_A[0]
                        .view(bridge_features_A[0].size(0), -1)
                        .float()
                    )
                    bridge_B_flat = (
                        bridge_features_B[0]
                        .view(bridge_features_B[0].size(0), -1)
                        .float()
                    )

                    # Downsample if features are too large (GPU memory optimization)
                    if bridge_A_flat.size(1) > 4096:
                        keep_dims = 4096
                        bridge_A_flat = bridge_A_flat[:, :keep_dims]
                        bridge_B_flat = bridge_B_flat[:, :keep_dims]

                    # Match dimensions if needed
                    min_dim = min(bridge_A_flat.size(1), bridge_B_flat.size(1))
                    bridge_A_flat = bridge_A_flat[:, :min_dim]
                    bridge_B_flat = bridge_B_flat[:, :min_dim]

                    # Small batch for GPU computation
                    batch_size = min(bridge_A_flat.size(0), bridge_B_flat.size(0), 8)
                    bridge_A_flat = bridge_A_flat[:batch_size]
                    bridge_B_flat = bridge_B_flat[:batch_size]

                    # Compute MMD with explicit float32
                    unweighted_losses["mmd_bridge_A"] = compute_patch_mmd_loss(
                        bridge_A_flat, bridge_B_flat, device=self.device, weight=1.0
                    )

                    unweighted_losses["mmd_bridge_B"] = compute_patch_mmd_loss(
                        bridge_B_flat, bridge_A_flat, device=self.device, weight=1.0
                    )
                else:
                    # CPU implementation as before
                    unweighted_losses["mmd_bridge_A"] = compute_patch_mmd_loss(
                        bridge_features_A[0],
                        bridge_features_B[0],
                        device=self.device,
                        weight=1.0,
                    )

                    unweighted_losses["mmd_bridge_B"] = compute_patch_mmd_loss(
                        bridge_features_B[0],
                        bridge_features_A[0],
                        device=self.device,
                        weight=1.0,
                    )

            except Exception as e:
                print(f"Error in GPU MMD bridge calculation: {e}")
                unweighted_losses["mmd_bridge_A"] = create_zero_loss()
                unweighted_losses["mmd_bridge_B"] = create_zero_loss()
        else:
            unweighted_losses["mmd_bridge_A"] = create_zero_loss()
            unweighted_losses["mmd_bridge_B"] = create_zero_loss()

        # Ensure all loss tensors maintain gradient connection
        for key in unweighted_losses:
            if (
                not isinstance(unweighted_losses[key], torch.Tensor)
                or not unweighted_losses[key].requires_grad
            ):
                unweighted_losses[key] = grad_reference.sum() * 0 + float(
                    unweighted_losses[key]
                )

        # Apply weights to losses
        # 1. Get lambda values
        lambda_ganloss_A = getattr(self.opt, "lambda_ganloss_A", 1.0)
        lambda_ganloss_B = getattr(self.opt, "lambda_ganloss_B", 1.0)
        lambda_cycle_A = getattr(self.opt, "lambda_cycle_A", 2.0)
        lambda_cycle_B = getattr(self.opt, "lambda_cycle_B", 2.0)
        lambda_identity = getattr(self.opt, "lambda_identity", 0.5)
        lambda_vgg = getattr(self.opt, "lambda_vgg", 1.0)
        lambda_feature_matching = getattr(self.opt, "lambda_feature_matching", 10.0)
        lambda_domain_adaptation = getattr(self.opt, "lambda_domain_adaptation", 1.0)
        lambda_mmd = getattr(self.opt, "lambda_mmd", 1.0)

        # 2. Apply weights to individual losses
        self.loss_G_A_gan = unweighted_losses["ganloss_A"] * lambda_ganloss_A
        self.loss_G_B_gan = unweighted_losses["ganloss_B"] * lambda_ganloss_B

        self.loss_cycle_A = unweighted_losses["cycle_A"] * lambda_cycle_A
        self.loss_cycle_B = unweighted_losses["cycle_B"] * lambda_cycle_B

        self.loss_identity_A = unweighted_losses["identity_A"] * lambda_identity
        self.loss_identity_B = unweighted_losses["identity_B"] * lambda_identity

        self.loss_vgg_A = unweighted_losses["vgg_A"] * lambda_vgg
        self.loss_vgg_B = unweighted_losses["vgg_B"] * lambda_vgg

        self.loss_feature_matching_A = (
            unweighted_losses["feature_matching_A"] * lambda_feature_matching
        )
        self.loss_feature_matching_B = (
            unweighted_losses["feature_matching_B"] * lambda_feature_matching
        )

        self.loss_domain_adaptation_A = (
            unweighted_losses["domain_adaptation_A"] * lambda_domain_adaptation
        )
        self.loss_domain_adaptation_B = (
            unweighted_losses["domain_adaptation_B"] * lambda_domain_adaptation
        )

        self.loss_mmd_A = unweighted_losses["mmd_A"] * lambda_mmd
        self.loss_mmd_B = unweighted_losses["mmd_B"] * lambda_mmd

        self.loss_mmd_bridge_A = (
            unweighted_losses["mmd_bridge_A"] * lambda_mmd_bridge * 1000
        )
        self.loss_mmd_bridge_B = (
            unweighted_losses["mmd_bridge_B"] * lambda_mmd_bridge * 1000
        )

        # 3. Calculate total generator losses for each direction
        self.loss_G_A = (
            self.loss_G_A_gan
            + self.loss_cycle_A
            + self.loss_vgg_A
            + self.loss_identity_A
            + self.loss_feature_matching_A
            + self.loss_domain_adaptation_A
            + self.loss_mmd_A
            + self.loss_mmd_bridge_A
        )

        self.loss_G_B = (
            self.loss_G_B_gan
            + self.loss_cycle_B
            + self.loss_vgg_B
            + self.loss_identity_B
            + self.loss_feature_matching_B
            + self.loss_domain_adaptation_B
            + self.loss_mmd_B
            + self.loss_mmd_bridge_B
        )

        # 4. Calculate weighted total loss
        constant_for_A = self.opt.constant_for_A
        self.loss_G = self.loss_G_A * constant_for_A + self.loss_G_B * (
            1 - constant_for_A
        )

        # Final safety check - clamp total loss
        self.loss_G = torch.clamp(self.loss_G, 0.0, 1000.0)
        if torch.isnan(self.loss_G) or torch.isinf(self.loss_G):
            # Create a detached tensor with gradient connection
            self.loss_G = grad_reference.sum() * 0 + 100.0

        return self.loss_G

    def compute_losses(self, scaler=None, accumulate_grad=False):
        """
        Enhanced compute_losses with null safety for image pools and Wasserstein GAN support.

        Args:
            scaler: GradScaler for mixed precision training
            accumulate_grad: Whether to accumulate gradients across multiple batches

        Returns:
            Dictionary of loss values
        """
        # Get update_disc flag from trainer
        update_disc = getattr(self, "current_update_disc", True)

        # Initialize disc counter if missing
        if not hasattr(self, "disc_update_counter"):
            self.disc_update_counter = 0

        # --- Forward pass first to set real_A and real_B ---
        with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=scaler is not None,
        ):
            # Forward pass to generate real_A, real_B, fake_A, fake_B
            self.forward()

            # Initialize fake outputs if not set (safety check)
            if not hasattr(self, "fake_B") or self.fake_B is None:
                self.fake_B = (
                    torch.zeros_like(self.real_A)
                    if hasattr(self, "real_A") and self.real_A is not None
                    else torch.zeros(1, device=self.device)
                )
            if not hasattr(self, "fake_A") or self.fake_A is None:
                self.fake_A = (
                    torch.zeros_like(self.real_B)
                    if hasattr(self, "real_B") and self.real_B is not None
                    else torch.zeros(1, device=self.device)
                )

            # --- Add null-safe detachment with explicit None checks ---
            self.fake_B_detached = (
                self.fake_B.detach()
                if self.fake_B is not None
                else (
                    torch.zeros_like(self.real_A)
                    if hasattr(self, "real_A")
                    else torch.zeros(1, device=self.device)
                )
            )
            self.fake_A_detached = (
                self.fake_A.detach()
                if self.fake_A is not None
                else (
                    torch.zeros_like(self.real_B)
                    if hasattr(self, "real_B")
                    else torch.zeros(1, device=self.device)
                )
            )

            # Ensure real_A/real_B exist and are not None before detaching
            self.real_A_detached = (
                self.real_A.detach()
                if (hasattr(self, "real_A") and self.real_A is not None)
                else (
                    torch.zeros_like(self.fake_B)
                    if self.fake_B is not None
                    else torch.zeros(1, device=self.device)
                )
            )
            self.real_B_detached = (
                self.real_B.detach()
                if (hasattr(self, "real_B") and self.real_B is not None)
                else (
                    torch.zeros_like(self.fake_A)
                    if self.fake_A is not None
                    else torch.zeros(1, device=self.device)
                )
            )

            self.set_requires_grad([self.netD_A, self.netD_B], False)
            self.loss_G = self._compute_generator_losses()

        # --- Add null checks before image pool queries ---
        # Safe detach with fallback tensors
        def safe_query(pool, tensor):
            """Handle empty pools and None tensors safely"""
            if tensor is None:
                return torch.zeros_like(self.real_A)  # Ensure real_A exists in scope

            # Explicitly detach tensor
            tensor_detached = tensor.detach()

            # Check pool_size attribute directly
            if pool.pool_size == 0:
                return tensor_detached

            return pool.query(tensor_detached)

        # Use it like this in compute_losses():
        fake_B = safe_query(self.fake_B_pool, self.fake_B_detached)
        fake_A = safe_query(self.fake_A_pool, self.fake_A_detached)

        # Handle gradient accumulation
        effective_g_loss = (
            self.loss_G / self.opt.accumulation_steps
            if accumulate_grad
            else self.loss_G
        )

        # Backprop generator
        if scaler:
            scaler.scale(effective_g_loss).backward(retain_graph=False)
        else:
            effective_g_loss.backward(retain_graph=False)

        # Initialize loss variables for discriminator
        self.loss_D_A = torch.tensor(0.0, device=self.device)
        self.loss_D_B = torch.tensor(0.0, device=self.device)

        # Debug print if updating discriminator
        if update_disc:
            print(
                f"Updating discriminator (freq {self.opt.disc_update_freq})", flush=True
            )

        # --- Discriminator updates ---
        use_wasserstein = getattr(self.opt, "use_wasserstein", False)

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        if update_disc:
            # Zero grads before backward pass (critical fix)
            self.optimizer_D.zero_grad(set_to_none=True)

            if use_wasserstein:
                # Prepare input with proper gradient tracking
                real_B_with_grad = self.real_B_detached.clone().requires_grad_(True)
                real_A_with_grad = self.real_A_detached.clone().requires_grad_(True)

                # Get discriminator outputs with Wasserstein flag
                pred_real_B = self.netD_A(real_B_with_grad, use_wasserstein=True)
                pred_fake_B = self.netD_A(fake_B, use_wasserstein=True)
                pred_real_A = self.netD_B(real_A_with_grad, use_wasserstein=True)
                pred_fake_A = self.netD_B(fake_A, use_wasserstein=True)

                # Wasserstein loss components
                loss_D_A_real = -torch.mean(pred_real_B)
                loss_D_A_fake = torch.mean(pred_fake_B)
                loss_D_B_real = -torch.mean(pred_real_A)
                loss_D_B_fake = torch.mean(pred_fake_A)

                # Calculate gradient penalties
                gp_A = compute_gradient_penalty(
                    self.netD_A,
                    real_B_with_grad,
                    fake_B,
                    self.device,
                    use_checkpoint=getattr(
                        self.opt, "use_gradient_checkpointing", False
                    ),
                )

                gp_B = compute_gradient_penalty(
                    self.netD_B,
                    real_A_with_grad,
                    fake_A,
                    self.device,
                    use_checkpoint=getattr(
                        self.opt, "use_gradient_checkpointing", False
                    ),
                )

                # Debug prints for component analysis
                print(
                    f"D_A real: {loss_D_A_real.item():.6f}, fake: {loss_D_A_fake.item():.6f}, gp: {gp_A.item():.6f}"
                )
                print(
                    f"D_B real: {loss_D_B_real.item():.6f}, fake: {loss_D_B_fake.item():.6f}, gp: {gp_B.item():.6f}"
                )

                # Get gradient penalty weight
                lambda_gp = getattr(self.opt, "lambda_gp", 10.0)

                # Final discriminator losses with gradient penalty
                self.loss_D_A = loss_D_A_real + loss_D_A_fake + lambda_gp * gp_A
                self.loss_D_B = loss_D_B_real + loss_D_B_fake + lambda_gp * gp_B
            else:
                # Standard GAN discriminator loss
                self.loss_D_A = discriminator_loss(
                    self.netD_A,
                    self.real_B_detached,
                    fake_B,
                    self.opt,
                    self.criterionGAN,
                )

                self.loss_D_B = discriminator_loss(
                    self.netD_B,
                    self.real_A_detached,
                    fake_A,
                    self.opt,
                    self.criterionGAN,
                )

                if self.opt.use_wasserstein:
                    # Add gradient penalty if specified
                    gp_weight = getattr(self.opt, "gradient_penalty_weight", 10.0)
                    if accumulate_grad:
                        gp_weight /= self.opt.accumulation_steps  # Critical scaling fix

                    gradient_penalty_A = compute_gradient_penalty(
                        self.netD_A,
                        self.real_B_detached,
                        fake_B,
                        self.device,
                        getattr(self.opt, "use_gradient_checkpointing", False),
                    )
                    self.loss_D_A += gp_weight * gradient_penalty_A

                    gradient_penalty_B = compute_gradient_penalty(
                        self.netD_B,
                        self.real_A_detached,
                        fake_A,
                        self.device,
                        getattr(self.opt, "use_gradient_checkpointing", False),
                    )
                    self.loss_D_B += gp_weight * gradient_penalty_B

            # Debug output
            print(
                f"D_A loss: {self.loss_D_A.item():.6f}, D_B loss: {self.loss_D_B.item():.6f}"
            )

            # Combine discriminator losses
            d_loss_total = self.loss_D_A + self.loss_D_B

            # Check for gradient connection
            if not d_loss_total.requires_grad:
                print("WARNING: Discriminator loss has no grad connection")
                grad_reference = torch.tensor(
                    1.0, device=d_loss_total.device, requires_grad=True
                )
                d_loss_total = d_loss_total * grad_reference

            # Backpropagate discriminator losses
            if scaler:
                scaler.scale(d_loss_total).backward()
            else:
                d_loss_total.backward()

        # Return all losses
        loss_dict = self.get_current_losses()

        # Detach intermediate tensors and cleanup
        self._cleanup_tensors()
        # Memory cleanup
        self.cleanup_memory()
        return loss_dict

    def _cleanup_tensors(self):
        """Cleanup intermediate tensors to free memory"""
        tensors_to_clear = [
            "fake_B",
            "fake_A",
            "rec_A",
            "rec_B",
            "idt_A",
            "idt_B",
            "fake_B_detached",
            "fake_A_detached",
            "real_A_detached",
            "real_B_detached",
        ]
        for name in tensors_to_clear:
            setattr(self, name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def optimize_parameters(self, scaler=None):
        """
        Optimize parameters and visualize features if enabled.

        Args:
            scaler: GradScaler for mixed precision training

        Returns:
            Dictionary of loss values
        """
        losses = self.compute_losses(scaler, accumulate_grad=False)

        # Update iteration counter
        if self.visualize_features:
            self.iter_count += 1

            # Visualize features at specified frequency
            if self.iter_count % self.vis_freq == 0:
                self._visualize_network_features()

        return losses

    def _visualize_network_features(self):
        """
        Visualize features in both generators and discriminators using slice-based visualization.
        Creates detailed visualizations of model features for analysis.
        """
        try:
            print("Visualizing network slice features...")
            epoch = self.iter_count // self.vis_freq
            vis_path = os.path.join(self.opt.vis_path, f"epoch_{epoch}")
            os.makedirs(vis_path, exist_ok=True)

            # Use the new SliceFeatureVisualizer
            from visualization.feature_visualizer import SliceFeatureVisualizer

            slice_visualizer = SliceFeatureVisualizer(
                save_path=vis_path,
                max_channels=self.opt.vis_max_channels,
                normalize=True,
                patch_size=(
                    self.opt.patch_size
                    if hasattr(self.opt, "patch_size")
                    else (200, 200, 20)
                ),
                stride_inplane=(
                    self.opt.stride_inplane
                    if hasattr(self.opt, "stride_inplane")
                    else 100
                ),
                stride_layer=(
                    self.opt.stride_layer if hasattr(self.opt, "stride_layer") else 10
                ),
            )

            # Use detached inputs to avoid unnecessary computation graph
            real_A_detached = self.real_A.detach()
            real_B_detached = self.real_B.detach()

            # Choose a middle slice to visualize
            middle_slice_idx = real_A_detached.shape[2] // 2

            # Visualize generator A features for middle slice
            slice_visualizer.visualize_generator_slice(
                self.netG_A,
                real_A_detached,
                middle_slice_idx,
                output_path=os.path.join(
                    vis_path, f"generator_A_slice_{middle_slice_idx}_features.png"
                ),
            )

            # Visualize generator B features for middle slice
            slice_visualizer.visualize_generator_slice(
                self.netG_B,
                real_B_detached,
                middle_slice_idx,
                output_path=os.path.join(
                    vis_path, f"generator_B_slice_{middle_slice_idx}_features.png"
                ),
            )

            # Visualize discriminator features for middle slice
            slice_visualizer.visualize_discriminator_slice(
                self.netD_A,
                real_B_detached,
                middle_slice_idx,
                output_path=os.path.join(
                    vis_path, f"discriminator_A_slice_{middle_slice_idx}_features.png"
                ),
            )

            slice_visualizer.visualize_discriminator_slice(
                self.netD_B,
                real_A_detached,
                middle_slice_idx,
                output_path=os.path.join(
                    vis_path, f"discriminator_B_slice_{middle_slice_idx}_features.png"
                ),
            )

            # Create fake outputs for visualization
            with torch.no_grad():
                fake_B = self.netG_A(real_A_detached)
                fake_A = self.netG_B(real_B_detached)

            # Visualize generator A features on fake B
            slice_visualizer.visualize_generator_slice(
                self.netG_A,
                fake_A,
                middle_slice_idx,
                output_path=os.path.join(
                    vis_path, f"generator_A_fake_slice_{middle_slice_idx}_features.png"
                ),
            )

            # Visualize generator B features on fake A
            slice_visualizer.visualize_generator_slice(
                self.netG_B,
                fake_B,
                middle_slice_idx,
                output_path=os.path.join(
                    vis_path, f"generator_B_fake_slice_{middle_slice_idx}_features.png"
                ),
            )

            # Visualize attention maps for both generators
            visualize_slice_attention_maps(
                self.netG_A,
                real_A_detached,
                middle_slice_idx,
                save_path=os.path.join(
                    vis_path, f"attention_maps_G_A_slice_{middle_slice_idx}.png"
                ),
            )

            visualize_slice_attention_maps(
                self.netG_B,
                real_B_detached,
                middle_slice_idx,
                save_path=os.path.join(
                    vis_path, f"attention_maps_G_B_slice_{middle_slice_idx}.png"
                ),
            )

            # Visualize gradient flow on backward pass
            # This requires a backward pass first
            self.set_requires_grad([self.netD_A, self.netD_B], False)
            fake_B = self.netG_A(real_A_detached)
            loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
            loss_G_A.backward(retain_graph=True)
            visualize_gradient_flow(
                self.netG_A, save_path=os.path.join(vis_path, "gradient_flow_G_A.png")
            )
            self.optimizer_G.zero_grad()

            # Do the same for G_B
            fake_A = self.netG_B(real_B_detached)
            loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)
            loss_G_B.backward(retain_graph=True)
            visualize_gradient_flow(
                self.netG_B, save_path=os.path.join(vis_path, "gradient_flow_G_B.png")
            )
            self.optimizer_G.zero_grad()

            print(f"Feature visualization complete for epoch {epoch}")

        except Exception as e:
            print(f"Error during feature visualization: {e}")
            import traceback

            traceback.print_exc()

    def cleanup_memory(self):
        """Clean up temporary tensors to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tensors_to_clear = ["fake_A", "fake_B", "rec_A", "rec_B", "idt_A", "idt_B"]

        for tensor_name in tensors_to_clear:
            if hasattr(self, tensor_name) and getattr(self, tensor_name) is not None:
                setattr(self, tensor_name, None)

        import gc

        gc.collect()

    def train(self):
        """Make models train mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.train()

    def eval(self):
        """Make models eval mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """
        Override the test method to include feature visualization during testing.
        Performs a full forward pass with memory optimization.
        """
        with torch.no_grad():
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, "net" + name)
                    net.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.fake_B = self.netG_A(self.real_A)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.fake_A = self.netG_B(self.real_B)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.rec_A = self.netG_B(self.fake_B)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.rec_B = self.netG_A(self.fake_A)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.idt_A = self.netG_B(self.real_A)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.idt_B = self.netG_A(self.real_B)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for loss_name in self.loss_names:
                setattr(self, "loss_" + loss_name, 0.0)

            # Visualize features during testing if enabled
            if hasattr(self, "visualize_features") and self.visualize_features:
                test_vis_path = os.path.join(self.opt.vis_path, "test")
                os.makedirs(test_vis_path, exist_ok=True)

                try:
                    # Visualize generator features for test samples
                    self.feature_visualizer.visualize_generator(
                        self.netG_A,
                        self.real_A,
                        output_path=os.path.join(
                            test_vis_path, "test_generator_A_features.png"
                        ),
                    )

                    self.feature_visualizer.visualize_generator(
                        self.netG_B,
                        self.real_B,
                        output_path=os.path.join(
                            test_vis_path, "test_generator_B_features.png"
                        ),
                    )

                    # Visualize attention maps
                    visualize_attention_maps(
                        self.netG_A,
                        self.real_A,
                        save_path=os.path.join(
                            test_vis_path, "test_attention_maps_G_A.png"
                        ),
                    )

                    visualize_attention_maps(
                        self.netG_B,
                        self.real_B,
                        save_path=os.path.join(
                            test_vis_path, "test_attention_maps_G_B.png"
                        ),
                    )

                except Exception as e:
                    print(f"Error during test feature visualization: {e}")
                    import traceback

                    traceback.print_exc()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_bridge_features(self, generator, input_tensor):
        """
        Extract bridge features with improved module detection.
        Used for MMD bridge loss computation.

        Args:
            generator: The generator network
            input_tensor: Input tensor

        Returns:
            List of extracted features
        """
        features = []
        device = input_tensor.device

        def hook_fn(module, input, output):
            # Store a copy of the output tensor
            if isinstance(output, torch.Tensor):
                features.append(output.detach().clone())

        # Improved module detection based on your specific architecture
        target_modules = []

        # Check if we have UNet or UNetWithSTN
        if hasattr(generator, "unet"):
            unet = generator.unet
            if hasattr(unet, "bridge"):
                target_modules.append(("bridge", unet.bridge))
            if hasattr(unet, "enc4"):
                target_modules.append(("enc4", unet.enc4))
        elif isinstance(generator, nn.Module):
            # If generator is directly the UNet
            if hasattr(generator, "bridge"):
                target_modules.append(("bridge", generator.bridge))
            if hasattr(generator, "enc4"):
                target_modules.append(("enc4", generator.enc4))

        # If no standard modules found, try to find any convolutional layers at the bottleneck
        if len(target_modules) == 0:

            # Let's print the model structure to help diagnose
            model_str = str(generator)

            # Function to recursively find bottleneck or middle layers
            def find_bottleneck_modules(module, prefix=""):
                found_modules = []
                for name, child in module.named_children():
                    full_name = f"{prefix}.{name}" if prefix else name

                    # Look for layers that are likely to be bottleneck layers
                    if isinstance(child, nn.Conv3d) and "enc" in full_name:
                        found_modules.append((full_name, child))
                    elif isinstance(child, nn.Sequential) and any(
                        "bottleneck" in str(submodule) or "bridge" in str(submodule)
                        for submodule in child.children()
                    ):
                        found_modules.append((full_name, child))

                    # Recursively search in child modules
                    found_modules.extend(find_bottleneck_modules(child, full_name))
                return found_modules

            bottleneck_modules = find_bottleneck_modules(generator)

            if bottleneck_modules:
                # Use the last few modules as they're likely closer to the bottleneck
                target_modules = bottleneck_modules[-2:]

        # Register hooks
        hooks = []
        for name, module in target_modules:
            hooks.append(module.register_forward_hook(hook_fn))

        # Run forward pass
        with torch.no_grad():
            _ = generator(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process features for MMD
        if features:
            processed = []
            for idx, feat in enumerate(features):
                # Ensure feature is on the correct device and has the right shape
                try:
                    # Global pooling to reduce size
                    feat = F.adaptive_avg_pool3d(feat, (1, 1, 1))
                    feat = feat.view(feat.size(0), -1)

                    # Normalize
                    feat = F.normalize(feat, p=2, dim=1)

                    processed.append(feat)

                except Exception as e:
                    print(f"Error processing feature {idx}: {e}")

            if processed:
                return processed

        # Create random features if nothing was extracted
        print("Using small random features")
        random_features = torch.randn(input_tensor.size(0), 64, device=device) * 0.1
        return [random_features]

    def _compute_bridge_mmd_loss(self, bridge_features, ref_features, device):
        """
        Compute MMD loss for bridge features with robust shape handling.

        Args:
            bridge_features: Features from first network
            ref_features: Features from second network
            device: Device to compute on

        Returns:
            MMD loss value
        """
        try:
            # Ensure tensors are 2D [B, Features]
            bridge_flat = (
                bridge_features.view(bridge_features.size(0), -1)
                if bridge_features.dim() > 2
                else bridge_features
            )
            ref_flat = (
                ref_features.view(ref_features.size(0), -1)
                if ref_features.dim() > 2
                else ref_features
            )

            # Handle size mismatch by truncating to smaller feature size
            min_features = min(bridge_flat.size(1), ref_flat.size(1))
            bridge_flat = bridge_flat[:, :min_features]
            ref_flat = ref_flat[:, :min_features]

            # Ensure sample dimensions match
            min_samples = min(bridge_flat.size(0), ref_flat.size(0))
            bridge_flat = bridge_flat[:min_samples]
            ref_flat = ref_flat[:min_samples]

            # Compute MMD loss
            return compute_patch_mmd_loss(
                bridge_flat, ref_flat, device=device, weight=1.0
            )

        except Exception as e:
            return torch.tensor(0.0, device=device)
