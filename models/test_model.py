import models.generator
import torch

import models.utils.model_utils
from .base_model import BaseModel
from .cycle_gan_model import CycleGANModel
from .utils.cycle_gan_utils import apply_spectral_norm


class TestModel(BaseModel):
    """
    TestModel is used for testing a pre-trained CycleGAN model.
    This model loads generators from a trained CycleGAN model and
    performs forward passes for inference without training capabilities.
    """

    def name(self):
        return "TestModel"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Configures the test model options.
        Ensures the model is in test mode and configures appropriate defaults.

        Args:
            parser: Command line argument parser
            is_train: Verification flag (must be False for TestModel)

        Returns:
            Modified parser with test-specific settings
        """
        assert not is_train, "TestModel cannot be used in train mode"
        parser = CycleGANModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode="single")
        parser.add_argument("--model_suffix", type=str, default="")
        return parser

    def initialize(self, opt):
        """
        Initialize the test model with options.
        Sets up the generator for inference.

        Args:
            opt: Command line options
        """
        assert not opt.isTrain
        BaseModel.initialize(self, opt)
        self._init_test_attributes(opt)
        self._create_generator(opt)

    def _init_test_attributes(self, opt):
        """
        Initialize test-specific attributes based on translation direction.

        Args:
            opt: Command line options
        """
        self.loss_names = []
        is_b_to_a = hasattr(opt, "which_direction") and opt.which_direction == "BtoA"

        self.model_names = ["G_B" if is_b_to_a else "G_A"]
        self.visual_names = ["real_B", "fake_A"] if is_b_to_a else ["real_A", "fake_B"]

        self.input_nc = opt.output_nc if is_b_to_a else opt.input_nc
        self.output_nc = opt.input_nc if is_b_to_a else opt.output_nc

    def _create_generator(self, opt):
        """
        Create generator with optional spectral normalization.
        Configures GPU usage and moves the model to the appropriate device.

        Args:
            opt: Command line options
        """
        try:
            self._check_gpu_memory()

            # Modified call to define_generator removing the netG parameter
            self.netG = models.utils.model_utils.define_generator(
                self.input_nc,
                self.output_nc,
                opt.ngf,
                opt.norm,
                not opt.no_dropout,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
                use_stn=getattr(
                    opt, "use_stn", True
                ),  # Default to True if not specified
                pretrained_path=getattr(opt, "pretrained_path_generator", None),
                patch_size=getattr(opt, "patch_size", None),
            )

            # Add this block to apply spectral normalization
            if hasattr(opt, "use_spectral_norm_G") and opt.use_spectral_norm_G:
                self.netG = apply_spectral_norm(self.netG)
                print("Applied spectral normalization to generator")

            self._move_to_device()
            self._set_generator_attribute(opt)

        except Exception as e:
            print(f"Error initializing generator: {str(e)}")
            raise

    def _check_gpu_memory(self):
        """
        Print GPU memory statistics for debugging purposes.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            props = torch.cuda.get_device_properties(0)
            print(f"GPU memory available: {props.total_memory / 1e9:.2f} GB")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    def _move_to_device(self):
        """
        Move the generator to the appropriate device (GPU or CPU).
        Wraps with DataParallel if multiple GPUs are available.
        """
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.netG.to(self.gpu_ids[0])
            if len(self.gpu_ids) > 1:
                self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

    def _set_generator_attribute(self, opt):
        """
        Set the appropriate generator attribute based on the translation direction.

        Args:
            opt: Command line options
        """
        attr_name = "netG_B" if opt.which_direction == "BtoA" else "netG_A"
        setattr(self, attr_name, self.netG)

    def set_input(self, input):
        """
        Set input tensor for the model.
        Handles input validation and moves data to the appropriate device.

        Args:
            input: Input tensor
        """
        try:
            if not isinstance(input, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(input)}")

            if self.opt.which_direction == "BtoA":
                self.real_B = input.to(self.device)
                self.real = self.real_B
            else:
                self.real_A = input.to(self.device)
                self.real = self.real_A

        except Exception as e:
            print(f"Error in set_input: {str(e)}")
            raise

    def forward(self):
        """
        Forward pass for inference.
        Uses mixed precision when enabled for better memory efficiency.
        """
        try:
            device_type = "cuda" if self.opt.gpu_ids else "cpu"
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=device_type, enabled=self.opt.mixed_precision
                ):
                    if self.opt.which_direction == "BtoA":
                        self.fake_A = self.netG(self.real_B)
                        self.fake = self.fake_A
                    else:
                        self.fake_B = self.netG(self.real_A)
                        self.fake = self.fake_B
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise

    def cleanup(self):
        """
        Release GPU memory by deleting the generator and emptying CUDA cache.
        """
        if hasattr(self, "netG"):
            del self.netG
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
