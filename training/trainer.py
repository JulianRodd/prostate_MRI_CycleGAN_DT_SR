import gc
import itertools
import os
import time
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from loss_functions.discriminator_loss import discriminator_loss

from memory_utils import (
    release_graph_memory,
    aggressive_memory_cleanup,
    analyze_memory,
)
from optimization_utils import (
    optimize_model_for_memory,
    setup_memory_optimizations,
    clip_generator_gradients,
    clip_discriminator_gradients,
)
from training.schedulerers import create_lr_schedulers, LambdaWeightScheduler
from training.validation_handler import ValidationHandler
from utils.utils import WandbStepTracker
from visualization.visualizer import Visualizer

PRINT_MODEL_IMAGES = True


class Trainer:
    """
    Memory-optimized trainer for CycleGAN models.

    Handles the entire training pipeline with specialized memory management
    for medical imaging data, which often has large volumes and high memory
    requirements. Includes mixed precision training, gradient accumulation,
    and adaptive discriminator updates.
    """

    def __init__(self, opt):
        """
        Initialize trainer with memory optimizations.

        Args:
            opt: Options object containing training parameters
        """
        print("Initializing trainer with memory optimizations...")
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.using_cpu = self.device.type == "cpu"

        # Apply memory-optimized configuration before model creation
        self.opt = optimize_model_for_memory(self.opt, self.using_cpu)

        from models.cycle_gan_model import CycleGANModel

        self.model = CycleGANModel()
        self.model.initialize(opt)
        self.model.setup(opt)

        self.validation_handler = ValidationHandler(self.model, self.device)
        self.visualizer = Visualizer(opt)
        self.lambda_scheduler = LambdaWeightScheduler(opt)
        self.lambda_scheduler.apply_to_options(opt)
        self.lr_schedulers = create_lr_schedulers(
            self.model.optimizer_G, self.model.optimizer_D, opt
        )

        self.epoch_start_time = 0
        self.iter_start_time = 0
        self.iter_data_time = 0
        self.total_iter = 0

        # Initialize discriminator update counter
        self.disc_update_counter = 0

        # Initialize separate scalers for mixed precision if needed
        if getattr(opt, "mixed_precision", False) and not self.using_cpu:
            if getattr(opt, "disc_update_freq", 1) > 1:
                self.scaler_G = torch.amp.GradScaler(enabled=True)
                self.scaler_D = torch.amp.GradScaler(enabled=True)
                print("Using separate scalers for G and D with discriminator frequency")
            else:
                self.scaler_G = torch.amp.GradScaler(enabled=True)
                self.scaler_D = torch.amp.GradScaler(enabled=True)
                print("Using separate scalers for G and D for better memory management")
        else:
            self.scaler_G = torch.amp.GradScaler(
                enabled=opt.mixed_precision and not self.using_cpu
            )
            self.scaler_D = torch.amp.GradScaler(
                enabled=opt.mixed_precision and not self.using_cpu
            )

        self.memory_cleanup_freq = 1 if self.using_cpu else 2  # Increased frequency
        setup_memory_optimizations(self.using_cpu)

        if hasattr(opt, "use_precomputed_patches") and opt.use_precomputed_patches:
            print("Using precomputed patches for faster training")
        else:
            print("Using on-the-fly patch generation")

        # Run memory analysis before training
        analyze_memory(self.model)

    def _should_update_discriminator(self, iter_idx, epoch):
        """
        Determine if discriminator should be updated in this iteration.

        Args:
            iter_idx: Current iteration index
            epoch: Current epoch

        Returns:
            Boolean indicating whether to update discriminator
        """
        # Default to updating every iteration
        if not hasattr(self.opt, "disc_update_freq") or self.opt.disc_update_freq <= 1:
            return True

        # Update counter and check if it's time to update
        self.disc_update_counter += 1
        should_update = self.disc_update_counter % self.opt.disc_update_freq == 0

        # For early epochs, update more frequently to establish good gradients
        if epoch < 5 and iter_idx % 2 == 0:
            should_update = True

        if should_update:
            print(
                f"Updating discriminator at step {self.disc_update_counter} (epoch {epoch}, iter {iter_idx})"
            )

        return should_update

    def _compute_discriminator_loss(self, fake_A, fake_B):
        """
        Compute discriminator losses with detached generated images.

        Args:
            fake_A: Fake images from domain A
            fake_B: Fake images from domain B

        Returns:
            Combined discriminator loss
        """
        # Get real images
        real_A = self.model.real_A
        real_B = self.model.real_B

        # Query image pool if available
        if hasattr(self.model, "fake_A_pool"):
            fake_A = self.model.fake_A_pool.query(fake_A)
        if hasattr(self.model, "fake_B_pool"):
            fake_B = self.model.fake_B_pool.query(fake_B)

        # Compute losses
        loss_D_A = discriminator_loss(
            self.model.netD_A,
            real_B,
            fake_B,
            self.opt,
            self.model.criterionGAN,
        )

        loss_D_B = discriminator_loss(
            self.model.netD_B,
            real_A,
            fake_A,
            self.opt,
            self.model.criterionGAN,
        )

        # Return combined loss
        return loss_D_A + loss_D_B

    def _extract_disc_losses(self, fake_A, fake_B):
        """
        Extract individual discriminator losses for reporting.

        Args:
            fake_A: Fake images from domain A
            fake_B: Fake images from domain B

        Returns:
            Tuple of (loss_D_A, loss_D_B)
        """
        real_A = self.model.real_A
        real_B = self.model.real_B

        # Query image pool if available
        if hasattr(self.model, "fake_A_pool"):
            fake_A = self.model.fake_A_pool.query(fake_A)
        if hasattr(self.model, "fake_B_pool"):
            fake_B = self.model.fake_B_pool.query(fake_B)

        # Compute individual losses
        loss_D_A = discriminator_loss(
            self.model.netD_A,
            real_B,
            fake_B,
            self.opt,
            self.model.criterionGAN,
        )

        loss_D_B = discriminator_loss(
            self.model.netD_B,
            real_A,
            fake_A,
            self.opt,
            self.model.criterionGAN,
        )

        return loss_D_A, loss_D_B

    def train_epoch(self, train_loader, epoch: int):
        """
        Train the model for a single epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Epoch start time
        """
        # Configuration
        accumulation_steps = getattr(self.opt, "accumulation_steps", 1)
        using_mixed_precision = (
            getattr(self.opt, "mixed_precision", False) and not self.using_cpu
        )

        # Reset the counter only at the beginning of training
        if (
            epoch == self.opt.epoch_count
            and getattr(self, "disc_update_counter", 0) == 0
        ):
            self.disc_update_counter = 0
            print(f"Initialized discriminator update counter at epoch {epoch}")

        # Setup for training
        self.epoch_start_time = time.time()
        self.iter_data_time = time.time()
        self.model.train()

        # Pre-emptively clear memory
        aggressive_memory_cleanup(self.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Metrics tracking
        peak_memory = 0
        epoch_losses = defaultdict(list)
        optimization_step = 0

        # Zero gradients at beginning of epoch
        self.model.optimizer_G.zero_grad(set_to_none=True)
        self.model.optimizer_D.zero_grad(set_to_none=True)

        # Training loop
        description = f"Epoch {epoch}" + (
            " (CPU Mode)" if self.using_cpu else " (GPU Mode)"
        )
        for i, data in enumerate(tqdm(train_loader, desc=description)):
            # Regular memory cleanup
            if i % self.memory_cleanup_freq == 0:
                aggressive_memory_cleanup(self.model)

            # Calculate time spent loading data
            self.iter_start_time = time.time()
            data_time = self.iter_start_time - self.iter_data_time

            try:
                # Process inputs without storing computation graph
                with torch.no_grad():
                    self.model.set_input(data)
                data = None  # Free data reference

                # Determine if discriminator should be updated
                update_disc = self._should_update_discriminator(i, epoch)
                self.model.current_update_disc = update_disc

                # Autocast context for mixed precision
                autocast_context = (
                    torch.cuda.amp.autocast()
                    if using_mixed_precision
                    else nullcontext()
                )

                # Forward pass with autocast
                with autocast_context:
                    # Forward pass
                    self.model.forward()

                    # Generator update (freeze discriminator during G update)
                    self.model.set_requires_grad(
                        [self.model.netD_A, self.model.netD_B], False
                    )
                    g_loss = self.model._compute_generator_losses()

                # Scale and backward for generator
                if using_mixed_precision:
                    self.scaler_G.scale(g_loss).backward()
                else:
                    g_loss.backward()

                # Check gradient norms for debugging
                g_total_norm = 0
                g_valid_count = 0
                for param in itertools.chain(
                    self.model.netG_A.parameters(),
                    self.model.netG_B.parameters(),
                ):
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        g_total_norm += param_norm.item() ** 2
                        g_valid_count += 1

                if g_valid_count > 0:
                    g_total_norm = g_total_norm ** (1.0 / 2)
                    if (
                        g_total_norm > 1000
                        or np.isnan(g_total_norm)
                        or np.isinf(g_total_norm)
                    ):
                        print(
                            f"Warning: Generator gradient norm very high or invalid: {g_total_norm}, applying clipping"
                        )
                        clip_generator_gradients(self.model.netG_A, self.model.netG_B)

                # Step generator optimizer if it's time or we're at the end of the loader
                generator_update_step = (i + 1) % accumulation_steps == 0 or (
                    i + 1 == len(train_loader)
                )
                if generator_update_step:
                    # Verify generator gradients are valid
                    has_valid_grads = False
                    for param in itertools.chain(
                        self.model.netG_A.parameters(),
                        self.model.netG_B.parameters(),
                    ):
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            has_valid_grads = True
                            break

                    if has_valid_grads:
                        if using_mixed_precision:
                            # Apply scaled optimizer step
                            self.scaler_G.step(self.model.optimizer_G)
                            self.scaler_G.update()
                        else:
                            self.model.optimizer_G.step()
                        print(
                            f"Updated generator at step {self.total_iter} (epoch {epoch}, iter {i})"
                        )
                        optimization_step += 1
                    else:
                        print(
                            "Warning: No valid gradients for generator - skipping update"
                        )

                    # Zero generator gradients after step
                    self.model.optimizer_G.zero_grad(set_to_none=True)

                # Discriminator update if scheduled
                if update_disc:
                    # Unfreeze discriminator
                    self.model.set_requires_grad(
                        [self.model.netD_A, self.model.netD_B], True
                    )

                    # Prepare discriminator inputs (detached)
                    fake_B = self.model.fake_B.detach()
                    fake_A = self.model.fake_A.detach()

                    with autocast_context:
                        d_loss = self._compute_discriminator_loss(fake_A, fake_B)

                    # Extract individual discriminator losses
                    if hasattr(self.model, "loss_D_A") and hasattr(
                        self.model, "loss_D_B"
                    ):
                        # Extract individual discriminator losses
                        loss_D_A, loss_D_B = self._extract_disc_losses(fake_A, fake_B)
                        self.model.loss_D_A = (
                            loss_D_A.item()
                            if isinstance(loss_D_A, torch.Tensor)
                            else loss_D_A
                        )
                        self.model.loss_D_B = (
                            loss_D_B.item()
                            if isinstance(loss_D_B, torch.Tensor)
                            else loss_D_B
                        )

                    # Scale and backward for discriminator
                    if using_mixed_precision:
                        self.scaler_D.scale(d_loss).backward()
                    else:
                        d_loss.backward()

                    # Check discriminator gradient norms
                    d_total_norm = 0
                    d_valid_count = 0
                    for param in itertools.chain(
                        self.model.netD_A.parameters(),
                        self.model.netD_B.parameters(),
                    ):
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            d_total_norm += param_norm.item() ** 2
                            d_valid_count += 1

                    if d_valid_count > 0:
                        d_total_norm = d_total_norm ** (1.0 / 2)
                        if (
                            d_total_norm > 100
                            or np.isnan(d_total_norm)
                            or np.isinf(d_total_norm)
                        ):
                            print(
                                f"Warning: Discriminator gradient norm very high: {d_total_norm}, applying clipping"
                            )
                            clip_discriminator_gradients(
                                self.model.netD_A, self.model.netD_B
                            )

                    # Step discriminator optimizer
                    if using_mixed_precision:
                        self.scaler_D.step(self.model.optimizer_D)
                        self.scaler_D.update()
                    else:
                        self.model.optimizer_D.step()

                    print(f"Updated discriminator at step {self.disc_update_counter}")

                    # Zero discriminator gradients after step
                    self.model.optimizer_D.zero_grad(set_to_none=True)

                # Get all current losses for logging
                loss_dict = self.model.get_current_losses()

                # Add generator and discriminator losses specifically
                if "G" not in loss_dict and hasattr(g_loss, "item"):
                    loss_dict["G"] = g_loss.item()

                if update_disc and "D" not in loss_dict and hasattr(d_loss, "item"):
                    loss_dict["D"] = d_loss.item()

                # Record epoch losses for averaging
                for k, v in loss_dict.items():
                    if isinstance(v, (int, float)):
                        epoch_losses[k].append(float(v))
                    elif isinstance(v, torch.Tensor):
                        epoch_losses[k].append(float(v.item()))

                # Print loss values periodically
                if i < 10 or i % 50 == 0 or i == len(train_loader) - 1:
                    print("\nCurrent losses:")
                    for k, v in loss_dict.items():
                        print(f"  {k}: {v:.6f}")

                # Log losses and metrics
                should_log = (i < 10) or (i % 5 == 0) or (i == len(train_loader) - 1)
                if should_log:
                    t_comp = time.time() - self.iter_start_time

                    # Log to wandb if enabled
                    if hasattr(self.opt, "use_wandb") and self.opt.use_wandb:
                        try:
                            import wandb

                            if wandb.run is not None:
                                log_dict = {k: v for k, v in loss_dict.items()}
                                log_dict["epoch"] = epoch
                                log_dict["iteration"] = i
                                log_dict["total_iter"] = self.total_iter
                                wandb.log(log_dict)
                                print("Successfully logged to wandb")
                        except Exception as e:
                            print(f"Wandb logging failed: {e}")

                    # Log through visualizer
                    try:
                        self.visualizer.print_current_losses(
                            epoch, self.total_iter, loss_dict, t_comp, data_time
                        )
                    except Exception as e:
                        print(f"Error in visualizer: {e}")

                # Track and report memory usage
                if not self.using_cpu and torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024**2
                    peak_memory = max(peak_memory, current_memory)
                    if i % 10 == 0:
                        print(
                            f"Memory usage: {current_memory:.2f}MB, Peak: {peak_memory:.2f}MB"
                        )

                # Release memory for next iteration
                release_graph_memory(self.model)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"WARNING: OOM in iteration {i}. Skipping batch and cleaning memory..."
                    )
                    if not self.using_cpu and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    # Log OOM incident
                    if hasattr(self.opt, "use_wandb") and self.opt.use_wandb:
                        try:
                            import wandb

                            if wandb.run is not None:
                                wandb.log(
                                    {"OOM_incident": 1, "epoch": epoch, "iteration": i}
                                )
                        except:
                            pass
                    continue
                else:
                    print(f"Error in iteration {i}: {str(e)}")
                    raise e

            # Increment iteration counter and update time
            self.total_iter += self.opt.batch_size
            self.iter_data_time = time.time()

        # End of epoch summary
        if epoch_losses:
            print("\n" + "=" * 40)
            print(f"EPOCH {epoch} SUMMARY:")

            epoch_summary = {}
            for k, values in epoch_losses.items():
                if values:
                    avg_value = sum(values) / len(values)
                    epoch_summary[k] = avg_value
                    print(f"  Average {k}: {avg_value:.4f}")

            self.visualizer.print_current_losses(epoch, 0, epoch_summary, 0, 0)

            # Log epoch summary to wandb
            if hasattr(self.opt, "use_wandb") and self.opt.use_wandb:
                try:
                    import wandb

                    if wandb.run is not None:
                        epoch_log = {f"epoch_{k}": v for k, v in epoch_summary.items()}
                        epoch_log["epoch"] = epoch
                        epoch_log["peak_memory_mb"] = peak_memory
                        wandb.log(epoch_log)
                except Exception as e:
                    print(f"Error logging epoch summary to wandb: {e}")

            print("=" * 40)

        print(f"Epoch {epoch}: Performed {optimization_step} optimization steps")

        # Final memory cleanup
        aggressive_memory_cleanup(self.model)
        return self.epoch_start_time

    def _check_available_memory(self):
        """
        Check if enough GPU memory is available for validation.

        Returns:
            Boolean indicating if sufficient memory is available
        """
        if torch.cuda.is_available():
            free_memory = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            )
            free_memory_gb = free_memory / (1024**3)
            print(f"Available GPU memory: {free_memory_gb:.2f}GB")

            if free_memory_gb < 2:
                return False
        return True

    def _update_learning_rates(self, epoch):
        """
        Update learning rates from schedulers.

        Args:
            epoch: Current epoch number
        """
        old_lr_G = self.model.optimizer_G.param_groups[0]["lr"]
        old_lr_D = self.model.optimizer_D.param_groups[0]["lr"]

        self.lr_schedulers["scheduler_G"].step()
        self.lr_schedulers["scheduler_D"].step()

        new_lr_G = self.model.optimizer_G.param_groups[0]["lr"]
        new_lr_D = self.model.optimizer_D.param_groups[0]["lr"]

        print(
            f"Learning rates: G: {old_lr_G:.7f} → {new_lr_G:.7f}, D: {old_lr_D:.7f} → {new_lr_D:.7f}"
        )

        self.visualizer.log_values(
            {"lr_generator": new_lr_G, "lr_discriminator": new_lr_D}, epoch
        )

    def train(self, train_loader, val_loader=None):
        """
        Main training loop with memory optimization.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
        """
        # Print dataset information
        if hasattr(train_loader.dataset, "manifest_path"):
            print(
                f"Using precomputed patches from: {train_loader.dataset.manifest_path}"
            )
            print(f"Total training patches: {len(train_loader.dataset)}")

        if val_loader is not None and hasattr(val_loader.dataset, "manifest_path"):
            print(
                f"Using precomputed validation patches from: {val_loader.dataset.manifest_path}"
            )
            print(f"Total validation patches: {len(val_loader.dataset)}")

        # Set total iterations for continued training
        if hasattr(self.opt, "continue_train") and self.opt.continue_train:
            completed_epochs = self.opt.epoch_count - 1
            dataloader_len = len(train_loader)
            self.total_iter = completed_epochs * dataloader_len
            print(
                f"Setting total iterations to {self.total_iter} for continued training"
            )

        # Configure training parameters
        total_epochs = self.opt.niter + self.opt.niter_decay
        final_epoch = self.opt.epoch_count + total_epochs - 1
        is_continuation = self.opt.continue_train

        print(
            f"{'Continuing' if is_continuation else 'Starting'} training from epoch "
            f"{self.opt.epoch_count} to {final_epoch}"
        )

        # Configure validation frequency
        validate_freq = self.opt.run_validation_interval

        # Analyze memory before training
        if torch.cuda.is_available():
            analyze_memory(self.model)

        # Main training loop
        for epoch in range(self.opt.epoch_count, final_epoch + 1):
            try:
                # Update lambda weights for loss components
                updated_weights = self.lambda_scheduler.update(epoch)
                self.lambda_scheduler.apply_to_options(self.opt)

                # Log lambda weights
                lambda_log = {
                    f"lambda_{k.replace('lambda_', '')}": v
                    for k, v in updated_weights.items()
                }
                self.visualizer.log_values(lambda_log, epoch)

                # Print lambda weights periodically
                if epoch % 10 == 0:
                    print(f"Epoch {epoch} lambda weights:")
                    for k, v in updated_weights.items():
                        print(f"  {k}: {v:.4f}")

                # Train single epoch
                epoch_start_time = self.train_epoch(train_loader, epoch)

                # Validation if scheduled
                should_validate = val_loader is not None and (
                    epoch == self.opt.epoch_count
                    or epoch % validate_freq == 0
                    or epoch == final_epoch
                )

                if should_validate:
                    # Clear memory before validation
                    aggressive_memory_cleanup(self.model)

                    print("\nRunning validation...")
                    val_metrics = self.validate(val_loader, epoch)

                    if val_metrics:
                        print("Validation metrics:")
                        self.visualizer.print_current_losses(
                            epoch, 0, val_metrics, 0, 0
                        )
                    else:
                        print("Warning: Validation returned no metrics")

                # Save checkpoints
                save_freq = self.opt.save_epoch_freq * (2 if self.using_cpu else 1)
                if epoch % save_freq == 0 or epoch == final_epoch:
                    print(f"\nSaving model checkpoint at epoch {epoch}")
                    self.model.save_networks(str(epoch))

                # Always save latest for recovery
                self.model.save_networks("latest")

                # Update learning rates
                self._update_learning_rates(epoch)

                # Print epoch summary
                self._print_epoch_summary(epoch, final_epoch, epoch_start_time)

                # Final memory cleanup
                aggressive_memory_cleanup(self.model)

            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                # Save latest model before aborting
                self.model.save_networks("latest")
                # Log error to wandb if enabled
                if hasattr(self.opt, "use_wandb") and self.opt.use_wandb:
                    try:
                        import wandb

                        if wandb.run is not None:
                            wandb.log({"training_error": str(e), "epoch": epoch})
                    except:
                        pass
                raise

        print("\nTraining completed successfully!")
        self.model.save_networks("latest")

    def validate(self, val_loader, epoch: int = 0):
        """
        Memory-optimized validation procedure.

        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()

        # Reset metrics
        val_losses = defaultdict(float)
        val_metrics = defaultdict(float)
        valid_samples = 0

        # Prepare validation handler
        if hasattr(self.validation_handler, "downscale_factor"):
            self.validation_handler.downscale_factor = 1.0

        # Check if running evaluation-style metrics
        eval_style_enabled = False
        if hasattr(self.opt, "run_eval_style_metrics"):
            eval_style_enabled = self.opt.run_eval_style_metrics

        # Determine if evaluation metrics should be reported this epoch
        eval_style_report = False
        if (
            eval_style_enabled
            and hasattr(self.opt, "eval_style_metrics_epoch")
            and epoch >= self.opt.eval_style_metrics_epoch
            and hasattr(self.opt, "eval_style_report_interval")
            and epoch % self.opt.eval_style_report_interval == 0
        ):
            eval_style_report = True
            print(f"\n{'-' * 50}")
            print(f"Will calculate evaluation-style FID metrics at epoch {epoch}")
            print(f"{'-' * 50}")

        # Setup validation plotting
        max_full_plots = getattr(self.opt, "max_full_plots", 3)
        dataset_prefix = getattr(self.opt, "dataset_prefix", self.opt.name[:3].upper())
        if hasattr(self.validation_handler, "setup_full_validation_plotting"):
            self.validation_handler.setup_full_validation_plotting(
                max_plots=max_full_plots,
                epoch=epoch,
                run_name=self.opt.name,
                dataset_prefix=dataset_prefix,
            )
            print(f"Will generate up to {max_full_plots} full validation image plots")

        # Check memory availability
        low_memory = not self._check_available_memory()
        if low_memory:
            print("Low memory detected - reducing validation scope")
            if hasattr(self.validation_handler, "enable_full_metrics"):
                self.validation_handler.enable_full_metrics = False
                print("Disabled full metrics calculation")

            if hasattr(self.validation_handler, "enable_full_dataset_fid"):
                if hasattr(self.validation_handler, "_collect_images_for_fid"):
                    setattr(self.validation_handler, "_collection_factor", 0.25)
                    print("Using reduced resolution (25%) for FID calculation")

            visualize_this_epoch = False
        else:
            # Determine if visualizations should be generated
            is_using_full_images = getattr(self.opt, "use_full_validation", False)
            visualize_this_epoch = False
            if not is_using_full_images and PRINT_MODEL_IMAGES:
                if epoch <= 5 or epoch % 5 == 0:
                    visualize_this_epoch = True
                    print("Will generate model visualization images for this epoch")
                if is_using_full_images:
                    print(
                        "Using full image validation - skipping redundant visualization"
                    )

        # Create visualization directory if needed
        image_folder = os.path.join("model_images", self.opt.name)
        if not os.path.exists(image_folder) and visualize_this_epoch:
            try:
                os.makedirs(image_folder, exist_ok=True)
                print(f"Created visualization directory: {image_folder}")
            except Exception as e:
                print(f"Warning: Could not create visualization directory: {e}")

        # Determine number of validation samples
        if torch.cuda.is_available():
            # For GPU, use more samples but be mindful of memory
            max_val_samples = min(
                128,  # Reduced from 256 to prevent OOM
                len(val_loader)
                // (getattr(val_loader.dataset, "patches_per_image", 1) or 1),
            )
        else:
            # For CPU, use minimal samples
            if hasattr(val_loader.dataset, "manifest_path"):
                available_samples = len(val_loader)
            else:
                patches_per_image = (
                    getattr(val_loader.dataset, "patches_per_image", 1) or 1
                )
                available_samples = len(val_loader) // patches_per_image

            max_val_samples = min(5, max(2, available_samples))
            print(f"CPU Mode: Using {max_val_samples} validation samples")

        print(f"Will validate on {max_val_samples} samples")
        processed_image_indices = set()

        # Determine patches per image
        patches_per_image = 1
        if not hasattr(val_loader.dataset, "manifest_path"):
            patches_per_image = getattr(val_loader.dataset, "patches_per_image", 1)

        # Reset peak memory statistics
        if not self.using_cpu and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / (1024**2)
            print(f"Starting validation with {start_mem:.2f} MB GPU memory in use")

        consecutive_failures = 0
        max_failures = 3
        plotted_image = False

        # Reset FID collectors
        if hasattr(self.validation_handler, "reset_collectors"):
            self.validation_handler.reset_collectors()
            if eval_style_enabled:
                print("Reset image collectors for evaluation-style FID calculation")

        # Process validation samples
        print(f"\nProcessing validation samples (target: {max_val_samples})...")
        for i, data in enumerate(
            tqdm(val_loader, desc="Validation", total=max_val_samples)
        ):
            # Stop if we've processed enough samples
            if len(processed_image_indices) >= max_val_samples:
                print(
                    f"Reached target of {max_val_samples} samples, stopping validation"
                )
                break

            # Track processed images
            if hasattr(val_loader.dataset, "manifest_path"):
                image_index = i
            else:
                image_index = i // patches_per_image

            if image_index in processed_image_indices:
                print(
                    f"Skipping patch {i % patches_per_image + 1} from already processed image {image_index}"
                )
                continue

            processed_image_indices.add(image_index)
            print(
                f"Processing sample {len(processed_image_indices)}/{max_val_samples} (image index: {image_index})"
            )

            # Generate visualization plots if needed
            if visualize_this_epoch and not plotted_image:
                try:
                    # Create visualizations for validation data
                    self._generate_validation_visualization(data, epoch, image_folder)
                    plotted_image = True
                except Exception as e:
                    print(f"Warning: Could not plot model images: {e}")

            # Clean memory before processing each sample
            aggressive_memory_cleanup(self.model)

            # Print input data shapes for debugging
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                print(
                    f"Original validation data shapes: A={data[0].shape}, B={data[1].shape}"
                )
                min_dim_A = min(data[0].shape[2:]) if data[0].dim() >= 4 else 0
                min_dim_B = min(data[1].shape[2:]) if data[1].dim() >= 4 else 0
                print(f"Minimum dimensions: A={min_dim_A}, B={min_dim_B}")

            try:
                # Run validation on current batch
                losses, metrics, is_valid = self.validation_handler.validate_batch(data)

                if not is_valid:
                    print(f"Skipping invalid validation sample {i}")
                    consecutive_failures += 1

                    # Adaptive validation scope reduction
                    if consecutive_failures >= max_failures:
                        print(
                            f"Too many consecutive failures ({max_failures}), reducing validation scope"
                        )
                        if (
                            hasattr(self.validation_handler, "enable_full_metrics")
                            and self.validation_handler.enable_full_metrics
                        ):
                            self.validation_handler.enable_full_metrics = False
                            print("Disabling full metrics calculation")
                            consecutive_failures = 0

                    aggressive_memory_cleanup(self.model)
                    continue

                # Reset failure counter on success
                consecutive_failures = 0

                # Accumulate loss values
                for k, v in losses.items():
                    if isinstance(v, (int, float)):
                        val_value = v
                    elif isinstance(v, torch.Tensor):
                        val_value = v.item()
                    else:
                        continue

                    if not np.isnan(val_value) and not np.isinf(val_value):
                        val_losses[f"val_{k}"] += val_value

                # Accumulate metric values
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        val_value = v.item()
                    else:
                        val_value = v

                    if not np.isnan(val_value) and not np.isinf(val_value):
                        val_metrics[f"val_{k}"] += val_value

                valid_samples += 1
                print(
                    f"Completed validation sample {len(processed_image_indices)}/{max_val_samples} successfully"
                )

                # Clean up model memory
                if hasattr(self.model, "cleanup_memory"):
                    self.model.cleanup_memory()

                # Clear intermediate tensors
                for tensor_name in ["fake_A", "fake_B", "rec_A", "rec_B"]:
                    if hasattr(self.model, tensor_name):
                        setattr(self.model, tensor_name, None)

                # More aggressive memory cleanup
                aggressive_memory_cleanup(self.model)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM error in validation sample {i}: {e}")
                    consecutive_failures += 1

                    # Progressive validation scope reduction on OOM
                    if (
                        hasattr(self.validation_handler, "enable_full_metrics")
                        and self.validation_handler.enable_full_metrics
                    ):
                        self.validation_handler.enable_full_metrics = False
                        print("Disabling full metrics calculation due to OOM")
                        consecutive_failures = 0
                    else:
                        print(f"Cannot recover from OOM, skipping sample {i}")

                    # Aggressive cleanup
                    aggressive_memory_cleanup(self.model)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Error in validation sample {i}: {str(e)}")
                    continue
            except Exception as e:
                print(f"Error in validation sample {i}: {str(e)}")
                aggressive_memory_cleanup(self.model)
                continue

        # Calculate full-dataset FID scores if enabled
        self._calculate_fid_metrics(val_metrics, eval_style_report)

        # Process and report validation results
        if valid_samples > 0:
            # Average accumulated metrics
            for k in val_losses.keys():
                val_losses[k] /= valid_samples
            for k in val_metrics.keys():
                if not k.startswith("val_fid_") and not k.startswith("val_eval_style_"):
                    val_metrics[k] /= valid_samples

            # Print validation results
            self._print_validation_results(
                val_losses, val_metrics, valid_samples, max_val_samples
            )

            # Log to wandb if enabled
            if hasattr(self.opt, "use_wandb") and self.opt.use_wandb:
                self._log_validation_to_wandb(val_losses, val_metrics, epoch)
        else:
            print("Warning: No valid samples in validation set")

        # Final memory cleanup
        aggressive_memory_cleanup(self.model)
        return {**val_losses, **val_metrics}

    def _generate_validation_visualization(self, data, epoch, image_folder):
        """
        Generate visualization images from validation data.

        Args:
            data: Validation data batch
            epoch: Current epoch number
            image_folder: Directory to save visualizations
        """
        print(f"Plotting model images for epoch {epoch}...")
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            print("Creating visualization at original resolution")

            # Handle 3D vs 2D data
            if data[0].dim() == 5:  # 3D data
                from visualization.plot_model_images import plot_model_images

                plot_model_images(self.model, data, epoch, self.opt.name)
            else:  # 2D data
                from visualization.plot_model_images import plot_model_images

                plot_model_images(self.model, data, epoch, self.opt.name)

            print(f"Saved model images to {image_folder}/epoch_{epoch}.png")

            # Force matplotlib cleanup
            try:
                import matplotlib.pyplot as plt

                plt.close("all")
            except:
                pass

    def _calculate_fid_metrics(self, val_metrics, eval_style_report):
        """
        Calculate and add FID metrics to validation results.

        Args:
            val_metrics: Dictionary to add FID metrics to
            eval_style_report: Boolean indicating if evaluation-style FID should be reported
        """
        if (
            hasattr(self.validation_handler, "enable_full_dataset_fid")
            and self.validation_handler.enable_full_dataset_fid
        ):
            print("\n===== Calculating full-dataset FID scores =====")
            try:
                # Calculate standard FID
                fid_metrics = self.validation_handler.calculate_full_dataset_fid()

                # Add calculated metrics
                for k, v in fid_metrics.items():
                    val_metrics[f"val_{k}"] = v
                print(f"Added full-dataset FID scores: {fid_metrics}")

                # Handle evaluation-style FID reporting
                if eval_style_report:
                    # Check if we have enough slices for reliable FID
                    min_slices_required = 50
                    if (
                        hasattr(self.validation_handler, "real_B_collector")
                        and hasattr(self.validation_handler, "fake_B_collector")
                        and len(self.validation_handler.real_B_collector)
                        >= min_slices_required
                        and len(self.validation_handler.fake_B_collector)
                        >= min_slices_required
                    ):
                        print(f"\n{'-' * 50}")
                        print(
                            f"Evaluation-Style FID Summary (matches post-training evaluation):"
                        )
                        print(
                            f"  - Real slices: {len(self.validation_handler.real_B_collector)}"
                        )
                        print(
                            f"  - Fake slices: {len(self.validation_handler.fake_B_collector)}"
                        )

                        # Report FID value
                        fid_value = fid_metrics.get("fid_domain", float("inf"))
                        print(f"  - FID score: {fid_value:.4f}")

                        # Report smoothed FID if available
                        if "fid_domain_smoothed" in fid_metrics:
                            smoothed_fid = fid_metrics.get(
                                "fid_domain_smoothed", float("inf")
                            )
                            print(f"  - Smoothed FID: {smoothed_fid:.4f}")

                        # Add evaluation-style FID metric
                        val_metrics["val_eval_style_fid"] = fid_value
                        print(f"{'-' * 50}\n")
                    else:
                        print(f"\nNot enough slices for reliable evaluation-style FID.")
                        print(f"Need at least {min_slices_required}, but have:")
                        print(
                            f"  - Real slices: {len(getattr(self.validation_handler, 'real_B_collector', []))}"
                        )
                        print(
                            f"  - Fake slices: {len(getattr(self.validation_handler, 'fake_B_collector', []))}"
                        )
                        val_metrics["val_eval_style_fid"] = float("inf")
            except Exception as e:
                print(f"Error calculating full-dataset FID: {e}")
                import traceback

                traceback.print_exc()

    def _print_validation_results(
        self, val_losses, val_metrics, valid_samples, max_val_samples
    ):
        """
        Print validation results with metrics grouped by category.

        Args:
            val_losses: Dictionary of validation losses
            val_metrics: Dictionary of validation metrics
            valid_samples: Number of valid samples processed
            max_val_samples: Maximum number of validation samples
        """
        print("\n=========== VALIDATION RESULTS ===========")
        print(f"Completed {valid_samples}/{max_val_samples} valid samples")

        # Define metric categories to exclude and group
        model_training_losses = [
            "val_D_A",
            "val_D_B",
            "val_G",
            "val_G_A",
            "val_G_B",
            "val_G_A_gan",
            "val_G_B_gan",
            "val_cycle_A",
            "val_cycle_B",
            "val_feature_matching_A",
            "val_feature_matching_B",
            "val_identity_A",
            "val_identity_B",
        ]
        excluded_metrics = model_training_losses + [
            "val_mse_domain_a",
            "val_mse_domain_b",
            "val_psnr_domain_a",
            "val_psnr_domain_b",
        ]

        # Filter and group metrics
        filtered_losses = {
            k: v for k, v in val_losses.items() if k not in excluded_metrics
        }
        filtered_metrics = {
            k: v
            for k, v in val_metrics.items()
            if k not in excluded_metrics
            and not k.startswith("val_mse_")
            and not k.startswith("val_psnr_")
        }

        all_metrics = {**filtered_losses, **filtered_metrics}

        # Define metric groups
        grouped_metrics = {
            "Domain Metrics": {},
            "Structure Metrics": {},
            "MI Metrics": {},
            "EPI Metrics": {},
            "SSIM Metrics": {},
            "NCC Metrics": {},
            "HFEN Metrics": {},
            "FID Metrics": {},
            "Combined Metrics": {},
            "Other Metrics": {},
        }

        # Group metrics by type
        for k, v in all_metrics.items():
            if k.startswith("val_mi_domain"):
                grouped_metrics["MI Metrics"][k] = v
            elif k.startswith("val_mi_structure"):
                grouped_metrics["MI Metrics"][k] = v
            elif k.startswith("val_epi_domain"):
                grouped_metrics["EPI Metrics"][k] = v
            elif k.startswith("val_epi_structure"):
                grouped_metrics["EPI Metrics"][k] = v
            elif k.startswith("val_ssim_domain"):
                grouped_metrics["SSIM Metrics"][k] = v
            elif k.startswith("val_ssim_structure"):
                grouped_metrics["SSIM Metrics"][k] = v
            elif k.startswith("val_ncc_"):
                grouped_metrics["NCC Metrics"][k] = v
            elif k.startswith("val_hfen_"):
                grouped_metrics["HFEN Metrics"][k] = v
            elif k.startswith("val_fid_") or k.startswith("val_eval_style_"):
                grouped_metrics["FID Metrics"][k] = v
            elif k.startswith("val_structural_similarity"):
                grouped_metrics["Combined Metrics"][k] = v
            elif k.startswith("val_mse_") or k.startswith("val_psnr_"):
                grouped_metrics["Domain Metrics"][k] = v
            elif "domain" in k:
                grouped_metrics["Domain Metrics"][k] = v
            elif "structure" in k:
                grouped_metrics["Structure Metrics"][k] = v
            else:
                grouped_metrics["Other Metrics"][k] = v

        # Print metrics by group
        for group_name, metrics_dict in grouped_metrics.items():
            if metrics_dict:
                print(f"\n{group_name}:")
                for metric, value in sorted(metrics_dict.items()):
                    print(f"  {metric}: {value:.4f}")

        # Print summary of key metrics
        print("\nKey Metrics Summary:")
        summary_metrics = [
            "val_ssim_domain_a",
            "val_ssim_domain_b",
            "val_mi_domain_a",
            "val_mi_domain_b",
            "val_epi_domain_a",
            "val_epi_domain_b",
            "val_fid_domain",
            "val_eval_style_fid",
        ]

        for metric in summary_metrics:
            if metric in all_metrics:
                print(f"  {metric}: {all_metrics[metric]:.4f}")

    def _log_validation_to_wandb(self, val_losses, val_metrics, epoch):
        """
        Log validation results to wandb.

        Args:
            val_losses: Dictionary of validation losses
            val_metrics: Dictionary of validation metrics
            epoch: Current epoch number
        """
        try:
            import wandb

            if wandb.run is not None:
                print("Directly logging validation metrics to wandb...")
                wandb_val_dict = {
                    k: v for k, v in {**val_losses, **val_metrics}.items()
                }
                wandb_val_dict["epoch"] = epoch

                # Calculate step for consistent logging
                iterations_per_epoch = getattr(
                    self.visualizer,
                    "detected_iterations_per_epoch",
                    getattr(self.visualizer, "estimated_iterations", 50),
                )

                proposed_step = epoch * iterations_per_epoch
                step_tracker = WandbStepTracker.get_instance()
                global_step = step_tracker.get_safe_step(proposed_step)

                print(
                    f"Logging validation metrics to wandb with global_step={global_step}"
                )
                wandb.log(wandb_val_dict, step=global_step)
                print(
                    f"Successfully logged {len(wandb_val_dict)} validation metrics to wandb"
                )
        except Exception as e:
            print(f"Error during direct wandb logging: {e}")

    def _print_epoch_summary(self, epoch, final_epoch, epoch_start_time):
        """
        Print training epoch summary.

        Args:
            epoch: Current epoch number
            final_epoch: Final epoch number
            epoch_start_time: Epoch start time
        """
        lr_G = self.model.optimizer_G.param_groups[0]["lr"]
        lr_D = self.model.optimizer_D.param_groups[0]["lr"]

        print(f"\nEnd of epoch {epoch}/{final_epoch}")
        print(f"Time taken: {time.time() - epoch_start_time:.2f} sec")
        print(
            f"Current learning rates - Generator: {lr_G:.7f}, Discriminator: {lr_D:.7f}"
        )

        # Print memory usage
        if torch.cuda.is_available():
            allocated_mem = torch.cuda.memory_allocated() / (1024**3)
            reserved_mem = torch.cuda.memory_reserved() / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            print(
                f"GPU memory - Allocated: {allocated_mem:.2f}GB, Reserved: {reserved_mem:.2f}GB, Peak: {max_allocated:.2f}GB"
            )

    def _get_current_lr(self) -> float:
        """
        Get current learning rate for reporting.

        Returns:
            Current learning rate
        """
        return self.model.optimizers[0].param_groups[0]["lr"]

    def _print_loss_stats(self, loss_dict):
        """
        Print statistics about the losses.

        Args:
            loss_dict: Dictionary of losses
        """
        print("\n--- Loss Statistics ---")
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.4f}")
        print("----------------------\n")

    def _print_gpu_stats(self):
        """
        Print GPU memory usage statistics.
        """
        if not self.using_cpu and torch.cuda.is_available():
            print("\n--- GPU Memory Statistics ---")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
            print(f"  Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")
            print(
                f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.1f} MB"
            )

            # Print memory by model component
            models = {
                "G_A": self.model.netG_A,
                "G_B": self.model.netG_B,
                "D_A": self.model.netD_A,
                "D_B": self.model.netD_B,
            }

            # Estimate model memory usage
            for name, model in models.items():
                param_size = sum(
                    p.numel() * p.element_size() for p in model.parameters()
                )
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                size_mb = (param_size + buffer_size) / 1024**2
                print(f"  {name} size: {size_mb:.1f} MB")

            print("----------------------------\n")

    def _save_checkpoints(self, epoch: int, final_epoch: int) -> None:
        """
        Save model checkpoints.

        Args:
            epoch: Current epoch number
            final_epoch: Final epoch number
        """
        if epoch % self.opt.save_epoch_freq == 0 or epoch == final_epoch:
            print(f"\nSaving model checkpoint at epoch {epoch}")
            self.model.save_networks(str(epoch))

        self.model.save_networks("latest")
