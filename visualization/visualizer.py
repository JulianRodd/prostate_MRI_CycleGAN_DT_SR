import gc
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch
import wandb

from utils.utils import _sanitize_wandb_values


class Visualizer:
    """
    Handles visualization and logging of training metrics and model performance.
    Integrates with Weights & Biases for experiment tracking.
    """

    def __init__(self, opt):
        """
        Initialize the visualizer with experiment options.

        Args:
            opt: Options containing experiment configuration
        """
        self.name = opt.name
        self.opt = opt
        self.run_id = self.name.split("_")[0]
        self.window_size = 50
        self.loss_scales = defaultdict(lambda: deque(maxlen=self.window_size))

        self.saved = False
        self.curves_dir = os.path.join(opt.checkpoints_dir, opt.name, "loss_curves")
        if not os.path.exists(self.curves_dir):
            os.makedirs(self.curves_dir)

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            is_continuation = hasattr(opt, "continue_train") and opt.continue_train
            continuation_status = "CONTINUING" if is_continuation else "NEW"
            epoch_info = f" from epoch {opt.which_epoch}" if is_continuation else ""

            log_file.write(
                f"================ Training Loss ({now}) ================\n"
                f"Run ID: {self.run_id} - {continuation_status} RUN{epoch_info}\n"
                f"Full name: {self.name}\n\n"
            )

        self.epoch_stats = defaultdict(dict)
        self.recent_epochs = []
        self.max_history_epochs = min(getattr(opt, "max_history_epochs", 100), 50)
        self.current_epoch = 0

        self.last_epoch = 0
        self.iterations_since_epoch_start = 0
        self.detected_iterations_per_epoch = None

        self.estimated_iterations = self._estimate_iterations_per_epoch(opt)
        print(f"Estimated iterations per epoch: {self.estimated_iterations}")

        self.current_epoch_values = defaultdict(list)

        self.best_metric = float("-inf")
        self.best_epoch = 0

        self.pending_wandb_logs = {}

        if hasattr(opt, "use_wandb") and opt.use_wandb:
            init_mode = "online"
            is_continuation = hasattr(opt, "continue_train") and opt.continue_train

            if is_continuation:
                prev_wandb_id_file = os.path.join(
                    opt.checkpoints_dir, opt.name, "wandb_id.txt"
                )
                wandb_id = None

                if os.path.exists(prev_wandb_id_file):
                    with open(prev_wandb_id_file, "r") as f:
                        wandb_id = f.read().strip()
                    print(f"Resuming wandb run with ID: {wandb_id}")
                    wandb_resume = "must"
                else:
                    print("No wandb ID found for continuation. Creating new run.")
                    wandb_resume = None
                    wandb_id = None
            else:
                wandb_resume = None
                wandb_id = None
        else:
            init_mode = "disabled"
            wandb_resume = None
            wandb_id = None

        config = {
            "run_id": self.run_id,
            "is_continuation": (
                is_continuation if "is_continuation" in locals() else False
            ),
            "continued_from": (
                opt.which_epoch
                if hasattr(opt, "continue_train") and opt.continue_train
                else None
            ),
            "starting_epoch": opt.epoch_count,
            "which_epoch": (
                opt.which_epoch
                if hasattr(opt, "continue_train") and opt.continue_train
                else None
            ),
        }

        for key, value in vars(opt).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                config[key] = value

        if hasattr(opt, "use_wandb") and opt.use_wandb:
            try:
                wandb.init(
                    project=opt.wandb_project,
                    group=opt.group_name,
                    mode=init_mode,
                    name=self.name,
                    id=wandb_id,
                    resume=wandb_resume,
                    config=config,
                )

                with open(
                    os.path.join(opt.checkpoints_dir, opt.name, "wandb_id.txt"), "w"
                ) as f:
                    f.write(wandb.run.id)
            except Exception as e:
                print(
                    f"Warning: Wandb initialization failed: {e}. Continuing without wandb."
                )
                self.opt.use_wandb = False

    def _estimate_iterations_per_epoch(self, opt):
        """
        Estimate the number of iterations per epoch based on options.

        Args:
            opt: Options containing dataset configuration

        Returns:
            int: Estimated number of iterations per epoch
        """
        patches_per_image = getattr(opt, "patches_per_image", 1)
        batch_size = getattr(opt, "batch_size", 1)
        estimation = (patches_per_image * 49) // batch_size
        return estimation

    def reset(self):
        """Reset the visualizer for a new epoch."""
        self.saved = False
        self.current_epoch_values.clear()
        gc.collect()

    def log_values(self, values_dict, epoch):
        """
        Log values to be used in wandb.

        Args:
            values_dict: Dictionary of values to log
            epoch: Current epoch number
        """
        for k, v in values_dict.items():
            self.pending_wandb_logs[k] = v
        self.pending_wandb_logs["_epoch"] = epoch

    def _update_history(self, epoch, loss_name, value):
        """
        Update historical metrics with current values.

        Args:
            epoch: Current epoch number
            loss_name: Name of the loss/metric
            value: Value of the loss/metric
        """
        self.epoch_stats[loss_name][epoch] = value

        if epoch not in self.recent_epochs:
            self.recent_epochs.append(epoch)
            self.recent_epochs.sort()

            if len(self.recent_epochs) > self.max_history_epochs:
                epochs_to_remove = self.recent_epochs[: -self.max_history_epochs]
                self.recent_epochs = self.recent_epochs[-self.max_history_epochs :]

                for old_epoch in epochs_to_remove:
                    if old_epoch == self.best_epoch:
                        continue

                    for loss_dict in self.epoch_stats.values():
                        if old_epoch in loss_dict and old_epoch != self.best_epoch:
                            del loss_dict[old_epoch]

                gc.collect()

    def get_plot_range(self, values):
        """
        Calculate appropriate plotting range for values.

        Args:
            values: List of values to determine range for

        Returns:
            tuple: (min_val, max_val) range for plotting
        """
        if not values:
            return 1e-10, 1

        positive_values = [v for v in values if v > 0]
        if not positive_values:
            return 1e-10, 1

        min_val = min(positive_values)
        max_val = max(values)

        min_val = max(min_val * 0.1, 1e-10)
        max_val = max_val * 1.5

        return min_val, max_val

    def _calculate_combined_metric(self, losses):
        """
        Calculate a combined quality metric from individual metrics.

        Args:
            losses: Dictionary of loss/metric values

        Returns:
            float: Combined metric value
        """
        domain_metrics = {
            "fid_domain": 0.5,
            "ncc_domain": 0.5,
        }

        sr_metrics = {
            "ssim_sr": 1 / 3,
            "psnr_sr": 1 / 3,
            "lpips_sr": 1 / 3,
        }

        domain_value = 0
        domain_weight = 0

        sr_value = 0
        sr_weight = 0

        for metric_name, weight in domain_metrics.items():
            val_key = f"val_{metric_name}"
            if val_key in losses:
                value = losses[val_key]
                # Convert tensor to CPU if needed before checking with NumPy
                if isinstance(value, torch.Tensor):
                    value_cpu = value.detach().cpu().item()
                else:
                    value_cpu = value

                if not np.isnan(value_cpu) and not np.isinf(value_cpu):
                    if metric_name == "fid_domain":
                        normalized_value = 1.0 / (1.0 + value_cpu / 10.0)
                    elif metric_name == "ncc_domain":
                        normalized_value = (value_cpu + 1.0) / 2.0
                    else:
                        normalized_value = 1.0 / (1.0 + value_cpu)

                    domain_value += weight * normalized_value
                    domain_weight += weight

        for metric_name, weight in sr_metrics.items():
            val_key = f"val_{metric_name}"
            if val_key in losses:
                value = losses[val_key]
                # Convert tensor to CPU if needed before checking with NumPy
                if isinstance(value, torch.Tensor):
                    value_cpu = value.detach().cpu().item()
                else:
                    value_cpu = value

                if not np.isnan(value_cpu) and not np.isinf(value_cpu):
                    if metric_name == "ssim_sr":
                        normalized_value = value_cpu
                    elif metric_name == "psnr_sr":
                        normalized_value = min(max(value_cpu - 20, 0) / 20.0, 1.0)
                    elif metric_name == "lpips_sr":
                        normalized_value = 1.0 - min(value_cpu, 1.0)
                    else:
                        normalized_value = min(value_cpu, 1.0)

                    sr_value += weight * normalized_value
                    sr_weight += weight

        if domain_weight > 0:
            domain_value /= domain_weight

        if sr_weight > 0:
            sr_value /= sr_weight

        if "val_metric_domain" in losses:
            domain_value_tensor = losses["val_metric_domain"]
            if isinstance(domain_value_tensor, torch.Tensor):
                domain_value = domain_value_tensor.detach().cpu().item()
            else:
                domain_value = domain_value_tensor

        if "val_metric_structure" in losses:
            sr_value_tensor = losses["val_metric_structure"]
            if isinstance(sr_value_tensor, torch.Tensor):
                sr_value = sr_value_tensor.detach().cpu().item()
            else:
                sr_value = sr_value_tensor

        if "val_metric_combined" in losses:
            combined_metric_tensor = losses["val_metric_combined"]
            if isinstance(combined_metric_tensor, torch.Tensor):
                return combined_metric_tensor.detach().cpu().item()
            else:
                return combined_metric_tensor

        combined_metric = (domain_value + sr_value) / 2.0
        return combined_metric

    def _calculate_domain_metric(self, losses):
        """
        Calculate domain translation quality metric.

        Args:
            losses: Dictionary of loss/metric values

        Returns:
            float: Domain metric value
        """
        if "val_metric_domain" in losses:
            domain_value = losses["val_metric_domain"]
            if isinstance(domain_value, torch.Tensor):
                return domain_value.detach().cpu().item()
            return domain_value

        domain_metrics = {
            "fid_domain": 1 / 3,
            "is_domain": 1 / 3,
            "kid_domain": 1 / 3,
        }

        metric_value = 0
        total_weight = 0

        for metric_name, weight in domain_metrics.items():
            val_key = f"val_{metric_name}"
            if val_key in losses:
                value = losses[val_key]
                # Convert tensor to CPU if needed
                if isinstance(value, torch.Tensor):
                    value_cpu = value.detach().cpu().item()
                else:
                    value_cpu = value

                if not np.isnan(value_cpu) and not np.isinf(value_cpu):
                    if metric_name == "fid_domain":
                        normalized_value = 1.0 / (1.0 + value_cpu / 10.0)
                    elif metric_name == "is_domain":
                        normalized_value = min(max(value_cpu - 1, 0) / 9.0, 1.0)
                    elif metric_name == "kid_domain":
                        normalized_value = 1.0 - min(value_cpu, 1.0)
                    else:
                        normalized_value = 1.0 / (1.0 + value_cpu)

                    metric_value += weight * normalized_value
                    total_weight += weight

        if total_weight > 0:
            metric_value /= total_weight

        return metric_value

    def _calculate_structure_metric(self, losses):
        """
        Calculate structural preservation quality metric.

        Args:
            losses: Dictionary of loss/metric values

        Returns:
            float: Structure metric value
        """
        if "val_metric_structure" in losses:
            structure_value = losses["val_metric_structure"]
            if isinstance(structure_value, torch.Tensor):
                return structure_value.detach().cpu().item()
            return structure_value

        sr_metrics = {"ssim_sr": 1 / 3, "psnr_sr": 1 / 3, "lpips_sr": 1 / 3}

        metric_value = 0
        total_weight = 0

        for metric_name, weight in sr_metrics.items():
            val_key = f"val_{metric_name}"
            if val_key in losses:
                value = losses[val_key]
                # Convert tensor to CPU if needed
                if isinstance(value, torch.Tensor):
                    value_cpu = value.detach().cpu().item()
                else:
                    value_cpu = value

                if not np.isnan(value_cpu) and not np.isinf(value_cpu):
                    if metric_name == "ssim_sr":
                        normalized_value = value_cpu
                    elif metric_name == "psnr_sr":
                        normalized_value = min(max(value_cpu - 20, 0) / 20.0, 1.0)
                    elif metric_name == "lpips_sr":
                        normalized_value = 1.0 - min(value_cpu, 1.0)
                    else:
                        normalized_value = min(value_cpu, 1.0)

                    metric_value += weight * normalized_value
                    total_weight += weight

        if total_weight > 0:
            metric_value /= total_weight

        return metric_value

    def print_current_losses(self, epoch, iters, losses, t, t_data):
        """
        Print current losses and log to wandb if enabled.

        Args:
            epoch: Current epoch number
            iters: Current iteration number within epoch
            losses: Dictionary of loss values
            t: Time for forward and backward pass
            t_data: Time for data loading
        """
        self.current_epoch = epoch

        if epoch > self.last_epoch:
            if self.iterations_since_epoch_start > 0:
                self.detected_iterations_per_epoch = self.iterations_since_epoch_start
                print(
                    f"Detected {self.detected_iterations_per_epoch} iterations per epoch"
                )
            self.iterations_since_epoch_start = 0
            self.last_epoch = epoch

        if iters > 0:
            self.iterations_since_epoch_start += 1

        message = (
            f"[Run: {self.run_id}] "
            f"(epoch: {epoch}, iters: {iters}, time: {t:.3f}, data: {t_data:.3f}) "
        )

        wandb_enabled = (
            hasattr(self.opt, "use_wandb")
            and self.opt.use_wandb
            and "wandb" in globals()
            and hasattr(wandb, "run")
            and wandb.run is not None
        )

        wandb_log_dict = {}
        if wandb_enabled:
            wandb_log_dict = {
                "epoch": epoch,
                "iters": iters,
            }

            if (
                self.pending_wandb_logs
                and self.pending_wandb_logs.get("_epoch", epoch) == epoch
            ):
                for k, v in self.pending_wandb_logs.items():
                    if k != "_epoch":
                        wandb_log_dict[k] = v
                self.pending_wandb_logs = {}

        is_validation = any(k.startswith("val_") for k in losses.keys())

        if is_validation:
            print(
                f"Processing validation metrics - will log {len(losses)} metrics to wandb"
            )

        training_metrics = [
            "D_A",
            "D_B",
            "G",
            "G_A",
            "G_B",
            "G_A_gan",
            "G_B_gan",
            "cycle_A",
            "cycle_B",
            "feature_matching_A",
            "feature_matching_B",
            "identity_A",
            "identity_B",
        ]

        if is_validation:
            filtered_losses = {}
            excluded_prefixes = [
                "val_vgg_",
                "val_domain_adaptation_",
                "val_da_contrast_",
                "val_da_texture_",
                "val_da_structure_",
            ]

            # Keep only the single FID metric (matching evaluation script)
            fid_metrics_to_keep = ["val_fid_domain"]

            for k, v in losses.items():
                # Skip excluded prefixes with negligible values
                skip = False
                for prefix in excluded_prefixes:
                    if k.startswith(prefix) and abs(v) < 1e-5:
                        skip = True
                        break

                # Handle FID metrics - only keep the main one
                if k.startswith("val_fid_"):
                    if k not in fid_metrics_to_keep:
                        skip = True

                if not skip:
                    filtered_losses[k] = v

            losses = filtered_losses

        if is_validation:
            meaningful_losses = {}
            for k, v in losses.items():
                if not (
                    k.startswith("val_") and k[4:] in training_metrics and abs(v) < 1e-6
                ):
                    meaningful_losses[k] = v

            losses = meaningful_losses

        if is_validation or iters == 0:
            iterations_per_epoch = (
                self.detected_iterations_per_epoch or self.estimated_iterations
            )
            global_step = epoch * iterations_per_epoch
        else:
            iterations_per_epoch = (
                self.detected_iterations_per_epoch or self.estimated_iterations
            )
            global_step = (epoch - 1) * iterations_per_epoch + iters

        if wandb_enabled:
            from utils.utils import WandbStepTracker

            step_tracker = WandbStepTracker.get_instance()
            old_step = global_step
            global_step = step_tracker.get_safe_step(global_step)
            if global_step != old_step:
                print(
                    f"Adjusted wandb step from {old_step} to {global_step} to maintain monotonic increase"
                )

        if not is_validation:
            for k, v in losses.items():
                if isinstance(v, (int, float)):
                    message += f"{k}: {v:.3f} "

                    if iters > 0:
                        self.current_epoch_values[k].append(float(v))

                    if wandb_enabled:
                        wandb_log_dict[k] = v

            if iters == 0 and self.current_epoch_values:
                for k, values in self.current_epoch_values.items():
                    if values:
                        mean_value = np.mean(values)
                        self._update_history(epoch, k, mean_value)
                        if wandb_enabled:
                            wandb_log_dict[f"epoch_{k}"] = mean_value

                self.current_epoch_values.clear()
        else:
            combined_metric = 0

            for k, v in losses.items():
                if k.startswith("val_"):
                    message += f"{k}: {v:.3f} "
                    self._update_history(epoch, k, v)

                    if wandb_enabled:
                        wandb_log_dict[k] = v

            combined_metric = self._calculate_combined_metric(losses)
            domain_metric = self._calculate_domain_metric(losses)
            structure_metric = self._calculate_structure_metric(losses)

            message += f"val_metric_combined: {combined_metric:.3f} "
            message += f"val_metric_domain: {domain_metric:.3f} "
            message += f"val_metric_structure: {structure_metric:.3f} "

            if wandb_enabled:
                wandb_log_dict["val_metric_combined"] = combined_metric
                wandb_log_dict["val_metric_domain"] = domain_metric
                wandb_log_dict["val_metric_structure"] = structure_metric

            if combined_metric > self.best_metric:
                self.best_metric = combined_metric
                self.best_epoch = epoch
                message += "[BEST] "

                if wandb_enabled:
                    wandb_log_dict["best_metric"] = combined_metric
                    wandb_log_dict["best_epoch"] = epoch

        if wandb_enabled and wandb_log_dict:
            try:
                if is_validation:
                    print(f"VALIDATION METRICS TO LOG: {list(wandb_log_dict.keys())}")
                    print(
                        f"Logging validation metrics to wandb with step={global_step}"
                    )

                sanitized_dict = _sanitize_wandb_values(wandb_log_dict)
                wandb.log(sanitized_dict, step=global_step)
            except Exception as e:
                print(f"Warning: wandb logging failed: {e}")
                print(f"Wandb log dictionary contained: {list(wandb_log_dict.keys())}")
                try:
                    sanitized_dict = _sanitize_wandb_values(wandb_log_dict)
                    wandb.log(sanitized_dict)
                    print("Fallback logging without step succeeded")
                except Exception as e2:
                    print(f"Fallback logging also failed: {e2}")

        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write(f"{message}\n")

        gc.collect()
