import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdaptationLoss(nn.Module):
    """
    Specialized loss for medical image domain adaptation between unregistered in-vivo and ex-vivo MRI.
    Robust implementation that handles mixed precision training and misaligned images.

    This loss combines multiple components:
    - Histogram matching loss: Aligns intensity distributions
    - Contrast adaptation loss: Matches contrast characteristics
    - Structural similarity loss: Preserves structural information
    - Gradient loss: Ensures boundary consistency
    - Normalized cross correlation loss: Maintains correlation between domains
    - Texture matching loss: Preserves important texture features

    Args:
        device: Device to run computations on
        histogram_weight: Weight for histogram matching component
        contrast_weight: Weight for contrast adaptation component
        structure_weight: Weight for structural similarity component
        gradient_weight: Weight for gradient matching component
        ncc_weight: Weight for normalized cross correlation component
        texture_weight: Weight for texture matching component
        return_components: Whether to return individual loss components
    """

    def __init__(
        self,
        device=None,
        histogram_weight=1,
        contrast_weight=1,
        structure_weight=1,
        gradient_weight=1,
        ncc_weight=1,
        texture_weight=1,
        return_components=True,
    ):
        super(DomainAdaptationLoss, self).__init__()
        self.device = device
        self.return_components = return_components

        self.histogram_weight = histogram_weight
        self.contrast_weight = contrast_weight
        self.structure_weight = structure_weight
        self.gradient_weight = gradient_weight
        self.ncc_weight = ncc_weight
        self.texture_weight = texture_weight

        self.max_component_value = 10.0

        self.register_buffer("sobel_x", self._create_sobel_kernel("x"))
        self.register_buffer("sobel_y", self._create_sobel_kernel("y"))

        if device:
            self.dummy = torch.ones(1, device=device)
            self.to(device)

    def _create_sobel_kernel(self, direction):
        """Create sobel kernel for edge detection"""
        if direction == "x":
            kernel = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            )
        else:
            kernel = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            )
        return kernel.view(1, 1, 3, 3)

    def forward(self, pred, target):
        """
        Calculate domain adaptation loss with extensive NaN checking.
        Returns total loss and individual components if return_components is True.

        Args:
            pred: Predicted/generated image
            target: Target image from other domain

        Returns:
            torch.Tensor: Total loss
            dict (optional): Individual loss components if return_components=True
        """
        if pred is None or target is None:
            if self.return_components:
                return torch.tensor(0.0, device=self.device), {}
            return torch.tensor(0.0, device=self.device)

        min_size = min(min(pred.shape[2:]), min(target.shape[2:]))
        if min_size < 3:
            print(f"Warning: Input tensor too small ({min_size}), minimum size is 3")
            if self.return_components:
                return torch.tensor(0.1, device=self.device), {}
            return torch.tensor(0.1, device=self.device)

        pred = self._ensure_float32(pred)
        target = self._ensure_float32(target)

        if pred.shape != target.shape:
            print(
                f"Warning: Shape mismatch - pred: {pred.shape}, target: {target.shape}"
            )
            try:
                target = F.interpolate(
                    target,
                    size=pred.shape[2:],
                    mode="bilinear" if len(pred.shape) == 4 else "trilinear",
                    align_corners=False,
                )
            except Exception as e:
                print(f"Resize failed: {e}")
                if self.return_components:
                    return torch.tensor(0.1, device=self.device), {}
                return torch.tensor(0.1, device=self.device)

        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.all(pred == pred[0, 0, 0, 0]) or torch.all(
            target == target[0, 0, 0, 0]
        ):
            print("Warning: Zero variance in input tensors")
            if self.return_components:
                return torch.tensor(0.1, device=self.device), {}
            return torch.tensor(0.1, device=self.device)

        pred = self._safe_normalize_01(pred)
        target = self._safe_normalize_01(target)

        try:
            if pred.dim() == 5:
                return self._process_3d_volume(pred, target)
            else:
                return self._calculate_domain_loss(pred, target)
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            import traceback

            traceback.print_exc()
            if self.return_components:
                return torch.tensor(0.1, device=self.device), {}
            return torch.tensor(0.1, device=self.device)

    def _safe_normalize_01(self, x):
        """Safely normalize tensor to [0,1] range without introducing NaN"""
        batch_size, channels = x.shape[0], x.shape[1]
        x_flat = x.view(batch_size, channels, -1)

        x_min = x_flat.min(dim=2, keepdim=True)[0]
        x_max = x_flat.max(dim=2, keepdim=True)[0]

        diff = x_max - x_min
        diff_too_small = diff < 1e-6

        x_normalized = torch.zeros_like(x)

        for b in range(batch_size):
            for c in range(channels):
                if not diff_too_small[b, c]:
                    norm_factor = diff[b, c]
                    offset = x_min[b, c]
                    channel = x[b, c].clone()
                    channel = (channel - offset) / norm_factor
                    x_normalized[b, c] = torch.clamp(channel, 0.0, 1.0)
                else:
                    x_normalized[b, c] = torch.ones_like(x[b, c]) * 0.5

        return x_normalized

    def _ensure_float32(self, tensor):
        """Ensure tensor is float32 for numerical stability"""
        if tensor.dtype != torch.float32:
            return tensor.to(torch.float32)
        return tensor

    def _process_3d_volume(self, pred, target):
        """
        Process 3D volumes by operating on multiple key slices with robust NaN protection.
        Returns a stable average loss across slices, handling potential NaN values safely.

        Args:
            pred: Predicted 3D volume [B,C,D,H,W]
            target: Target 3D volume [B,C,D,H,W]

        Returns:
            torch.Tensor: Volume loss (average of slice losses)
            dict (optional): Individual loss components if return_components=True
        """
        b, c, d, h, w = pred.shape

        if d < 3 or h < 3 or w < 3:
            print(
                f"Warning: 3D volume too small ({d}x{h}x{w}), needs minimum of 3 in each dimension"
            )
            if self.return_components:
                return torch.tensor(0.1, device=self.device), {}
            return torch.tensor(0.1, device=self.device)

        slice_indices = [d // 4, d // 2, 3 * d // 4] if d >= 8 else [d // 2]

        valid_losses = []
        valid_weights = []
        all_components = {}

        for idx in slice_indices:
            if idx < d:
                try:
                    pred_slice = pred[:, :, idx].clone()
                    target_slice = target[:, :, idx].clone()

                    if torch.isnan(pred_slice).any() or torch.isnan(target_slice).any():
                        print(f"Warning: NaN values in slice {idx}, skipping")
                        continue

                    if self.return_components:
                        slice_loss, components = self._calculate_domain_loss(
                            pred_slice, target_slice
                        )
                        for k, v in components.items():
                            if k not in all_components:
                                all_components[k] = []
                            all_components[k].append(
                                v * (1.5 if idx == d // 2 else 1.0)
                            )
                    else:
                        slice_loss = self._calculate_domain_loss(
                            pred_slice, target_slice
                        )

                    if not torch.isnan(slice_loss) and not torch.isinf(slice_loss):
                        weight = 1.5 if idx == d // 2 else 1.0
                        valid_losses.append(slice_loss * weight)
                        valid_weights.append(weight)
                except Exception as e:
                    print(f"Error processing slice {idx}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

        if len(valid_losses) > 0:
            total_weight = sum(valid_weights)
            if total_weight > 0:
                stacked_losses = torch.stack(valid_losses)
                weighted_loss = torch.sum(stacked_losses) / total_weight

                weighted_loss = torch.clamp(weighted_loss, 0.0, 100.0)

                if self.return_components and all_components:
                    avg_components = {}
                    for k, v_list in all_components.items():
                        if v_list:
                            avg_components[k] = sum(v_list) / total_weight
                    return weighted_loss, avg_components

                return weighted_loss

        print(
            "Falling back to middle slice only due to errors in multi-slice processing"
        )
        try:
            middle_slice = d // 2
            pred_middle = pred[:, :, middle_slice]
            target_middle = target[:, :, middle_slice]
            return self._calculate_domain_loss(pred_middle, target_middle)
        except Exception as e:
            print(f"Error in fallback slice processing: {e}")
            import traceback

            traceback.print_exc()
            if self.return_components:
                return torch.tensor(0.01, device=self.device), {}
            return torch.tensor(0.01, device=self.device)

    def _calculate_domain_loss(self, pred, target):
        """
        Calculate domain adaptation loss components with extreme NaN protection.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            torch.Tensor: Total loss
            dict (optional): Individual loss components if return_components=True
        """
        try:
            if torch.isnan(pred).any() or torch.isnan(target).any():
                print("Warning: NaN detected in inputs to domain loss calculation")
                if self.return_components:
                    return torch.tensor(0.01, device=self.device), {}
                return torch.tensor(0.01, device=self.device)

            try:
                histogram_loss = self._histogram_matching_loss(pred, target)
                histogram_loss = torch.clamp(
                    torch.nan_to_num(histogram_loss, nan=0.01),
                    0.0,
                    self.max_component_value,
                )
            except Exception as e:
                print(f"Error in histogram loss: {e}")
                histogram_loss = torch.tensor(0.01, device=self.device)

            try:
                texture_loss = self._texture_matching_loss(pred, target)
                texture_loss = torch.clamp(
                    torch.nan_to_num(texture_loss, nan=0.01),
                    0.0,
                    self.max_component_value,
                )
            except Exception as e:
                print(f"Error in texture loss: {e}")
                texture_loss = torch.tensor(0.01, device=self.device)

            try:
                contrast_loss = self._contrast_adaptation_loss(pred, target)
                contrast_loss = torch.clamp(
                    torch.nan_to_num(contrast_loss, nan=0.01),
                    0.0,
                    self.max_component_value,
                )
            except Exception as e:
                print(f"Error in contrast loss: {e}")
                contrast_loss = torch.tensor(0.01, device=self.device)

            try:
                structure_loss = self._structural_similarity_loss(pred, target)
                structure_loss = torch.clamp(
                    torch.nan_to_num(structure_loss, nan=0.01),
                    0.0,
                    self.max_component_value,
                )
            except Exception as e:
                print(f"Error in structure loss: {e}")
                structure_loss = torch.tensor(0.01, device=self.device)

            try:
                gradient_loss = self._gradient_loss(pred, target)
                gradient_loss = torch.clamp(
                    torch.nan_to_num(gradient_loss, nan=0.01),
                    0.0,
                    self.max_component_value,
                )
            except Exception as e:
                print(f"Error in gradient loss: {e}")
                gradient_loss = torch.tensor(0.01, device=self.device)

            try:
                ncc_loss = self._normalized_cross_correlation_loss(pred, target)
                ncc_loss = torch.clamp(
                    torch.nan_to_num(ncc_loss, nan=0.01), 0.0, self.max_component_value
                )
            except Exception as e:
                print(f"Error in NCC loss: {e}")
                ncc_loss = torch.tensor(0.01, device=self.device)

            weighted_components = {
                "histogram": self.histogram_weight * histogram_loss,
                "texture": self.texture_weight * texture_loss,
                "contrast": self.contrast_weight * contrast_loss,
                "structure": self.structure_weight * structure_loss,
                "gradient": self.gradient_weight * gradient_loss,
                "ncc": self.ncc_weight * ncc_loss,
            }

            valid_components = {}
            for name, comp in weighted_components.items():
                if not torch.isnan(comp) and not torch.isinf(comp):
                    valid_components[name] = comp

            if not valid_components:
                if self.return_components:
                    return torch.tensor(0.01, device=self.device), {}
                return torch.tensor(0.01, device=self.device)

            total_loss = sum(valid_components.values())

            if total_loss < 1e-6:
                total_loss = torch.tensor(0.01, device=self.device)

            total_loss = torch.clamp(total_loss, 0.0, 100.0)

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if self.return_components:
                    return torch.tensor(0.01, device=self.device), {}
                return torch.tensor(0.01, device=self.device)

            if self.return_components:
                component_values = {k: v.item() for k, v in valid_components.items()}
                return total_loss, component_values

            return total_loss

        except Exception as e:
            print(f"Exception in domain loss calculation: {e}")
            import traceback

            traceback.print_exc()
            if self.return_components:
                return torch.tensor(0.01, device=self.device), {}
            return torch.tensor(0.01, device=self.device)

    def _histogram_matching_loss(self, pred, target):
        """
        Calculate histogram matching loss to align intensity distributions
        without requiring spatial alignment.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            torch.Tensor: Histogram matching loss
        """
        try:
            b, c, h, w = pred.shape

            if h * w < 100:
                return torch.tensor(0.01, device=self.device)

            hist_loss = 0
            for i in range(b):
                for j in range(c):
                    pred_channel = pred[i, j].flatten()
                    target_channel = target[i, j].flatten()

                    if torch.allclose(pred_channel, pred_channel[0]) or torch.allclose(
                        target_channel, target_channel[0]
                    ):
                        hist_loss += torch.tensor(0.5, device=self.device)
                        continue

                    bins = min(20, max(5, int(math.sqrt(h * w) / 5)))

                    pred_hist = torch.histc(pred_channel, bins=bins, min=0, max=1)
                    target_hist = torch.histc(target_channel, bins=bins, min=0, max=1)

                    pred_sum = pred_hist.sum()
                    target_sum = target_hist.sum()

                    if pred_sum > 0 and target_sum > 0:
                        pred_hist = pred_hist / (pred_sum + 1e-8)
                        target_hist = target_hist / (target_sum + 1e-8)
                        channel_loss = F.l1_loss(pred_hist, target_hist)
                    else:
                        channel_loss = torch.tensor(0.5, device=self.device)

                    hist_loss += channel_loss

            hist_loss = hist_loss / (b * c)
            return hist_loss

        except Exception as e:
            print(f"Exception in histogram loss: {e}")
            import traceback

            traceback.print_exc()
            return torch.tensor(0.01, device=self.device)

    def _texture_matching_loss(self, pred, target):
        """
        Match texture statistics across domains without requiring pixel alignment.
        Uses simple statistical moments of filter responses.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            torch.Tensor: Texture matching loss
        """
        try:
            if pred.size(2) < 3 or pred.size(3) < 3:
                return torch.tensor(0.01, device=self.device)

            pred_gradx = F.conv2d(pred, self.sobel_x, padding=1)
            pred_grady = F.conv2d(pred, self.sobel_y, padding=1)

            target_gradx = F.conv2d(target, self.sobel_x, padding=1)
            target_grady = F.conv2d(target, self.sobel_y, padding=1)

            pred_grad_mag = torch.sqrt(pred_gradx**2 + pred_grady**2 + 1e-8)
            target_grad_mag = torch.sqrt(target_gradx**2 + target_grady**2 + 1e-8)

            try:
                loss_mean = F.l1_loss(
                    pred_grad_mag.mean(dim=[2, 3]), target_grad_mag.mean(dim=[2, 3])
                )
            except Exception:
                loss_mean = torch.tensor(0.01, device=self.device)

            try:
                loss_var = F.l1_loss(
                    pred_grad_mag.var(dim=[2, 3]), target_grad_mag.var(dim=[2, 3])
                )
            except Exception:
                loss_var = torch.tensor(0.01, device=self.device)

            try:
                pred_skew = self._compute_skewness(pred_grad_mag)
                target_skew = self._compute_skewness(target_grad_mag)
                loss_skew = F.l1_loss(pred_skew, target_skew)
            except Exception:
                loss_skew = torch.tensor(0.01, device=self.device)

            texture_loss = (
                torch.clamp(loss_mean, 0, 10)
                + torch.clamp(loss_var, 0, 10)
                + 0.5 * torch.clamp(loss_skew, 0, 10)
            )
            return texture_loss

        except Exception as e:
            print(f"Exception in texture loss: {e}")
            import traceback

            traceback.print_exc()
            return torch.tensor(0.01, device=self.device)

    def _compute_skewness(self, x):
        """
        Compute skewness of a tensor along spatial dimensions with safeguards.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Skewness measure
        """
        try:
            mean = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            std = torch.sqrt(var + 1e-8)

            std_flat = std.squeeze()
            mask = std_flat > 1e-6

            if not mask.any():
                return torch.zeros_like(std_flat)

            diff_cubed = (x - mean) ** 3
            mean_diff_cubed = diff_cubed.mean(dim=[2, 3])

            safe_std_cube = std_flat**3 + 1e-8
            skewness = mean_diff_cubed / safe_std_cube

            skewness = skewness * mask.float()

            skewness = torch.clamp(skewness, -10, 10)

            return skewness

        except Exception as e:
            print(f"Error in skewness computation: {e}")
            return torch.zeros_like(x[:, :, 0, 0])

    def _contrast_adaptation_loss(self, pred, target):
        """
        Ultra-stable contrast adaptation loss.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            torch.Tensor: Contrast adaptation loss
        """
        try:
            pred_mean = pred.mean(dim=[2, 3], keepdim=True)
            target_mean = target.mean(dim=[2, 3], keepdim=True)

            pred_std = torch.sqrt(
                ((pred - pred_mean) ** 2).mean(dim=[2, 3], keepdim=True) + 1e-5
            )
            target_std = torch.sqrt(
                ((target - target_mean) ** 2).mean(dim=[2, 3], keepdim=True) + 1e-5
            )

            mean_loss = F.l1_loss(pred_mean, target_mean)
            std_loss = F.l1_loss(pred_std, target_std)

            loss = torch.clamp(mean_loss, 0, 5) + torch.clamp(std_loss, 0, 5)

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.01, device=self.device)

            return loss

        except Exception as e:
            print(f"Exception in contrast loss: {e}")
            import traceback

            traceback.print_exc()
            return torch.tensor(0.01, device=self.device)

    def _structural_similarity_loss(self, pred, target):
        """
        Ultra-simplified structural similarity loss.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            torch.Tensor: Structural similarity loss
        """
        try:
            if pred.size(2) < 3 or pred.size(3) < 3:
                return torch.tensor(0.01, device=self.device)

            pred_mean = pred.mean(dim=[2, 3], keepdim=True)
            target_mean = target.mean(dim=[2, 3], keepdim=True)

            eps = 1e-6
            pred_var = ((pred - pred_mean) ** 2).mean(dim=[2, 3], keepdim=True) + eps
            target_var = ((target - target_mean) ** 2).mean(
                dim=[2, 3], keepdim=True
            ) + eps

            pred_std = torch.sqrt(pred_var)
            target_std = torch.sqrt(target_var)

            pred_norm = (pred - pred_mean) / pred_std
            target_norm = (target - target_mean) / target_std

            pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=0.0, neginf=0.0)
            target_norm = torch.nan_to_num(target_norm, nan=0.0, posinf=0.0, neginf=0.0)

            pred_norm = torch.clamp(pred_norm, -5, 5)
            target_norm = torch.clamp(target_norm, -5, 5)

            loss = F.l1_loss(pred_norm, target_norm)

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.01, device=self.device)

            return torch.clamp(loss, 0, 10)

        except Exception as e:
            print(f"Exception in structure loss: {e}")
            import traceback

            traceback.print_exc()
            return torch.tensor(0.01, device=self.device)

    def _gradient_loss(self, pred, target):
        """
        Simplified gradient loss for stability.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            torch.Tensor: Gradient matching loss
        """
        try:
            if pred.size(2) < 3 or pred.size(3) < 3:
                return torch.tensor(0.01, device=self.device)

            pred_grads = self._safe_gradients(pred)
            target_grads = self._safe_gradients(target)

            if pred_grads is None or target_grads is None:
                return torch.tensor(0.01, device=self.device)

            pred_grads = torch.nan_to_num(pred_grads, nan=0.0, posinf=1.0, neginf=-1.0)
            target_grads = torch.nan_to_num(
                target_grads, nan=0.0, posinf=1.0, neginf=-1.0
            )

            pred_grads = torch.clamp(pred_grads, 0, 10)
            target_grads = torch.clamp(target_grads, 0, 10)

            loss = F.l1_loss(pred_grads, target_grads)

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.01, device=self.device)

            return torch.clamp(loss, 0, 10)

        except Exception as e:
            print(f"Exception in gradient loss: {e}")
            import traceback

            traceback.print_exc()
            return torch.tensor(0.01, device=self.device)

    def _safe_gradients(self, x):
        """
        Safely calculate gradient magnitudes with error handling.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor or None: Gradient magnitudes or None if calculation fails
        """
        try:
            if x.size(2) < 3 or x.size(3) < 3:
                return None

            if x.shape[1] > 1:
                grads = []
                for c in range(x.shape[1]):
                    gx = F.conv2d(x[:, c : c + 1], self.sobel_x, padding=1)
                    gy = F.conv2d(x[:, c : c + 1], self.sobel_y, padding=1)
                    g_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
                    g_mag = torch.nan_to_num(g_mag, nan=0.0, posinf=1.0, neginf=0.0)
                    grads.append(g_mag)
                return torch.cat(grads, dim=1)
            else:
                gx = F.conv2d(x, self.sobel_x, padding=1)
                gy = F.conv2d(x, self.sobel_y, padding=1)
                g_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
                return torch.nan_to_num(g_mag, nan=0.0, posinf=1.0, neginf=0.0)
        except Exception as e:
            print(f"Error calculating gradients: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _normalized_cross_correlation_loss(self, pred, target):
        """
        Ultra-simplified NCC loss with safety measures.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            torch.Tensor: Normalized cross correlation loss
        """
        try:
            if pred.size(2) < 3 or pred.size(3) < 3:
                return torch.tensor(0.01, device=self.device)

            b, c, h, w = pred.shape

            pred_flat = pred.view(b, c, -1)
            target_flat = target.view(b, c, -1)

            pred_mean = pred_flat.mean(dim=2, keepdim=True)
            target_mean = target_flat.mean(dim=2, keepdim=True)

            pred_centered = pred_flat - pred_mean
            target_centered = target_flat - target_mean

            eps = 1e-8
            pred_norm_factor = torch.sqrt(
                (pred_centered**2).sum(dim=2, keepdim=True) + eps
            )
            target_norm_factor = torch.sqrt(
                (target_centered**2).sum(dim=2, keepdim=True) + eps
            )

            pred_norm = torch.zeros_like(pred_centered)
            target_norm = torch.zeros_like(target_centered)

            for b_idx in range(b):
                for c_idx in range(c):
                    if pred_norm_factor[b_idx, c_idx] > 1e-6:
                        pred_norm[b_idx, c_idx] = (
                            pred_centered[b_idx, c_idx] / pred_norm_factor[b_idx, c_idx]
                        )
                    if target_norm_factor[b_idx, c_idx] > 1e-6:
                        target_norm[b_idx, c_idx] = (
                            target_centered[b_idx, c_idx]
                            / target_norm_factor[b_idx, c_idx]
                        )

            correlation = (pred_norm * target_norm).sum(dim=2)
            correlation = torch.clamp(correlation, -1.0, 1.0)

            loss = 1.0 - correlation.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.01, device=self.device)

            return torch.clamp(loss, 0, 1)

        except Exception as e:
            print(f"Exception in NCC loss: {e}")
            import traceback

            traceback.print_exc()
            return torch.tensor(0.01, device=self.device)
