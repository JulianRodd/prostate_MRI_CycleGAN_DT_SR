import gc
import traceback
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from metrics.do_metrics import DomainMetricsCalculator
from metrics.sr_metrics import PSNR, SSIM, LPIPS


class MetricNames:
    """
    Container class for metric name constants.

    Defines standard metric names and provides methods to retrieve metric groups.
    """

    SSIM_SR = "ssim_sr"
    PSNR_SR = "psnr_sr"
    LPIPS_SR = "lpips_sr"
    FID_DOMAIN = "fid_domain"
    NCC_DOMAIN = "ncc_domain"
    METRIC_DOMAIN = "metric_domain"
    METRIC_STRUCTURE = "metric_structure"
    METRIC_COMBINED = "metric_combined"

    @classmethod
    def get_all(cls):
        """Return all metric names"""
        return [
            cls.SSIM_SR,
            cls.PSNR_SR,
            cls.LPIPS_SR,
            cls.FID_DOMAIN,
            cls.NCC_DOMAIN,
            cls.METRIC_DOMAIN,
            cls.METRIC_STRUCTURE,
            cls.METRIC_COMBINED,
        ]

    @classmethod
    def get_higher_better(cls):
        """Return metrics where higher values are better"""
        return [
            cls.SSIM_SR,
            cls.PSNR_SR,
            cls.NCC_DOMAIN,
        ]

    @classmethod
    def get_lower_better(cls):
        """Return metrics where lower values are better"""
        return [
            cls.LPIPS_SR,
            cls.FID_DOMAIN,
        ]


class ImageNormalizer:
    """
    Utility class for normalizing images to specific ranges.

    Handles min-max normalization for consistent metric calculation.
    """

    def __init__(self):
        self.reference_stats = {}

    def normalize_to_range(self, image, min_val=0.0, max_val=1.0):
        """
        Normalize image to specified range.

        Args:
            image (torch.Tensor): Input image
            min_val (float): Minimum target value. Default: 0.0
            max_val (float): Maximum target value. Default: 1.0

        Returns:
            torch.Tensor: Normalized image
        """
        img_min = image.min()
        img_max = image.max()

        scale = img_max - img_min
        if scale == 0:
            scale = 1e-8

        normalized = (image - img_min) / scale
        return normalized * (max_val - min_val) + min_val

    def normalize_to_neg_one_to_one(self, image):
        """
        Normalize image to [-1, 1] range.

        Args:
            image (torch.Tensor): Input image

        Returns:
            torch.Tensor: Normalized image
        """
        return self.normalize_to_range(image, min_val=-1.0, max_val=1.0)


class MetricsCalculator:
    """
    Comprehensive calculator for image translation and super-resolution metrics.

    Combines multiple metrics for evaluating both domain translation quality and
    structural fidelity of medical images.

    Args:
        device (torch.device): Device for computation
        enable_fid (bool): Whether to enable FID calculation. Default: True
    """

    def __init__(self, device: torch.device, enable_fid: bool = True):
        self.device = device
        self.normalizer = ImageNormalizer()
        self.domain_metrics = DomainMetricsCalculator(device=device)
        self.fid_available = enable_fid
        self.lpips_3d_model = None
        self.lpips_available = True
        self.lpips_model = None
        self.fid_calculator = None

    def calculate_ncc(
        self, real_images: torch.Tensor, gen_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Normalized Cross-Correlation between real and generated images.

        Args:
            real_images (torch.Tensor): Real images
            gen_images (torch.Tensor): Generated images

        Returns:
            torch.Tensor: NCC value [-1 to 1]
        """
        try:
            real_norm = self.normalizer.normalize_to_range(real_images)
            gen_norm = self.normalizer.normalize_to_range(gen_images)

            ncc_value = self.domain_metrics.calculate_ncc(real_norm, gen_norm)

            if isinstance(ncc_value, torch.Tensor):
                return ncc_value.to(self.device)
            else:
                return torch.tensor(ncc_value, device=self.device)
        except Exception as e:
            print(f"Error calculating NCC: {str(e)}")
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device)

    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate LPIPS between two images.

        Args:
            img1 (torch.Tensor): First image
            img2 (torch.Tensor): Second image

        Returns:
            torch.Tensor: LPIPS score [0-1]
        """
        try:
            img1_norm = self.normalizer.normalize_to_range(img1)
            img2_norm = self.normalizer.normalize_to_range(img2)

            if img1_norm.dim() == 5:
                if self.lpips_3d_model is None:
                    self.lpips_3d_model = LPIPS(device=self.device)
                lpips_value = self.lpips_3d_model(img1_norm, img2_norm)
            else:
                if not self.lpips_available:
                    return torch.tensor(1.0, device=self.device)

                img1_prep = self.normalizer.normalize_to_neg_one_to_one(img1_norm)
                img2_prep = self.normalizer.normalize_to_neg_one_to_one(img2_norm)

                if img1_prep.shape[1] == 1:
                    img1_prep = img1_prep.repeat(1, 3, *([1] * (img1_prep.dim() - 2)))
                if img2_prep.shape[1] == 1:
                    img2_prep = img2_prep.repeat(1, 3, *([1] * (img2_prep.dim() - 2)))

                with torch.no_grad():
                    lpips_value = self.lpips_model(img1_prep, img2_prep)

                if lpips_value.dim() > 0:
                    lpips_value = lpips_value.mean()

            return lpips_value
        except Exception as e:
            print(f"Error calculating LPIPS: {str(e)}")
            traceback.print_exc()
            return torch.tensor(1.0, device=self.device)

    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate SSIM between two images.

        Args:
            img1 (torch.Tensor): First image
            img2 (torch.Tensor): Second image

        Returns:
            torch.Tensor: SSIM value [0-1]
        """
        try:
            ssim_calculator = SSIM(window_size=11, device=self.device)

            if img1.dim() == 5:
                depth = img1.shape[2]
                max_slices = min(8, depth)
                slice_indices = np.linspace(0, depth - 1, max_slices, dtype=int)
                ssim_values = []

                for idx in slice_indices:
                    img1_slice = img1[:, :, idx].detach()
                    img2_slice = img2[:, :, idx].detach()
                    ssim_val = ssim_calculator(img1_slice, img2_slice)
                    ssim_values.append(ssim_val.item())

                    del img1_slice
                    del img2_slice
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                ssim_value = sum(ssim_values) / len(ssim_values)
                return torch.tensor(ssim_value, device=self.device)
            else:
                return 1.0 - ssim_calculator(img1, img2)
        except Exception as e:
            print(f"Error calculating SSIM: {str(e)}")
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device)

    def _collect_images_for_fid(self):
        """Collect image slices for full-dataset FID calculation"""
        if not self.enable_full_dataset_fid:
            return

        try:
            with torch.no_grad():
                collection_factor = 0.25

                def process_for_collection(tensor):
                    if self._has_invalid_values(tensor):
                        return None

                    if tensor.dim() == 5:
                        B, C, D, H, W = tensor.shape

                        max_dim_size = 128
                        target_h = min(
                            max_dim_size, max(16, int(H * collection_factor))
                        )
                        target_w = min(
                            max_dim_size, max(16, int(W * collection_factor))
                        )

                        all_slices = []
                        for z in range(D):
                            try:
                                slice_tensor = tensor[:, :, z : z + 1, :, :].squeeze(2)

                                if (
                                    slice_tensor.shape[2] != target_h
                                    or slice_tensor.shape[3] != target_w
                                ):
                                    slice_tensor = F.interpolate(
                                        slice_tensor,
                                        size=(target_h, target_w),
                                        mode="bilinear",
                                        align_corners=False,
                                    )

                                all_slices.append(slice_tensor.detach().cpu())
                            except Exception as e:
                                continue

                        return all_slices

                    elif tensor.dim() == 4:
                        H, W = tensor.shape[2], tensor.shape[3]
                        max_dim_size = 128
                        target_h = min(
                            max_dim_size, max(16, int(H * collection_factor))
                        )
                        target_w = min(
                            max_dim_size, max(16, int(W * collection_factor))
                        )

                        downscaled = F.interpolate(
                            tensor,
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                        return [downscaled.detach().cpu()]
                    else:
                        return None

                if hasattr(self.model, "real_A") and not self._has_invalid_values(
                    self.model.real_A
                ):
                    real_A_slices = process_for_collection(self.model.real_A)
                    if real_A_slices:
                        self.real_A_collector.extend(real_A_slices)

                if hasattr(self.model, "fake_A") and not self._has_invalid_values(
                    self.model.fake_A
                ):
                    fake_A_slices = process_for_collection(self.model.fake_A)
                    if fake_A_slices:
                        self.fake_A_collector.extend(fake_A_slices)

                if hasattr(self.model, "real_B") and not self._has_invalid_values(
                    self.model.real_B
                ):
                    real_B_slices = process_for_collection(self.model.real_B)
                    if real_B_slices:
                        self.real_B_collector.extend(real_B_slices)

                if hasattr(self.model, "fake_B") and not self._has_invalid_values(
                    self.model.fake_B
                ):
                    fake_B_slices = process_for_collection(self.model.fake_B)
                    if fake_B_slices:
                        self.fake_B_collector.extend(fake_B_slices)

                self.collected_samples += 1

                if self.collected_samples % 2 == 0:
                    min_slices_required = 50
                    if (
                        len(self.real_B_collector) < min_slices_required
                        or len(self.fake_B_collector) < min_slices_required
                    ):
                        remaining = min_slices_required - min(
                            len(self.real_B_collector), len(self.fake_B_collector)
                        )
                        print(
                            f"Need at least {min_slices_required} slices for reliable FID calculation"
                        )
                        print(
                            f"Currently at {min(len(self.real_B_collector), len(self.fake_B_collector))}, need {remaining} more"
                        )

        except Exception as e:
            print(f"Error collecting images for FID: {e}")
            traceback.print_exc()

    def _has_invalid_values(self, tensor):
        """
        Check if tensor has NaN or Inf values.

        Args:
            tensor (torch.Tensor): Input tensor

        Returns:
            bool: True if tensor has invalid values, False otherwise
        """
        if tensor is None:
            return True
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()

    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate PSNR between two images.

        Args:
            img1 (torch.Tensor): First image
            img2 (torch.Tensor): Second image

        Returns:
            torch.Tensor: PSNR value in dB
        """
        try:
            img1_norm = self.normalizer.normalize_to_range(img1)
            img2_norm = self.normalizer.normalize_to_range(img2)

            psnr_calculator = PSNR(device=self.device)

            if img1_norm.dim() == 5:
                psnr_values = []
                for d in range(img1_norm.shape[2]):
                    img1_slice = img1_norm[:, :, d]
                    img2_slice = img2_norm[:, :, d]
                    psnr_value = psnr_calculator(img1_slice, img2_slice)
                    psnr_values.append(psnr_value)
                psnr_value = torch.mean(torch.stack(psnr_values))
            else:
                psnr_value = psnr_calculator(img1_norm, img2_norm)

            return psnr_value
        except Exception as e:
            print(f"Error calculating PSNR: {str(e)}")
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device)

    def _ensure_fid_calculator(self):
        """Initialize FID calculator if not already created"""
        if self.fid_calculator is None and hasattr(
            self.domain_metrics, "fid_calculator"
        ):
            self.fid_calculator = self.domain_metrics.fid_calculator
        elif self.fid_calculator is None and hasattr(
            self.domain_metrics, "_ensure_fid_calculator"
        ):
            self.domain_metrics._ensure_fid_calculator()
            self.fid_calculator = self.domain_metrics.fid_calculator

        if self.fid_calculator is None:
            from metrics.do_metrics import FID

            self.fid_calculator = FID(
                device=self.device,
                slice_sampling=True,
                max_size=512,
                max_slices=256,
                batch_size=8,
                max_feature_dim=2048,
            )

    def calculate_slice_based_fid(
        self, real_images: torch.Tensor, gen_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate FID using a slice-based approach for 3D volumes.

        Args:
            real_images (torch.Tensor): Real image volume
            gen_images (torch.Tensor): Generated image volume

        Returns:
            torch.Tensor: FID score
        """
        try:
            real_norm = self.normalizer.normalize_to_range(real_images)
            gen_norm = self.normalizer.normalize_to_range(gen_images)

            if hasattr(self.domain_metrics, "calculate_slice_based_fid"):
                fid_score = self.domain_metrics.calculate_slice_based_fid(
                    real_norm, gen_norm
                )
                return torch.tensor(fid_score, device=self.device)

            self._ensure_fid_calculator()

            if self.fid_calculator is None:
                print("Error: FID calculator initialization failed")
                return torch.tensor(float("inf"), device=self.device)

            fid_score = self.fid_calculator.calculate_slice_based(real_norm, gen_norm)
            return torch.tensor(fid_score, device=self.device)

        except Exception as e:
            print(f"Error calculating slice-based FID score: {e}")
            traceback.print_exc()
            return torch.tensor(float("inf"), device=self.device)

    def calculate_fid(
        self, real_images: torch.Tensor, gen_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate FID between real and generated images.

        Args:
            real_images (torch.Tensor): Real images
            gen_images (torch.Tensor): Generated images

        Returns:
            torch.Tensor: FID score
        """
        try:
            real_norm = self.normalizer.normalize_to_range(real_images)
            gen_norm = self.normalizer.normalize_to_range(gen_images)

            if hasattr(self.domain_metrics, "calculate_fid"):
                fid_score = self.domain_metrics.calculate_fid(real_norm, gen_norm)
                return torch.tensor(fid_score, device=self.device)

            self._ensure_fid_calculator()

            if self.fid_calculator is None:
                print("Error: FID calculator initialization failed")
                return torch.tensor(float("inf"), device=self.device)

            fid_score = self.fid_calculator.calculate(real_norm, gen_norm)
            return torch.tensor(fid_score, device=self.device)

        except Exception as e:
            print(f"Error calculating FID score: {e}")
            traceback.print_exc()
            return torch.tensor(float("inf"), device=self.device)

    def calculate_full_dataset_fid(self):
        """
        Calculate FID using collected image slices from the full dataset.

        Returns:
            dict: Dictionary with FID score
        """
        metrics = {}

        if not self.enable_full_dataset_fid:
            print("Full-dataset FID calculation is disabled")
            return {"fid_domain": float("inf")}

        min_slices_required = 50
        if (
            len(self.real_B_collector) < min_slices_required
            or len(self.fake_B_collector) < min_slices_required
        ):
            print(
                f"Not enough slices for reliable FID calculation (need at least {min_slices_required}, got {len(self.real_B_collector)} real, {len(self.fake_B_collector)} fake)"
            )
            print(f"FID calculation skipped - results would be unreliable")
            metrics["fid_domain"] = float("inf")
            return metrics

        self._initialize_metrics_calculator()

        if (
            not hasattr(self.metrics_calculator, "fid_available")
            or not self.metrics_calculator.fid_available
        ):
            print("FID calculation not available")
            return {"fid_domain": float("inf")}

        try:
            if (
                "fid_domain" not in metrics
                and len(self.real_B_collector) >= min_slices_required
                and len(self.fake_B_collector) >= min_slices_required
            ):
                try:
                    real_B_tensor = torch.cat(self.real_B_collector, dim=0)
                    fake_B_tensor = torch.cat(self.fake_B_collector, dim=0)

                    real_B_tensor = real_B_tensor.to(self.device)
                    fake_B_tensor = fake_B_tensor.to(self.device)

                    fid_domain = self.metrics_calculator.calculate_fid(
                        real_B_tensor, fake_B_tensor
                    ).item()
                    metrics["fid_domain"] = fid_domain

                    del real_B_tensor, fake_B_tensor
                    self._aggressive_memory_cleanup()

                except Exception as e:
                    print(f"Error calculating domain FID: {e}")
                    metrics["fid_domain"] = float("inf")

            if "fid_domain" not in metrics:
                metrics["fid_domain"] = float("inf")

            self.fid_history = getattr(self, "fid_history", [])
            if len(self.fid_history) >= 5:
                self.fid_history.pop(0)
            self.fid_history.append(metrics["fid_domain"])
            smoothed_fid = sum(self.fid_history) / len(self.fid_history)
            metrics["fid_domain_smoothed"] = smoothed_fid

            return metrics

        except Exception as e:
            print(f"Error in full-dataset FID calculation: {e}")
            traceback.print_exc()
            return {"fid_domain": float("inf")}
        finally:
            self.reset_collectors()

    def _aggressive_memory_cleanup(self):
        """Aggressively clean up memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def reset_collectors(self):
        """Reset image collectors"""
        self.real_A_collector = []
        self.fake_A_collector = []
        self.real_B_collector = []
        self.fake_B_collector = []
        self.collected_samples = 0

    def calculate_metrics(
        self, images_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate all available metrics for domain translation evaluation.

        Args:
            images_dict (Dict[str, torch.Tensor]): Dictionary with image tensors
                - 'real_A': Original source domain images
                - 'real_B': Original target domain images
                - 'fake_A': Generated source domain images
                - 'fake_B': Generated target domain images

        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        required_keys = ["real_A", "real_B", "fake_A", "fake_B"]
        for key in required_keys:
            if key not in images_dict:
                raise ValueError(f"Missing required image: {key}")

        metrics_dict = {}

        max_volume = 100 * 1024 * 1024
        for key, img in images_dict.items():
            num_voxels = img.numel()
            if num_voxels > max_volume:
                print(
                    f"Warning: {key} has {num_voxels} voxels, which may cause memory issues"
                )
                if img.dim() == 5:
                    scale_factor = (max_volume / num_voxels) ** (1 / 3)
                    if scale_factor < 0.5:
                        print(
                            f"Downsampling {key} to {scale_factor:.2f} of original size for metrics"
                        )
                        images_dict[key] = F.interpolate(
                            img,
                            scale_factor=scale_factor,
                            mode="trilinear",
                            align_corners=False,
                        )

        try:
            metrics_dict[MetricNames.SSIM_SR] = self.calculate_ssim(
                images_dict["real_A"], images_dict["fake_B"]
            ).item()
            torch.cuda.empty_cache()
            gc.collect()

            metrics_dict[MetricNames.PSNR_SR] = self.calculate_psnr(
                images_dict["real_A"], images_dict["fake_B"]
            ).item()
            torch.cuda.empty_cache()
            gc.collect()

            metrics_dict[MetricNames.LPIPS_SR] = self.calculate_lpips(
                images_dict["real_A"], images_dict["fake_B"]
            ).item()
            torch.cuda.empty_cache()
            gc.collect()

            try:
                if images_dict["real_B"].dim() == 5:
                    metrics_dict[MetricNames.FID_DOMAIN] = (
                        self.calculate_slice_based_fid(
                            images_dict["real_B"], images_dict["fake_B"]
                        ).item()
                    )
                else:
                    metrics_dict[MetricNames.FID_DOMAIN] = self.calculate_fid(
                        images_dict["real_B"], images_dict["fake_B"]
                    ).item()
            except Exception as e:
                print(f"Error calculating FID: {e}")
                metrics_dict[MetricNames.FID_DOMAIN] = float("inf")

            torch.cuda.empty_cache()
            gc.collect()

            try:
                metrics_dict[MetricNames.NCC_DOMAIN] = self.calculate_ncc(
                    images_dict["real_B"], images_dict["fake_B"]
                ).item()
            except Exception as e:
                print(f"Error calculating NCC: {e}")
                metrics_dict[MetricNames.NCC_DOMAIN] = 0.0

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error during metrics calculation: {e}")
            traceback.print_exc()
            if not metrics_dict:
                metrics_dict = {
                    MetricNames.SSIM_SR: 0.0,
                    MetricNames.PSNR_SR: 0.0,
                    MetricNames.LPIPS_SR: 1.0,
                    MetricNames.FID_DOMAIN: float("inf"),
                    MetricNames.NCC_DOMAIN: 0.0,
                }

        if MetricNames.FID_DOMAIN in metrics_dict:
            fid_value = metrics_dict[MetricNames.FID_DOMAIN]
            if np.isinf(fid_value):
                fid_normalized = 0.0
            else:
                fid_normalized = max(0, min(1.0, 1.0 - fid_value / 10.0))
        else:
            fid_normalized = 0.0

        if MetricNames.NCC_DOMAIN in metrics_dict:
            ncc_value = metrics_dict[MetricNames.NCC_DOMAIN]
            ncc_normalized = (ncc_value + 1.0) / 2.0
        else:
            ncc_normalized = 0.0

        metrics_dict[MetricNames.METRIC_DOMAIN] = (
            fid_normalized + ncc_normalized
        ) / 2.0

        ssim_normalized = metrics_dict.get(MetricNames.SSIM_SR, 0.0)
        psnr_normalized = min(
            max(metrics_dict.get(MetricNames.PSNR_SR, 0.0) - 20, 0) / 20.0, 1.0
        )
        lpips_normalized = 1.0 - metrics_dict.get(MetricNames.LPIPS_SR, 1.0)

        metrics_dict[MetricNames.METRIC_STRUCTURE] = (
            ssim_normalized + psnr_normalized + lpips_normalized
        ) / 3.0

        metrics_dict[MetricNames.METRIC_COMBINED] = (
            metrics_dict[MetricNames.METRIC_DOMAIN]
            + metrics_dict[MetricNames.METRIC_STRUCTURE]
        ) / 2.0

        self._sanitize_metrics(metrics_dict)
        return metrics_dict

    def _sanitize_metrics(self, metrics_dict: Dict[str, float]) -> None:
        """
        Ensure all metrics have valid values (no NaN or Inf).

        Args:
            metrics_dict (Dict[str, float]): Dictionary of metrics to sanitize
        """
        for name, value in list(metrics_dict.items()):
            if np.isnan(value) or np.isinf(value):
                if name in MetricNames.get_higher_better():
                    metrics_dict[name] = 0.0
                elif name in MetricNames.get_lower_better():
                    metrics_dict[name] = float("inf")
                else:
                    metrics_dict[name] = 0.0
