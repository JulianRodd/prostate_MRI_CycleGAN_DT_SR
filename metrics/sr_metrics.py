import torch
from piqa import SSIM as PIQA_SSIM, PSNR as PIQA_PSNR

from metrics.prostateMRIFeatureMetrics import ProstateMRIFeatureMetrics


class SSIM(torch.nn.Module):
    """
    Structural Similarity Index (SSIM) calculator for image quality assessment.

    Optimized for both 2D and 3D medical images. Higher values indicate
    better structural similarity (max value: 1.0).

    Args:
        window_size (int): Size of the window for local statistics. Default: 11
        size_average (bool): Whether to average over batch. Default: True
        device (str): Device for computation. Default: 'cpu'
    """
    def __init__(self, window_size=11, size_average=True, device="cpu"):
        super(SSIM, self).__init__()
        self.device = device
        self.window_size = window_size

    def forward(self, img1, img2):
        """
        Calculate SSIM between two images.

        Args:
            img1 (torch.Tensor): First image
            img2 (torch.Tensor): Second image

        Returns:
            torch.Tensor: SSIM value [0-1]
        """
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if img1.dim() == 5:
            B, C, D, H, W = img1.shape
            img1 = img1.squeeze(1)
            img2 = img2.squeeze(1)
            ssim = PIQA_SSIM(window_size=self.window_size, n_channels=D).to(self.device)
        else:
            _, C, _, _ = img1.shape
            ssim = PIQA_SSIM(window_size=self.window_size, n_channels=C).to(self.device)

        img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)

        ssim_value = ssim(img1, img2)
        return ssim_value


class PSNR(torch.nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) calculator for image quality assessment.

    Measures the ratio between maximum possible power of a signal and the power
    of corrupting noise. Higher values indicate better quality (typically 30-50dB
    is considered good for medical images).

    Args:
        device (str): Device for computation. Default: 'cpu'
    """
    def __init__(self, device="cpu"):
        super(PSNR, self).__init__()
        self.device = device
        self.psnr = PIQA_PSNR()

    def forward(self, img1, img2):
        """
        Calculate PSNR between two images.

        Args:
            img1 (torch.Tensor): First image
            img2 (torch.Tensor): Second image

        Returns:
            torch.Tensor: PSNR value in dB
        """
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if img1.dim() == 5:
            B, C, D, H, W = img1.shape
            psnr_values = []

            for d in range(D):
                slice1 = img1[:, :, d, :, :]
                slice2 = img2[:, :, d, :, :]
                slice1 = (slice1 - slice1.min()) / (slice1.max() - slice1.min() + 1e-8)
                slice2 = (slice2 - slice2.min()) / (slice2.max() - slice2.min() + 1e-8)
                psnr_values.append(self.psnr(slice1, slice2))

            return sum(psnr_values) / len(psnr_values)
        else:
            img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
            img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
            return self.psnr(img1, img2)


class LPIPS(torch.nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) calculator.

    Uses prostate-specific feature extractors to calculate perceptual similarity
    between images. Lower values indicate better perceptual similarity.

    Args:
        device (str): Device for computation. Default: 'cpu'
    """
    def __init__(self, device="cpu"):
        super(LPIPS, self).__init__()
        self.device = device

        self.feature_extractor = ProstateMRIFeatureMetrics(
            device=self.device,
            use_layers=[
                "model.submodule.encoder1",
                "model.submodule.encoder3",
                "model.submodule.encoder5",
            ],
            layer_weights={
                "model.submodule.encoder1": 0.2,
                "model.submodule.encoder3": 0.3,
                "model.submodule.encoder5": 0.5,
            },
        )

        self.max_size = 64 if device == "cpu" else 256

    def forward(self, img1, img2):
        """
        Calculate LPIPS between two images.

        Args:
            img1 (torch.Tensor): First image
            img2 (torch.Tensor): Second image

        Returns:
            torch.Tensor: LPIPS score [0-1] (lower is better)
        """
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if img1.dim() == 5:
            img1 = self._resize_if_needed(img1)
            img2 = self._resize_if_needed(img2)

        with torch.no_grad():
            lpips_score = self.feature_extractor.calculate_lpips(img1, img2)

        return lpips_score

    def _resize_if_needed(self, volume):
        """
        Resize volume if dimensions exceed maximum size limit for memory efficiency.

        Args:
            volume (torch.Tensor): Input volume tensor

        Returns:
            torch.Tensor: Resized volume if needed, otherwise original volume
        """
        _, _, d, h, w = volume.shape

        if d > self.max_size or h > self.max_size or w > self.max_size:
            d_factor = min(1.0, self.max_size / d)
            h_factor = min(1.0, self.max_size / h)
            w_factor = min(1.0, self.max_size / w)
            factor = min(d_factor, h_factor, w_factor)

            new_d = max(4, int(d * factor))
            new_h = max(16, int(h * factor))
            new_w = max(16, int(w * factor))

            return torch.nn.functional.interpolate(
                volume,
                size=(new_d, new_h, new_w),
                mode="trilinear",
                align_corners=False,
            )

        return volume