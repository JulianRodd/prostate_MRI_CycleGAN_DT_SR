import torch
from piqa import SSIM


class SSIMLoss(SSIM):
    """
    Structural Similarity Index (SSIM) loss implementation.

    Normalizes input tensors before computing SSIM to ensure
    consistent behavior regardless of input value range.

    Args:
        window_size: Size of the window for structural similarity calculation
        device: Computing device (CPU/GPU)
    """

    def __init__(self, window_size=11, device=None):
        super().__init__(window_size=window_size, n_channels=1)
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.to(self.device)

    def forward(self, x, y):
        """
        Compute SSIM loss between input tensors after normalization.

        Args:
            x: First input tensor
            y: Second input tensor

        Returns:
            torch.Tensor: 1 - SSIM (as a loss that should be minimized)
        """
        x = x.to(self.device)
        y = y.to(self.device)

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        x_normalized = (x - x_min) / x_range
        y_normalized = (y - y_min) / y_range

        return 1.0 - super().forward(x_normalized, y_normalized)
