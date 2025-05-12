import torch
from torch import nn
from torch.nn import functional as F

from metrics.prostateMRIFeatureMetrics import ProstateMRIFeatureMetrics


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using domain-specific pre-trained prostate MRI anatomy model.

    Implements direct encoder feature extraction for 3D volumes,
    avoiding skip connection issues in the UNet architecture.

    Args:
        device: Device to run computations on
        use_amp: Whether to use automatic mixed precision
        multi_slice: Whether to process multiple slices for 3D volumes
        structure_oriented: Whether to focus on structural features (vs. appearance)
    """

    def __init__(
        self,
        device=None,
        use_amp=True,
        structure_oriented=True,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = use_amp
        self.structure_oriented = structure_oriented

        self.feature_extractor = ProstateMRIFeatureMetrics(
            device=self.device,
            use_layers=[
                "model.submodule.encoder3",
                "model.submodule.encoder5",
            ],
            layer_weights={
                "model.submodule.encoder3": 0.4,
                "model.submodule.encoder5": 0.6,
            },
        )

    def forward(self, input, target):
        """
        Calculate perceptual loss between input and target volumes.

        Args:
            input: Generated volumes [B, C, D, H, W]
            target: Reference volumes [B, C, D, H, W]

        Returns:
            torch.Tensor: Perceptual loss value
        """
        if min(input.shape[2:]) < 8:
            return F.l1_loss(input, target)

        if self.structure_oriented:
            return self.feature_extractor.perceptual_loss(input, target)
        else:
            return self.feature_extractor.calculate_lpips(input, target)
