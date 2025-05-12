import torch
import torch.nn.functional as F


def identity_loss(
    identity_image: torch.Tensor,
    real_image: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Simple identity loss using L1 loss.

    Args:
        identity_image (torch.Tensor): The identity-mapped image
        real_image (torch.Tensor): The original input image
        reduction (str): Specifies the reduction. Options: 'none' | 'mean' | 'sum'

    Returns:
        torch.Tensor: The computed L1 loss
    """
    if not (
        isinstance(identity_image, torch.Tensor)
        and isinstance(real_image, torch.Tensor)
    ):
        raise TypeError("Both inputs must be torch.Tensor")

    if identity_image.size() != real_image.size():
        raise ValueError(
            f"Input shapes must match: got {identity_image.size()} and {real_image.size()}"
        )

    identity_image = torch.nan_to_num(identity_image, nan=0.0)
    real_image = torch.nan_to_num(real_image, nan=0.0)

    return F.l1_loss(identity_image, real_image, reduction=reduction)
