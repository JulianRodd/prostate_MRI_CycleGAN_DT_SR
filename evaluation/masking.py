import torch


def create_mask_from_invivo(invivo_tensor, threshold=-0.95):
    """
    Create a binary mask from the in vivo image where values > threshold are foreground.

    Args:
        invivo_tensor: Input in vivo tensor [B, C, D, H, W] or [B, C, H, W]
        threshold: Value to threshold at (default: -0.95, slightly above -1 background)

    Returns:
        Binary mask of same shape as input
    """
    with torch.no_grad():
        # Create binary mask where values > threshold are 1, others are 0
        mask = (invivo_tensor > threshold).float()

        # Ensure mask is on the same device as the input
        mask = mask.to(invivo_tensor.device)

        return mask


def apply_mask_to_exvivo(exvivo_tensor, mask):
    """
    Apply binary mask to ex vivo output, setting background to -1.

    Args:
        exvivo_tensor: Generated ex vivo tensor
        mask: Binary mask from in vivo input

    Returns:
        Masked ex vivo tensor
    """
    with torch.no_grad():
        # Ensure mask has the same shape as exvivo_tensor
        if mask.shape != exvivo_tensor.shape:
            # Handle potential dimension mismatch
            # For example, if mask is [B, 1, D, H, W] and exvivo is [B, C, D, H, W]
            if mask.shape[1] == 1 and exvivo_tensor.shape[1] > 1:
                mask = mask.repeat(
                    1, exvivo_tensor.shape[1], *([1] * (len(mask.shape) - 2))
                )

        # Apply mask: keep foreground values, set background to -1
        masked_exvivo = exvivo_tensor * mask + (-1.0) * (1.0 - mask)

        return masked_exvivo
