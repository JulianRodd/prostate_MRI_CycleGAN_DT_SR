import torch
import torch.nn.functional as F

from loss_functions.identity_loss import identity_loss


def compute_loss_by_type(input_tensor, target_tensor, loss_type, device, criterions):
    """
    Compute loss based on the specified type.

    Args:
        input_tensor: The input tensor
        target_tensor: The target tensor
        loss_type: The type of loss to compute ('l1', 'l2', 'ssim', 'perceptual', or 'none')
        device: Torch device
        criterions: Dictionary containing loss function instances

    Returns:
        torch.Tensor: The computed loss
    """
    if loss_type is None or loss_type == "":
        return torch.tensor(0.0, device=device)

    loss_type = loss_type.lower()

    if loss_type == "l1":
        return identity_loss(input_tensor, target_tensor)
    elif loss_type == "l2":
        return F.mse_loss(input_tensor, target_tensor)
    elif loss_type == "ssim":
        return criterions["ssim"](input_tensor, target_tensor)
    elif loss_type == "perceptual":
        return criterions["perceptual"](input_tensor, target_tensor)
    elif loss_type in ["none", "null"]:
        return torch.tensor(0.0, device=device)
    else:
        print(f"Warning: Unsupported loss type: {loss_type}, falling back to L1 loss")
        return F.l1_loss(input_tensor, target_tensor)


def compute_combined_loss(
    input_tensor, target_tensor, loss_type_1, loss_type_2, device, criterions
):
    """
    Computes combined loss using two loss types.

    Args:
        input_tensor: The input tensor
        target_tensor: The target tensor
        loss_type_1: First loss type
        loss_type_2: Second loss type (optional, can be 'none')
        device: Torch device
        criterions: Dictionary containing loss function instances

    Returns:
        torch.Tensor: The combined loss
    """
    loss_type_2 = str(loss_type_2).lower() if loss_type_2 is not None else "none"

    loss_1 = compute_loss_by_type(
        input_tensor, target_tensor, loss_type_1, device, criterions
    )

    if loss_type_2 in ["none", "null"]:
        return loss_1

    loss_2 = compute_loss_by_type(
        input_tensor, target_tensor, loss_type_2, device, criterions
    )
    return (loss_1 + loss_2) / 2
