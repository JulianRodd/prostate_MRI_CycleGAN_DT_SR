import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as spectral_norm


def apply_spectral_norm(module):
    """
    Apply spectral normalization to appropriate layers in the module.

    Recursively applies spectral normalization to convolutional and linear layers
    to improve training stability. Checks if spectral normalization is already
    applied before applying it.

    Args:
        module: PyTorch module to apply spectral normalization to

    Returns:
        Module with spectral normalization applied
    """
    # For modules that should have spectral norm applied
    if isinstance(
            module,
            (
                    torch.nn.Conv2d,
                    torch.nn.Conv3d,
                    torch.nn.ConvTranspose2d,
                    torch.nn.ConvTranspose3d,
                    torch.nn.Linear,
            ),
    ):
        # Check if spectral norm is already applied by looking for weight_orig
        if hasattr(module, "weight_orig"):
            # Already has spectral norm applied
            return module

        # Apply spectral norm only if not already applied
        return spectral_norm(module)

    # For other container modules, apply recursively to children
    for name, child in module.named_children():
        module._modules[name] = apply_spectral_norm(child)

    return module


def verify_spectral_norm(net_g_a, net_g_b):
    """
    Verify that spectral normalization has been applied to the generators.

    Counts the number of layers with spectral normalization applied and
    prints debug information. Useful for debugging spectral norm application.

    Args:
        net_g_a: Generator A network
        net_g_b: Generator B network

    Returns:
        bool: True if both generators have spectral norm applied, False otherwise
    """
    def count_spectral_norm(module):
        count = 0
        for child in module.modules():
            if hasattr(child, "weight_orig"):
                count += 1
        return count

    g_a_count = count_spectral_norm(net_g_a)
    g_b_count = count_spectral_norm(net_g_b)

    print(f"Generator A has {g_a_count} spectral norm layers")
    print(f"Generator B has {g_b_count} spectral norm layers")

    return g_a_count > 0 and g_b_count > 0


def compute_r1_penalty(discriminator, real_samples, use_checkpoint=False):
    """
    Compute R1 gradient penalty for discriminator.

    This penalizes gradients on real samples only, which improves training stability.

    Args:
        discriminator: The discriminator network
        real_samples: Real data samples
        use_checkpoint: Whether to use gradient checkpointing in the model

    Returns:
        Tensor: R1 gradient penalty value
    """
    # Create a clone of real_samples that requires gradients
    real_samples_with_grad = real_samples.detach().clone().requires_grad_(True)

    # Get discriminator output
    pred_real = discriminator(real_samples_with_grad)
    if isinstance(pred_real, tuple):
        # If discriminator returns features too, just use the prediction
        pred_real = pred_real[0]

    # Sum to create a scalar output
    pred_real_sum = pred_real.sum()

    try:
        # Calculate gradients - use create_graph=False if using gradient checkpointing
        # to avoid incompatibility
        gradients = torch.autograd.grad(
            outputs=pred_real_sum,
            inputs=real_samples_with_grad,
            create_graph=not use_checkpoint,  # Avoid create_graph with checkpointing
            retain_graph=not use_checkpoint,  # Avoid retain_graph with checkpointing
            only_inputs=True,
        )[0]

        # Calculate R1 penalty (squared gradient norm)
        r1_penalty = gradients.pow(2).reshape(gradients.shape[0], -1).sum(1).mean()

        return r1_penalty
    except RuntimeError as e:
        # If we get an error related to checkpointing, return a small constant penalty
        if "checkpoint" in str(e).lower():
            print(
                "Warning: R1 penalty computation failed due to checkpointing incompatibility"
            )
            print(f"Error: {e}")
            print("Using fallback penalty value")
            return torch.tensor(0.1, device=real_samples.device)
        else:
            # Re-raise other errors
            raise


def compute_gradient_penalty(
        discriminator,
        real_samples,
        fake_samples,
        device,
        use_checkpoint=False,
        use_wasserstein=True,
):
    """
    Compute gradient penalty for improved WGAN training stability.

    Implements the gradient penalty from WGAN-GP with additional safety measures
    for handling tensor size mismatches and numerical stability.

    Args:
        discriminator: The discriminator network
        real_samples: Real data samples
        fake_samples: Generated data samples
        device: Device to compute on
        use_checkpoint: Whether to use gradient checkpointing
        use_wasserstein: Whether to use Wasserstein GAN formulation

    Returns:
        Tensor: Gradient penalty value
    """
    # Ensure consistent tensor types
    real_samples = real_samples.float()
    fake_samples = fake_samples.float()

    # Ensure fake_samples have the same spatial dimensions as real_samples
    if real_samples.shape != fake_samples.shape:
        fake_samples = F.interpolate(
            fake_samples,
            size=real_samples.shape[2:],  # Get spatial dimensions (D, H, W)
            mode="trilinear",
            align_corners=True,
        )

    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, 1).to(device)

    # Generate interpolated samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)

    # Forward pass through discriminator with Wasserstein flag
    disc_interpolates = discriminator(interpolates, use_wasserstein=use_wasserstein)

    # Prepare gradient outputs with safe handling of different output shapes
    if isinstance(disc_interpolates, tuple):
        disc_interpolates = disc_interpolates[0]

    # Create a flat version to ensure valid gradients
    flat_disc_interpolates = disc_interpolates.sum()

    try:
        # Calculate gradients with safety handling
        gradients = torch.autograd.grad(
            outputs=flat_disc_interpolates,
            inputs=interpolates,
            create_graph=not use_checkpoint,
            retain_graph=not use_checkpoint,
            only_inputs=True,
        )[0]

        # Reshape and calculate norms with safety clipping
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Clip gradient norms for stability (prevents outlier impact)
        gradient_norm = torch.clamp(gradient_norm, 0.0, 10.0)

        # Compute penalty with scaling factor to reduce magnitude
        gp_scale = 0.1  # Scale down the gradient penalty
        gradient_penalty = (((gradient_norm - 1) ** 2) * gp_scale).mean()

        # Final safety clamp to prevent extreme values
        gradient_penalty = torch.clamp(gradient_penalty, 0.0, 10.0)

        # Debug info
        print(
            f"GP stats: min={gradient_norm.min().item():.4f}, max={gradient_norm.max().item():.4f}, mean={gradient_norm.mean().item():.4f}"
        )

        return gradient_penalty

    except RuntimeError as e:
        # Handle runtime errors
        print(f"Warning: Gradient penalty error: {e}")
        return torch.tensor(0.1, device=device)