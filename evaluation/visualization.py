import os
import torch
import numpy as np


def get_smallest_dimension_info(volume):
    """
    Identifies the smallest dimension in a volume for consistent slice extraction.

    Args:
        volume: Tensor of shape [B, C, D, H, W]

    Returns:
        Tuple of (dimension_size, dimension_index, dimension_name)
    """
    if volume.dim() != 5:
        return None, None, None

    B, C, D, H, W = volume.shape

    # Use actual size values, not indices
    dims = [(D, 2, "D"), (H, 3, "H"), (W, 4, "W")]
    dims.sort(key=lambda x: x[0])  # Sort by size

    smallest_dim_size, smallest_dim_idx, smallest_dim_name = dims[0]
    print(f"Volume shape: [B={B}, C={C}, D={D}, H={H}, W={W}]")
    print(f"Smallest dimension is {smallest_dim_name} with {smallest_dim_size} slices")

    return smallest_dim_size, smallest_dim_idx, smallest_dim_name


def save_slice_visualization(
    image_tensor,
    save_path,
    title=None,
    cmap="gray",
    window_center=None,
    window_width=None,
    percentile_norm=True,
):
    """
    Save a visualization of a T2w MRI slice with optimal clinical display settings.

    Args:
        image_tensor: Tensor of shape [C, H, W] or [1, C, H, W]
        save_path: Path to save the visualization
        title: Not used but kept for compatibility
        cmap: Colormap to use (default: gray for T2w MRI)
        window_center: Center of the intensity window (if None, auto-calculated)
        window_width: Width of the intensity window (if None, auto-calculated)
        percentile_norm: Use percentile-based normalization for better T2w contrast
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Extract the slice and convert to numpy
    if image_tensor.dim() == 4:  # [1, C, H, W]
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor[0]  # [C, H, W]

    # Handle channels
    if image_tensor.shape[0] > 1:  # Multi-channel
        # Use the first channel or average channels
        image_np = image_tensor[0].detach().cpu().numpy()
    else:
        image_np = image_tensor[0].detach().cpu().numpy()  # Single channel

    # Apply appropriate normalization for T2w MRI
    if percentile_norm:
        # Percentile-based normalization works well for T2w MRI
        # For T2w images, fluid is bright (high percentile) and should be scaled accordingly

        # Filter out background (-1 is typical for normalized MRI background in GAN outputs)
        foreground_mask = image_np > -0.95

        if foreground_mask.sum() > 0:  # Ensure there are foreground pixels
            foreground_values = image_np[foreground_mask]

            # Get percentiles for more robust scaling
            p_low, p_high = np.percentile(foreground_values, [1, 99.5])

            # Clip to remove outliers
            normalized = np.clip(image_np, p_low, p_high)

            # Scale to 0-1 range
            normalized = (normalized - p_low) / (p_high - p_low)
        else:
            # Fallback to simple normalization if no foreground
            normalized = (image_np - image_np.min()) / (
                image_np.max() - image_np.min() + 1e-8
            )
    else:
        # Standard normalization
        normalized = (image_np - image_np.min()) / (
            image_np.max() - image_np.min() + 1e-8
        )

    # Apply windowing if provided (common in medical imaging visualization)
    if window_center is not None and window_width is not None:
        # Window level transformation
        vmin = window_center - window_width / 2
        vmax = window_center + window_width / 2
        normalized = np.clip((normalized - vmin) / (vmax - vmin), 0, 1)

    # METHOD 1: Using PIL directly for maximum control
    try:
        from PIL import Image

        # Convert normalized array to uint8 for PIL
        image_uint8 = (normalized * 255).astype(np.uint8)

        # For grayscale images
        image_pil = Image.fromarray(image_uint8, mode="L")

        # Save directly with PIL - no borders, axes, or padding
        image_pil.save(save_path)
        print(f"Saved T2w MRI image to: {save_path}")
        return
    except (ImportError, Exception) as e:
        print(f"PIL save failed: {e}, falling back to matplotlib")
        # Fall back to matplotlib if PIL not available
        pass

    # METHOD 2: Using matplotlib with all elements removed
    import matplotlib.pyplot as plt

    # Use a higher DPI for MRI images to preserve detail
    dpi = 150
    height, width = normalized.shape
    figsize = (width / dpi, height / dpi)  # Size in inches

    # Create figure with exact dimensions
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)

    # Set axis to cover entire figure with no padding
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()  # Turn off all axis elements
    fig.add_axes(ax)

    # Plot the image without colorbar, axis, or title
    # Use nearest interpolation to preserve exact pixel values of the MRI
    ax.imshow(normalized, cmap=cmap, aspect="equal", interpolation="nearest")

    # Save with zero padding
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

    print(f"Saved T2w MRI image to: {save_path}")


def save_middle_slice_visualizations(model, data_item, model_name, item_idx, base_dir):
    """
    Save visualizations of the middle slices from various model outputs,
    using the smallest dimension for consistent slice extraction.
    Uses masked versions of the generated outputs.

    Args:
        model: CycleGAN model
        data_item: Data item from the dataloader
        model_name: Name of the model run
        item_idx: Index of the validation item
        base_dir: Base directory for saving visualizations
    """
    # Import from masking module
    from masking import create_mask_from_invivo, apply_mask_to_exvivo

    with torch.no_grad():
        # Process the data item
        model.set_input(data_item)

        # Create mask from in vivo input
        invivo_mask = create_mask_from_invivo(model.real_A)

        model.test()

        # Apply mask to outputs
        model.fake_B = apply_mask_to_exvivo(model.fake_B, invivo_mask)
        if hasattr(model, "fake_A") and model.fake_A is not None:
            model.fake_A = apply_mask_to_exvivo(model.fake_A, invivo_mask)
        if hasattr(model, "rec_A") and model.rec_A is not None:
            model.rec_A = apply_mask_to_exvivo(model.rec_A, invivo_mask)
        if hasattr(model, "rec_B") and model.rec_B is not None:
            model.rec_B = apply_mask_to_exvivo(model.rec_B, invivo_mask)

        # Create base directory for this model and validation item
        save_dir = os.path.join(
            base_dir, "predicted_prostate", model_name, f"img{item_idx}"
        )
        os.makedirs(save_dir, exist_ok=True)

        # Get relevant tensors
        fake_exvivo = model.fake_B  # Generated ex vivo (from A -> B)

        # Find the smallest dimension for slicing
        if fake_exvivo is not None and fake_exvivo.dim() == 5:  # [B, C, D, H, W]
            B, C, D, H, W = fake_exvivo.shape

            # Identify smallest dimension - typically W for prostate MRI
            smallest_dim_size, smallest_dim_idx, smallest_dim_name = (
                get_smallest_dimension_info(fake_exvivo)
            )

            if smallest_dim_size is None:
                # Fallback to D/2 if something went wrong
                print("Warning: Could not determine smallest dimension, using D/2")
                mid_slice_idx = D // 2
                mid_slice = fake_exvivo[0, :, mid_slice_idx]
            else:
                # Use the middle of the smallest dimension
                mid_slice_idx = smallest_dim_size // 2

                # Extract the middle slice along the correct dimension
                if smallest_dim_idx == 2:  # D dimension
                    mid_slice = fake_exvivo[0, :, mid_slice_idx]
                    print(f"Using middle slice from D dimension: {mid_slice_idx}/{D}")
                elif smallest_dim_idx == 3:  # H dimension
                    mid_slice = fake_exvivo[0, :, :, mid_slice_idx].transpose(1, 2)
                    print(f"Using middle slice from H dimension: {mid_slice_idx}/{H}")
                else:  # W dimension
                    mid_slice = fake_exvivo[0, :, :, :, mid_slice_idx].transpose(1, 2)
                    print(f"Using middle slice from W dimension: {mid_slice_idx}/{W}")

            # Save the extracted middle slice
            save_path = os.path.join(save_dir, "fake_exvivo_masked.png")
            save_slice_visualization(
                mid_slice,
                save_path,
                title=f"Generated Ex Vivo (Middle {smallest_dim_name} Slice, Masked)",
            )

            # Try to save other types if available
            if hasattr(model, "fake_A") and model.fake_A is not None:
                fake_invivo = model.fake_A  # Generated in vivo (from B -> A)
                if fake_invivo.dim() == 5:
                    # Use the same dimension for consistency
                    if smallest_dim_idx == 2:  # D dimension
                        mid_slice = fake_invivo[0, :, mid_slice_idx]
                    elif smallest_dim_idx == 3:  # H dimension
                        mid_slice = fake_invivo[0, :, :, mid_slice_idx].transpose(1, 2)
                    else:  # W dimension
                        mid_slice = fake_invivo[0, :, :, :, mid_slice_idx].transpose(
                            1, 2
                        )

                    save_path = os.path.join(save_dir, "fake_invivo_masked.png")
                    save_slice_visualization(
                        mid_slice,
                        save_path,
                        title=f"Generated In Vivo (Middle {smallest_dim_name} Slice, Masked)",
                    )

            # Similar approach for reconstructed images
            if hasattr(model, "rec_A") and model.rec_A is not None:
                recon_invivo = model.rec_A
                if recon_invivo.dim() == 5:
                    if smallest_dim_idx == 2:  # D dimension
                        mid_slice = recon_invivo[0, :, mid_slice_idx]
                    elif smallest_dim_idx == 3:  # H dimension
                        mid_slice = recon_invivo[0, :, :, mid_slice_idx].transpose(1, 2)
                    else:  # W dimension
                        mid_slice = recon_invivo[0, :, :, :, mid_slice_idx].transpose(
                            1, 2
                        )

                    save_path = os.path.join(save_dir, "recon_invivo_masked.png")
                    save_slice_visualization(
                        mid_slice,
                        save_path,
                        title=f"Reconstructed In Vivo (Middle {smallest_dim_name} Slice, Masked)",
                    )

            if hasattr(model, "rec_B") and model.rec_B is not None:
                recon_exvivo = model.rec_B
                if recon_exvivo.dim() == 5:
                    if smallest_dim_idx == 2:  # D dimension
                        mid_slice = recon_exvivo[0, :, mid_slice_idx]
                    elif smallest_dim_idx == 3:  # H dimension
                        mid_slice = recon_exvivo[0, :, :, mid_slice_idx].transpose(1, 2)
                    else:  # W dimension
                        mid_slice = recon_exvivo[0, :, :, :, mid_slice_idx].transpose(
                            1, 2
                        )

                    save_path = os.path.join(save_dir, "recon_exvivo_masked.png")
                    save_slice_visualization(
                        mid_slice,
                        save_path,
                        title=f"Reconstructed Ex Vivo (Middle {smallest_dim_name} Slice, Masked)",
                    )
