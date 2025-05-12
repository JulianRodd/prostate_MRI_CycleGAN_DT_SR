import csv
import gc
import math
import os
import sys
import traceback
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options.base_options import BaseOptions
from models.cycle_gan_model import CycleGANModel
from metrics.do_metrics import DomainMetricsCalculator
from dataset.data_loader import setup_dataloaders
from utils.utils import set_seed
import os
import torch
import numpy as np
from tqdm import tqdm

# Global cache for training data
CACHED_TRAIN_EXVIVO_DATA = None

from metrics.val_metrics import MetricsCalculator


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

        # Optional: Apply morphological operations if needed
        # This could involve erosion/dilation using conv operations
        # For simplicity, we're using the direct threshold mask

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


def calculate_additional_metrics_for_validation(model, validation_data, device):
    """
    Calculate PSNR, SSIM, LPIPS, and NCC metrics for the validation dataset.
    Now uses masked versions of the generated outputs.

    Args:
        model: CycleGAN model
        validation_data: DataLoader for validation samples
        device: Computation device

    Returns:
        dict: Dictionary with average metrics
    """
    metrics_calc = MetricsCalculator(device=device)

    # Initialize accumulators for metrics
    total_metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "ncc": 0.0}
    num_samples = 0

    # Process each validation sample
    for i, data in enumerate(tqdm(validation_data, desc="Calculating metrics")):
        with torch.no_grad():
            # Set input data
            model.set_input(data)

            # Create mask from in vivo input
            invivo_mask = create_mask_from_invivo(model.real_A)

            # Run forward pass
            model.test()

            # Get the outputs
            real_A = model.real_A
            real_B = model.real_B
            fake_B = model.fake_B
            fake_A = model.fake_A if hasattr(model, "fake_A") else None

            # Skip if any outputs are None
            if real_A is None or real_B is None or fake_B is None:
                print(f"Warning: Skipping sample {i} due to None outputs")
                continue

            # Apply mask to fake outputs
            masked_fake_B = apply_mask_to_exvivo(fake_B, invivo_mask)
            masked_fake_A = (
                apply_mask_to_exvivo(fake_A, invivo_mask)
                if fake_A is not None
                else None
            )

            # Create image dictionary for metrics calculation
            images_dict = {
                "real_A": real_A,
                "real_B": real_B,
                "fake_A": (
                    masked_fake_A
                    if masked_fake_A is not None
                    else torch.zeros_like(real_A)
                ),
                "fake_B": masked_fake_B,
            }

            try:
                # Calculate metrics for this sample
                sample_metrics = metrics_calc.calculate_metrics(images_dict)

                # Accumulate metrics
                total_metrics["psnr"] += sample_metrics.get("psnr_sr", 0.0)
                total_metrics["ssim"] += sample_metrics.get("ssim_sr", 0.0)
                total_metrics["lpips"] += sample_metrics.get("lpips_sr", 1.0)
                total_metrics["ncc"] += sample_metrics.get("ncc_domain", 0.0)

                num_samples += 1

                print(
                    f"Sample {i}: PSNR={sample_metrics.get('psnr_sr', 0.0):.4f}, SSIM={sample_metrics.get('ssim_sr', 0.0):.4f}, LPIPS={sample_metrics.get('lpips_sr', 1.0):.4f}, NCC={sample_metrics.get('ncc_domain', 0.0):.4f}"
                )

            except Exception as e:
                print(f"Error calculating metrics for sample {i}: {e}")
                traceback.print_exc()

            # Clean up memory
            del images_dict, masked_fake_B
            if masked_fake_A is not None:
                del masked_fake_A
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Calculate averages
    if num_samples > 0:
        for key in total_metrics:
            total_metrics[key] /= num_samples

    print(f"Calculated metrics over {num_samples} validation samples:")
    for key, value in total_metrics.items():
        print(f"  - {key}: {value:.4f}")

    return total_metrics


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


def save_middle_slice_visualizations(model, data_item, model_name, item_idx, base_dir):
    """
    Save visualizations of the middle slices from various model outputs,
    using the smallest dimension for consistent slice extraction.
    Now uses masked versions of the generated outputs.

    Args:
        model: CycleGAN model
        data_item: Data item from the dataloader
        model_name: Name of the model run
        item_idx: Index of the validation item
        base_dir: Base directory for saving visualizations
    """
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


class FIDEvalOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--models",
            nargs="+",
            type=str,
            default=None,
            help="Specific model names to evaluate (default: all models in checkpoints dir)",
        )

        parser.add_argument(
            "--csv_path",
            type=str,
            default=None,
            help="Path to CSV file containing model configurations to evaluate",
        )

        parser.add_argument(
            "--output_file",
            type=str,
            default=None,
            help="Path to output CSV file for results",
        )

        parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="Which epoch to load for evaluation",
        )
        parser.add_argument(
            "--use_full_validation",
            action="store_true",
            help="Use full images for validation instead of patches",
            default=True,
        )
        parser.add_argument(
            "--patches_per_image",
            type=int,
            required=True,
            help="Number of patches per image",
        )

        # Rest of existing arguments...
        parser.add_argument(
            "--use_spectral_norm_G",
            action="store_true",
            help="use spectral normalization in generator",
        )
        parser.add_argument(
            "--use_stn",
            action="store_true",
            help="use spatial transformer network in generator",
            default=False,
        )
        parser.add_argument(
            "--use_residual",
            action="store_true",
            help="use residual blocks in generator",
            default=False,
        )
        parser.add_argument(
            "--use_full_attention",
            action="store_true",
            help="use full attention in generator",
            default=False,
        )

        parser.add_argument(
            "--use_lsgan",
            action="store_true",
            help="use least squares GAN",
            default=True,
        )
        parser.add_argument(
            "--use_hinge", action="store_true", help="use hinge loss", default=False
        )
        parser.add_argument(
            "--use_relativistic",
            action="store_true",
            help="use relativistic discriminator",
            default=False,
        )
        parser.add_argument(
            "--lambda_identity",
            type=float,
            default=0.5,
            help="weight for identity loss",
        )

        parser.add_argument(
            "--lambda_domain_adaptation",
            type=float,
            default=1.0,
            help="weight for domain adaptation loss",
        )
        parser.add_argument(
            "--lambda_da_contrast",
            type=float,
            default=1.0,
            help="weight for contrast component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_structure",
            type=float,
            default=1.0,
            help="weight for structure component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_texture",
            type=float,
            default=1.0,
            help="weight for texture component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_histogram",
            type=float,
            default=0.0,
            help="weight for histogram component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_gradient",
            type=float,
            default=0.0,
            help="weight for gradient component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_ncc",
            type=float,
            default=0.0,
            help="weight for NCC component in domain adaptation",
        )

        # Added memory optimization parameters
        parser.add_argument(
            "--batch_slice_processing",
            action="store_true",
            help="Process slices in batches to reduce memory usage",
            default=True,
        )
        parser.add_argument(
            "--slice_batch_size",
            type=int,
            default=32,
            help="Batch size for slice processing",
        )
        parser.add_argument(
            "--memory_profiling",
            action="store_true",
            help="Enable memory profiling",
            default=False,
        )

        # Check if patch_size already exists in the parser to avoid conflicts
        has_patch_size = False
        for action in parser._actions:
            if action.dest == "patch_size":
                has_patch_size = True
                break

        # Only add patch_size if it doesn't already exist
        if not has_patch_size:
            parser.add_argument(
                "--patch_size",
                nargs="+",
                type=int,
                default=[64, 64, 32],
                help="Size of patches for sliding window approach [D, H, W]",
            )
            parser.add_argument(
                "--min_patch_size",
                nargs="+",
                type=int,
                default=[16, 16, 8],
                help="Minimum size of valid patches [D, H, W]",
            )

        self.isTrain = False
        return parser


def predict_with_sliding_window(
    model, validation_data, device, patch_size=[64, 64, 32], min_patch_size=[16, 16, 8]
):
    """
    Process validation data using a sliding window approach for memory efficiency,
    with automatic stride calculation.

    Args:
        model: CycleGAN model
        validation_data: DataLoader for validation samples
        device: Computation device (CPU/GPU)
        patch_size: Size of patches to process [D, H, W]
        min_patch_size: Minimum patch size to process [D, H, W]

    Returns:
        tuple: (real_B, fake_B) tensors of the full processed volumes
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize results collectors
    all_real_B = []
    all_fake_B = []

    # Process each validation sample
    for i, data in enumerate(
        tqdm(validation_data, desc="Processing validation samples")
    ):
        with torch.no_grad():
            # Process current sample with sliding window
            real_B, fake_B = process_sample_with_sliding_window(
                model, data, device, patch_size, min_patch_size
            )

            # Store results if valid
            if real_B is not None and fake_B is not None:
                all_real_B.append(real_B.detach().cpu())
                all_fake_B.append(fake_B.detach().cpu())

            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Return the model's results (should be set by process_sample_with_sliding_window)
    return model.real_B, model.fake_B


def process_sample_with_sliding_window(
    model, data, device, patch_size=[64, 64, 32], min_patch_size=[16, 16, 8]
):
    """
    Process a single sample with sliding window to handle large 3D volumes efficiently.
    Uses automatic stride calculation based on patch size.
    Now applies masking to the generated outputs.

    Args:
        model: CycleGAN model
        data: Sample data
        device: Computation device
        patch_size: Size of patches to process [D, H, W]
        min_patch_size: Minimum patch size to process [D, H, W]

    Returns:
        tuple: (real_B, masked_fake_B) for the processed sample
    """
    with torch.no_grad():
        # Set the input data
        model.set_input(data)

        # Get input tensors
        real_A = model.real_A  # invivo
        real_B = model.real_B  # exvivo

        # Check if we have valid inputs
        if real_A is None:
            print("Warning: real_A is None, skipping sample")
            return None, None

        # Create mask from in vivo input
        invivo_mask = create_mask_from_invivo(real_A)

        # Process differently based on dimensions
        if real_A.dim() == 5:  # 3D volume [B, C, W, H, D]
            B, C, W, H, D = real_A.shape
            print(f"Original data shape: [B={B}, C={C}, W={W}, H={H}, D={D}]")

            # Calculate stride as half of patch dimensions
            stride_inplane = max(patch_size[1] // 2, 1)  # Half of H dimension
            stride_layer = max(patch_size[0] // 2, 1)  # Half of D dimension

            print(
                f"Using patch size: {patch_size}, stride_inplane: {stride_inplane}, stride_layer: {stride_layer}"
            )

            # Generate patch indices for sliding window
            patch_indices = generate_patch_indices(
                real_A.shape[2:],
                patch_size,
                stride_inplane,
                stride_layer,
                min_patch_size,
            )

            print(f"Generated {len(patch_indices)} patches for processing")

            # Initialize output tensors with same shape as input
            fake_B = torch.zeros_like(real_A, device="cpu")
            weight_map = torch.zeros_like(real_A, device="cpu")

            # Process patches one by one to save memory
            for idx, patch_idx in enumerate(
                tqdm(patch_indices, desc="Processing patches")
            ):
                istart, iend, jstart, jend, kstart, kend = patch_idx

                # Extract patch and corresponding mask patch
                patch_A = real_A[:, :, istart:iend, jstart:jend, kstart:kend].to(device)
                patch_mask = invivo_mask[
                    :, :, istart:iend, jstart:jend, kstart:kend
                ].to(device)

                try:
                    # Forward pass for this patch
                    patch_fake_B = model.netG_A(patch_A)

                    # Apply mask to patch result
                    patch_fake_B = apply_mask_to_exvivo(patch_fake_B, patch_mask)

                    # Add result to output tensor (on CPU to save GPU memory)
                    fake_B[
                        :, :, istart:iend, jstart:jend, kstart:kend
                    ] += patch_fake_B.cpu()
                    weight_map[:, :, istart:iend, jstart:jend, kstart:kend] += 1.0

                except Exception as e:
                    print(f"Error processing patch {idx}: {e}")

                # Clean up to free memory
                del patch_A, patch_mask
                if "patch_fake_B" in locals():
                    del patch_fake_B

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Average overlapping regions
            epsilon = 1e-8
            weight_map = torch.where(
                weight_map > 0,
                weight_map,
                torch.tensor(epsilon, device=weight_map.device),
            )
            fake_B = fake_B / weight_map

            # Apply final mask to entire volume
            masked_fake_B = apply_mask_to_exvivo(fake_B, invivo_mask.cpu())

            # Move final results to device and store in model
            model.fake_B = masked_fake_B.to(device)

            # Clean up processing tensors
            del weight_map, fake_B

        else:  # Already 2D [B, C, H, W]
            # Standard forward pass for 2D data
            model.test()

            # Apply mask to the result
            model.fake_B = apply_mask_to_exvivo(model.fake_B, invivo_mask.to(device))

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model.real_B, model.fake_B


def generate_patch_indices(
    image_shape, patch_size, stride_inplane, stride_layer, min_patch_size
):
    """
    Generate patch indices for sliding window processing.

    Args:
        image_shape: Shape of the image [W, H, D]
        patch_size: Size of patches to process [D, H, W]
        stride_inplane: Stride for in-plane dimensions (H, W)
        stride_layer: Stride for layer dimension (D)
        min_patch_size: Minimum patch size to consider valid [D, H, W]

    Returns:
        list: List of patch indices [istart, iend, jstart, jend, kstart, kend]
    """
    inum = max(
        1,
        int(math.ceil((image_shape[0] - patch_size[0]) / float(stride_inplane))) + 1,
    )
    jnum = max(
        1,
        int(math.ceil((image_shape[1] - patch_size[1]) / float(stride_inplane))) + 1,
    )
    knum = max(
        1,
        int(math.ceil((image_shape[2] - patch_size[2]) / float(stride_layer))) + 1,
    )

    patch_indices = []

    for k in range(knum):
        for i in range(inum):
            for j in range(jnum):
                istart = min(i * stride_inplane, max(0, image_shape[0] - patch_size[0]))
                iend = min(istart + patch_size[0], image_shape[0])

                if iend - istart < min_patch_size[0]:
                    continue

                jstart = min(j * stride_inplane, max(0, image_shape[1] - patch_size[1]))
                jend = min(jstart + patch_size[1], image_shape[1])

                if jend - jstart < min_patch_size[1]:
                    continue

                kstart = min(k * stride_layer, max(0, image_shape[2] - patch_size[2]))
                kend = min(kstart + patch_size[2], image_shape[2])

                if kend - kstart < min_patch_size[2]:
                    continue

                if (
                    (iend - istart) >= min_patch_size[0]
                    and (jend - jstart) >= min_patch_size[1]
                    and (kend - kstart) >= min_patch_size[2]
                ):
                    patch_indices.append([istart, iend, jstart, jend, kstart, kend])

    if not patch_indices:
        print(f"Warning: No valid patches found for image of shape {image_shape}.")
        print(f"Creating a single central patch with minimum sizes {min_patch_size}")

        istart = max(0, image_shape[0] // 2 - min_patch_size[0] // 2)
        iend = min(image_shape[0], istart + min_patch_size[0])

        jstart = max(0, image_shape[1] // 2 - min_patch_size[1] // 2)
        jend = min(image_shape[1], jstart + min_patch_size[1])

        kstart = max(0, image_shape[2] // 2 - min_patch_size[2] // 2)
        kend = min(image_shape[2], kstart + min_patch_size[2])

        if (iend - istart) >= 4 and (jend - jstart) >= 4 and (kend - kstart) >= 4:
            patch_indices.append([istart, iend, jstart, jend, kstart, kend])
        else:
            print(f"Cannot create valid patches for this image - dimensions too small")

    return patch_indices


def extract_slices_from_volumes(volumes, device, min_content_percentage=0.001):
    """
    Extract 2D slices from a list of 3D volumes in a memory-efficient way.

    Args:
        volumes: List of 3D volumes [B, C, D, H, W]
        device: Computation device
        min_content_percentage: Minimum non-background content to keep slice

    Returns:
        Tensor of 2D slices [N, C, H, W]
    """
    all_slices = []
    total_slices = 0
    accepted_slices = 0

    for vol_idx, vol in enumerate(volumes):
        if vol.dim() == 5:  # 3D volume [B, C, D, H, W]
            B, C, D, H, W = vol.shape
            print(
                f"Processing volume {vol_idx} with shape [B={B}, C={C}, D={D}, H={H}, W={W}]"
            )

            # Extract all slices along the smallest dimension for consistent orientation
            dims = [(D, 2, "D"), (H, 3, "H"), (W, 4, "W")]
            dims.sort(key=lambda x: x[0])  # Sort by size

            slice_dim_size, slice_dim_idx, slice_dim_name = dims[0]
            print(
                f"Vol {vol_idx}: Using {slice_dim_name} dimension with {slice_dim_size} slices"
            )

            # Extract all slices
            for slice_idx in range(slice_dim_size):
                total_slices += 1

                if slice_dim_idx == 2:  # D dimension
                    slice_tensor = vol[0:1, :, slice_idx : slice_idx + 1, :, :].squeeze(
                        2
                    )
                elif slice_dim_idx == 3:  # H dimension
                    slice_tensor = vol[0:1, :, :, slice_idx : slice_idx + 1, :].squeeze(
                        3
                    )
                else:  # W dimension
                    slice_tensor = vol[0:1, :, :, :, slice_idx : slice_idx + 1].squeeze(
                        4
                    )

                # Skip slices with invalid values
                if torch.isnan(slice_tensor).any() or torch.isinf(slice_tensor).any():
                    continue

                # Skip slices with almost no content (very loose threshold)
                non_background = ((slice_tensor > -0.95).float().mean()).item()
                if non_background < min_content_percentage:
                    continue

                all_slices.append(
                    slice_tensor.detach().cpu()
                )  # Move to CPU to save GPU memory
                accepted_slices += 1

                # Periodically clear CUDA cache
                if accepted_slices % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(
        f"Extracted {accepted_slices}/{total_slices} valid slices from {len(volumes)} volumes"
    )

    if not all_slices:
        print("Warning: No valid slices extracted! Using fallback.")
        # Create a single dummy slice as fallback
        return torch.zeros(1, volumes[0].shape[1], 32, 32, device=device)

    # Combine all slices
    print("Combining extracted slices...")
    result = torch.cat(all_slices, dim=0).to(device)
    print(f"Final extracted slices shape: {result.shape}")

    # Clear memory
    del all_slices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def combine_real_datasets_memory_efficient(val_real, train_real, device):
    """
    Combine validation and training real data in a memory-efficient way.

    When data dimensions don't match, we use batch processing rather than
    trying to manipulate the entire tensor at once.

    Args:
        val_real: Validation real data
        train_real: Training real data
        device: Computation device

    Returns:
        Combined real data
    """
    import torch.nn.functional as F

    # Get shapes
    val_shape = val_real.shape
    train_shape = train_real.shape

    print(f"Validation data shape: {val_shape}")
    print(f"Training data shape: {train_shape}")

    # Check if dimensions and channels match
    if len(val_shape) == len(train_shape) and val_shape[1] == train_shape[1]:
        # If spatial dimensions differ, need to resize training data
        if val_shape[2:] != train_shape[2:]:
            print(
                "Spatial dimensions differ - will resize training data to match validation"
            )
            # We'll do this in batches to avoid OOM

            # Move validation to CPU to save memory
            val_real_cpu = val_real.cpu()

            # Define batch size for processing
            batch_size = min(100, train_real.shape[0])
            num_batches = (train_real.shape[0] + batch_size - 1) // batch_size

            print(
                f"Processing {train_real.shape[0]} training samples in {num_batches} batches of size {batch_size}"
            )

            resized_train_batches = []

            for i in range(0, train_real.shape[0], batch_size):
                end_idx = min(i + batch_size, train_real.shape[0])
                print(
                    f"Processing batch {i // batch_size + 1}/{num_batches} ({i}:{end_idx})"
                )

                # Get the current batch
                train_batch = train_real[i:end_idx]

                # Resize to match validation spatial dimensions
                try:
                    mode = "bilinear" if len(val_shape) == 4 else "trilinear"
                    resized_batch = F.interpolate(
                        train_batch, size=val_shape[2:], mode=mode, align_corners=True
                    )

                    # Move to CPU to save GPU memory
                    resized_train_batches.append(resized_batch.cpu())

                    # Clear GPU memory
                    del train_batch, resized_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error resizing batch: {e}")
                    # Skip this batch
                    continue

            # Combine all batches
            if resized_train_batches:
                train_real_resized = torch.cat(resized_train_batches, dim=0)
                # Move validation and resized training back to device for concatenation
                val_real = val_real_cpu.to(device)
                train_real_resized = train_real_resized.to(device)

                # Now we can concatenate
                combined = torch.cat([val_real, train_real_resized], dim=0)

                # Clean up
                del val_real_cpu, train_real_resized, resized_train_batches
                return combined

    # If dimensions don't match, we need to extract features separately
    print("WARNING: Cannot efficiently combine datasets with different dimensions")
    print("Will calculate FID scores separately and use weighted average")

    return None


def extract_slices_from_volumes(volumes, device, min_content_percentage=0.001):
    """
    Extract 2D slices from a list of 3D volumes in a memory-efficient way.
    Always uses the smallest dimension for consistent slice extraction.

    Args:
        volumes: List of 3D volumes [B, C, D, H, W]
        device: Computation device
        min_content_percentage: Minimum non-background content to keep slice

    Returns:
        Tensor of 2D slices [N, C, H, W]
    """
    all_slices = []
    total_slices = 0
    accepted_slices = 0

    for vol_idx, vol in enumerate(volumes):
        if vol.dim() == 5:  # 3D volume [B, C, D, H, W]
            # Find the smallest dimension (typically W for prostate MRI)
            smallest_dim_size, smallest_dim_idx, smallest_dim_name = (
                get_smallest_dimension_info(vol)
            )

            if smallest_dim_size is None:
                print(
                    f"Warning: Skipping volume {vol_idx} - could not determine dimensions"
                )
                continue

            print(
                f"Vol {vol_idx}: Using {smallest_dim_name} dimension with {smallest_dim_size} slices"
            )

            # Extract all slices along the smallest dimension
            for slice_idx in range(smallest_dim_size):
                total_slices += 1

                if smallest_dim_idx == 2:  # D dimension
                    slice_tensor = vol[0:1, :, slice_idx : slice_idx + 1, :, :].squeeze(
                        2
                    )
                elif smallest_dim_idx == 3:  # H dimension
                    # Transpose to maintain consistent orientation
                    slice_tensor = vol[0:1, :, :, slice_idx : slice_idx + 1, :].squeeze(
                        3
                    )
                    slice_tensor = slice_tensor.transpose(2, 3)
                else:  # W dimension
                    # Transpose to maintain consistent orientation
                    slice_tensor = vol[0:1, :, :, :, slice_idx : slice_idx + 1].squeeze(
                        4
                    )
                    slice_tensor = slice_tensor.transpose(2, 3)

                # Skip slices with invalid values
                if torch.isnan(slice_tensor).any() or torch.isinf(slice_tensor).any():
                    continue

                # Skip slices with almost no content (very loose threshold)
                non_background = ((slice_tensor > -0.95).float().mean()).item()
                if non_background < min_content_percentage:
                    continue

                all_slices.append(
                    slice_tensor.detach().cpu()
                )  # Move to CPU to save GPU memory
                accepted_slices += 1

                # Periodically clear CUDA cache
                if accepted_slices % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(
        f"Extracted {accepted_slices}/{total_slices} valid slices from {len(volumes)} volumes"
    )

    if not all_slices:
        print("Warning: No valid slices extracted! Using fallback.")
        # Create a single dummy slice as fallback
        return torch.zeros(1, volumes[0].shape[1], 32, 32, device=device)

    # Combine all slices
    print("Combining extracted slices...")
    result = torch.cat(all_slices, dim=0).to(device)
    print(f"Final extracted slices shape: {result.shape}")

    # Clear memory
    del all_slices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def calculate_slice_counts(validation_volumes, training_volumes):
    """
    Calculate the actual slice counts for validation and training data
    using the smallest dimension of each volume.

    Args:
        validation_volumes: List of validation 3D volumes
        training_volumes: List of training 3D volumes

    Returns:
        Tuple of (validation_slice_count, training_slice_count)
    """
    val_slices = 0

    for vol in validation_volumes:
        if vol.dim() == 5:  # 3D volume [B, C, D, H, W]
            # Find the smallest dimension
            B, C, D, H, W = vol.shape
            dims = [D, H, W]
            smallest_dim = min(dims)
            val_slices += smallest_dim

    train_slices = 0
    for vol in training_volumes:
        if vol.dim() == 5:  # 3D volume [B, C, D, H, W]
            # Find the smallest dimension
            B, C, D, H, W = vol.shape
            dims = [D, H, W]
            smallest_dim = min(dims)
            train_slices += smallest_dim

    return val_slices, train_slices


def evaluate_slice_based_fid(
    model,
    validation_data,
    device,
    train_exvivo_data=None,
    model_name=None,
    base_dir=None,
):
    """
    Evaluate FID with memory-efficient weighted average for combined score
    and visualization of middle slices. Now with consistent slice extraction
    and masking of outputs.

    Args:
        model: CycleGAN model
        validation_data: DataLoader for validation samples
        device: Computation device (CPU/GPU)
        train_exvivo_data: DataLoader or list of training exvivo samples
        model_name: Name of the model (for visualization)
        base_dir: Base directory for saving visualizations

    Returns:
        dict: Dictionary with FID scores
    """
    print("\n** Starting Enhanced FID Evaluation with Masking **")
    metrics_calc = DomainMetricsCalculator(device=device)

    # Collect ALL validation volumes
    all_real_B = []
    all_fake_B = []
    all_masks = []

    # Process each validation sample
    print("\n== Processing all validation volumes ==")
    for i, data in enumerate(
        tqdm(validation_data, desc="Processing validation volumes")
    ):
        with torch.no_grad():
            model.set_input(data)

            # Create mask from in vivo input
            invivo_mask = create_mask_from_invivo(model.real_A)

            model.test()

            # Apply mask to output
            fake_B_masked = apply_mask_to_exvivo(model.fake_B, invivo_mask)
            model.fake_B = fake_B_masked

            # Get the current validation volume's real and fake data
            real_B = model.real_B  # validation exvivo
            fake_B = model.fake_B  # generated masked exvivo

            # Make sure we have valid data before adding
            if real_B is not None and fake_B is not None:
                all_real_B.append(real_B.detach().clone())
                all_fake_B.append(fake_B.detach().clone())
                all_masks.append(invivo_mask.detach().clone())
                print(f"Added validation volume {i + 1} with shape: {real_B.shape}")

                # Save visualizations if requested
                if model_name is not None and base_dir is not None:
                    save_middle_slice_visualizations(
                        model, data, model_name, i, base_dir
                    )
            else:
                print(f"Skipping validation volume {i + 1} due to None output")

    if not all_real_B or not all_fake_B:
        print("Error: No valid validation volumes were processed")
        return {
            "fid_val": float("inf"),
            "fid_train": float("inf"),
            "fid_combined": float("inf"),
        }

    # Combine all volumes
    print(f"\nCombining {len(all_real_B)} validation volumes")

    # Concatenate along batch dimension
    combined_real_B = torch.cat(all_real_B, dim=0)
    combined_fake_B = torch.cat(all_fake_B, dim=0)
    combined_masks = torch.cat(all_masks, dim=0)

    print(f"Combined validation real data shape: {combined_real_B.shape}")
    print(f"Combined validation fake data shape: {combined_fake_B.shape}")
    print(f"Combined validation masks shape: {combined_masks.shape}")

    # Extract 2D slices along the consistent smallest dimension for FID calculation
    print("\n== Extracting slices for FID calculation ==")
    validation_real_slices = extract_slices_from_volumes(all_real_B, device)
    validation_fake_slices = extract_slices_from_volumes(all_fake_B, device)

    print(f"Extracted validation real slices: {validation_real_slices.shape}")
    print(f"Extracted validation fake slices: {validation_fake_slices.shape}")

    # Calculate FID against validation exvivo data
    print("\n== Calculating FID against validation exvivo data ==")
    fid_val_score = metrics_calc.calculate_slice_based_fid(
        validation_real_slices, validation_fake_slices
    )
    result = {"fid_val": fid_val_score}
    print(f"FID against validation exvivo: {fid_val_score:.3f}")

    # Now process training exvivo data (if available)
    if train_exvivo_data is not None:
        print("\n== Processing training exvivo data for additional FID reference ==")

        # Calculate actual slice counts based on the smallest dimension of each volume
        val_slices, train_slices = calculate_slice_counts(all_real_B, train_exvivo_data)

        print(f"Calculated actual slice counts:")
        print(f"  - Validation slices: {val_slices}")
        print(f"  - Training slices: {train_slices}")

        # Process one random training volume for the training FID score
        if len(train_exvivo_data) > 0:
            try:
                # Use a representative volume
                idx = len(train_exvivo_data) // 2
                training_vol = train_exvivo_data[idx : idx + 1]

                # Extract slices from this volume
                training_real_slices = extract_slices_from_volumes(training_vol, device)
                print(
                    f"Extracted {training_real_slices.shape[0]} slices from representative training volume"
                )

                # Calculate FID against training data (using masked fake outputs)
                fid_train_score = metrics_calc.calculate_slice_based_fid(
                    training_real_slices, validation_fake_slices
                )

                result["fid_train"] = fid_train_score

                # Calculate combined FID using weighted average
                total_slices = val_slices + train_slices
                val_weight = val_slices / total_slices
                train_weight = train_slices / total_slices

                print(f"\n== Calculating weighted combined FID score ==")
                print(f"  - Validation slices: {val_slices} (weight: {val_weight:.3f})")
                print(
                    f"  - Training slices: {train_slices} (weight: {train_weight:.3f})"
                )

                combined_fid = (
                    val_weight * fid_val_score + train_weight * fid_train_score
                )

                result["fid_combined"] = combined_fid

                print(f"FID against validation exvivo: {fid_val_score:.3f}")
                print(f"FID against training exvivo: {fid_train_score:.3f}")
                print(f"Weighted combined FID: {combined_fid:.3f}")

                # Clean up
                del training_real_slices

            except Exception as e:
                print(f"Error calculating training FID: {e}")
                import traceback

                traceback.print_exc()

                result["fid_train"] = float("inf")
                result["fid_combined"] = float("inf")
        else:
            print("Warning: No training volumes available")
            result["fid_train"] = float("inf")
            result["fid_combined"] = float("inf")

    # Final cleanup
    del validation_real_slices
    del validation_fake_slices
    del combined_masks, combined_real_B, combined_fake_B, all_masks

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def ensure_compatible_dimensions(tensor_a, tensor_b, device):
    """
    Ensure that two tensors have compatible dimensions for concatenation
    by adding or removing dimensions as needed.

    Args:
        tensor_a: First tensor
        tensor_b: Second tensor
        device: Device to use

    Returns:
        Tuple of tensors with compatible dimensions
    """
    import torch.nn.functional as F

    # Get shapes
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape

    # Print shapes for debugging
    print(f"Shape A: {shape_a}")
    print(f"Shape B: {shape_b}")

    # Check if dimensions match
    if len(shape_a) == len(shape_b):
        # If the channel dimensions match
        if shape_a[1] == shape_b[1]:
            # If spatial dimensions differ, resize
            if shape_a[2:] != shape_b[2:]:
                print(
                    f"Spatial dimensions differ. Resizing tensor B to match tensor A spatial dimensions."
                )
                tensor_b = F.interpolate(
                    tensor_b,
                    size=shape_a[2:],
                    mode="bilinear" if len(shape_a) == 4 else "trilinear",
                    align_corners=True,
                )
            return tensor_a, tensor_b

    # If we get here, we need to handle dimensional mismatch
    # Case 1: Different number of dimensions
    if len(shape_a) != len(shape_b):
        print(f"Dimension count mismatch: A has {len(shape_a)}, B has {len(shape_b)}")

        # Determine which has more dimensions
        if len(shape_a) > len(shape_b):
            # A has more dimensions, we need to add dimensions to B
            print(f"Adding dimensions to tensor B")

            # Reshape B to match A's dimension count
            if len(shape_a) == 5 and len(shape_b) == 4:  # 5D vs 4D
                # Add a dimension in the middle (assuming B is [B,C,H,W] and need to become [B,C,D,H,W])
                b, c, h, w = shape_b
                # Reshape to [B, C, 1, H, W]
                tensor_b = tensor_b.unsqueeze(2)
                # Repeat to match A's depth dimension
                tensor_b = tensor_b.repeat(1, 1, shape_a[2], 1, 1)
                print(f"Reshaped B to {tensor_b.shape}")

            elif len(shape_a) == 4 and len(shape_b) == 3:  # 4D vs 3D
                # Add a dimension (assuming B is [B,H,W] and need to become [B,C,H,W])
                if shape_b[0] == shape_a[0]:  # Batch dimensions match
                    # B is [B,H,W], reshape to [B,1,H,W]
                    tensor_b = tensor_b.unsqueeze(1)
                    print(f"Reshaped B to {tensor_b.shape}")

            # Now resize to match spatial dimensions
            if tensor_b.shape[2:] != shape_a[2:]:
                print(f"Resizing spatial dimensions of B to match A")
                tensor_b = F.interpolate(
                    tensor_b,
                    size=shape_a[2:],
                    mode="bilinear" if len(shape_a) == 4 else "trilinear",
                    align_corners=True,
                )
                print(f"Final B shape: {tensor_b.shape}")
        else:
            # B has more dimensions, add dimensions to A
            print(f"Adding dimensions to tensor A")

            # Similar logic as above but for tensor A
            if len(shape_b) == 5 and len(shape_a) == 4:
                a, c, h, w = shape_a
                tensor_a = tensor_a.unsqueeze(2)
                tensor_a = tensor_a.repeat(1, 1, shape_b[2], 1, 1)
                print(f"Reshaped A to {tensor_a.shape}")

            elif len(shape_b) == 4 and len(shape_a) == 3:
                if shape_a[0] == shape_b[0]:
                    tensor_a = tensor_a.unsqueeze(1)
                    print(f"Reshaped A to {tensor_a.shape}")

            # Resize spatial dimensions
            if tensor_a.shape[2:] != shape_b[2:]:
                print(f"Resizing spatial dimensions of A to match B")
                tensor_a = F.interpolate(
                    tensor_a,
                    size=shape_b[2:],
                    mode="bilinear" if len(shape_b) == 4 else "trilinear",
                    align_corners=True,
                )
                print(f"Final A shape: {tensor_a.shape}")

    # Case: Same number of dimensions but channel mismatch
    elif shape_a[1] != shape_b[1]:
        print(f"Channel dimension mismatch: A has {shape_a[1]}, B has {shape_b[1]}")

        # Handle channel mismatch - prioritize the one with fewer channels
        if shape_a[1] < shape_b[1]:
            # Reduce B's channels
            print(f"Reducing tensor B's channels to match tensor A")
            if shape_b[1] > 1:
                # Take first N channels or average channels
                tensor_b = tensor_b[:, : shape_a[1], ...]
                print(f"Reduced B to shape {tensor_b.shape}")
        else:
            # Reduce A's channels
            print(f"Reducing tensor A's channels to match tensor B")
            if shape_a[1] > 1:
                tensor_a = tensor_a[:, : shape_b[1], ...]
                print(f"Reduced A to shape {tensor_a.shape}")

        # Now resize spatial dimensions if needed
        if tensor_a.shape[2:] != tensor_b.shape[2:]:
            print(f"Resizing tensor B's spatial dimensions to match tensor A")
            tensor_b = F.interpolate(
                tensor_b,
                size=tensor_a.shape[2:],
                mode="bilinear" if len(shape_a) == 4 else "trilinear",
                align_corners=True,
            )
            print(f"Final B shape: {tensor_b.shape}")

    # Final check to make sure dimensions are compatible
    if tensor_a.shape[1:] != tensor_b.shape[1:]:
        print(f"WARNING: Failed to make dimensions compatible!")
        print(f"A shape: {tensor_a.shape}, B shape: {tensor_b.shape}")
        print("Performing final emergency reshape of B to match A exactly...")

        # Create new tensor B with A's shape, filled with zeros
        new_tensor_b = torch.zeros_like(tensor_a[: tensor_b.shape[0]], device=device)

        # Copy as much data as possible
        min_channels = min(tensor_a.shape[1], tensor_b.shape[1])

        # Copy channel by channel to avoid issues
        for c in range(min_channels):
            try:
                # Get compatible spatial slices
                min_dims = [
                    min(tensor_a.shape[i + 2], tensor_b.shape[i + 2])
                    for i in range(len(tensor_a.shape) - 2)
                ]
                slices = tuple(slice(0, d) for d in min_dims)

                # Try to copy what we can
                if len(min_dims) == 2:  # 2D
                    new_tensor_b[:, c, : min_dims[0], : min_dims[1]] = tensor_b[
                        :, c, : min_dims[0], : min_dims[1]
                    ]
                elif len(min_dims) == 3:  # 3D
                    new_tensor_b[:, c, : min_dims[0], : min_dims[1], : min_dims[2]] = (
                        tensor_b[:, c, : min_dims[0], : min_dims[1], : min_dims[2]]
                    )
            except Exception as e:
                print(f"Error copying channel {c}: {e}")
                continue

        tensor_b = new_tensor_b
        print(f"Emergency reshape complete. Final B shape: {tensor_b.shape}")

    return tensor_a, tensor_b


def log_fid_to_wandb(model_name, results, opt):
    """
    Log FID evaluation results and additional metrics to Weights & Biases.

    Args:
        model_name: Name of the model being evaluated
        results: Dictionary containing FID scores and additional metrics
        opt: Options object
    """
    import wandb
    import os
    from utils.utils import _sanitize_wandb_values

    # Skip if wandb not enabled
    if not hasattr(opt, "use_wandb") or not opt.use_wandb:
        return

    try:
        # Extract the run ID (3-character prefix) from model name
        run_id = model_name.split("_")[0]
        config_name = model_name.split("_")[1]
        full_name = model_name

        print(
            f"Logging evaluation results to WandB for run {run_id}, config {config_name}"
        )

        # Check if there's an existing wandb ID file for this model
        wandb_id_file = os.path.join(opt.checkpoints_dir, model_name, "wandb_id.txt")
        wandb_id = None

        if os.path.exists(wandb_id_file):
            with open(wandb_id_file, "r") as f:
                wandb_id = f.read().strip()
            print(f"Found existing wandb ID: {wandb_id}")

        # Determine if we need to create a new run or resume existing
        if wandb.run is None:
            print("Initializing new wandb connection")

            # Set up wandb config
            config = {
                "run_id": run_id,
                "config_name": config_name,
                "evaluation_type": "full_metrics",
                "model_name": model_name,
            }

            # Add any relevant options from opt
            for key, value in vars(opt).items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    config[key] = value

            # Initialize wandb
            os.environ["WANDB_START_METHOD"] = "thread"
            # Same wandb project as in training
            wandb_project = getattr(opt, "wandb_project", "prostate_SR-domain_cor")

            try:
                # Try to authenticate with the key used in the training code
                wandb.login(key="cde9483f01d3d4c883d033dbde93150f7d5b22d5", timeout=60)
            except Exception as e:
                print(f"Warning: WandB login failed: {e}, trying to proceed anyway")

            init_mode = "online" if opt.use_wandb else "disabled"

            # Initialize with the same run if ID exists
            if wandb_id:
                try:
                    wandb.init(
                        project=wandb_project,
                        group=getattr(opt, "group_name", "experiments"),
                        mode=init_mode,
                        name=full_name,
                        id=wandb_id,
                        resume="allow",
                        config=config,
                    )
                except Exception as e:
                    print(f"Failed to resume wandb run: {e}, creating new run")
                    wandb_id = None

            # Create new run if no ID or resuming failed
            if not wandb_id or wandb.run is None:
                wandb.init(
                    project=wandb_project,
                    group=getattr(opt, "group_name", "experiments"),
                    mode=init_mode,
                    name=full_name,
                    config=config,
                )

        # Prepare metrics for logging with 'eval/' prefix
        metrics = {
            "eval/fid_val": results.get("fid_val", float("inf")),
            "eval/fid_train": results.get("fid_train", float("inf")),
            "eval/fid_combined": results.get("fid_combined", float("inf")),
            "eval/configuration": config_name,
            "eval/run_id": run_id,
        }

        # Add additional metrics with eval/ prefix
        if "psnr" in results:
            metrics["eval/psnr"] = results.get("psnr", 0.0)
        if "ssim" in results:
            metrics["eval/ssim"] = results.get("ssim", 0.0)
        if "lpips" in results:
            metrics["eval/lpips"] = results.get("lpips", 1.0)
        if "ncc" in results:
            metrics["eval/ncc"] = results.get("ncc", 0.0)

        # Add any error messages
        if "error" in results and results["error"]:
            metrics["eval/error"] = results["error"]

        # Sanitize values to prevent NaN/Inf issues
        sanitized_metrics = _sanitize_wandb_values(metrics)

        # Log to wandb
        wandb.log(sanitized_metrics)
        print(f"Successfully logged evaluation metrics to wandb: {sanitized_metrics}")

    except Exception as e:
        print(f"Error logging to wandb: {e}")
        import traceback

        traceback.print_exc()


def parse_model_name(model_name, opt):
    """Parse model parameters from name"""
    if "_ngf" in model_name and "_ndf" in model_name:
        try:
            parts = model_name.split("_")
            for part in parts:
                if part.startswith("ngf"):
                    opt.ngf = int(part[3:])
                elif part.startswith("ndf"):
                    opt.ndf = int(part[3:])
                elif part.startswith("patch"):
                    try:
                        patch_dims = part.split("patch")[1].split("_")
                        if len(patch_dims) >= 3:
                            # Store as a list for compatibility
                            opt.patch_size = [
                                int(patch_dims[0]),
                                int(patch_dims[1]),
                                int(patch_dims[2]),
                            ]
                    except Exception as e:
                        print(f"Error parsing patch size from {part}: {e}")

            opt.use_stn = "stn" in model_name.lower()
            opt.use_residual = "residual" in model_name.lower()

            print(
                f"Parsed model parameters: ngf={opt.ngf}, ndf={opt.ndf}, patch_size={getattr(opt, 'patch_size', 'not set')}"
            )
            print(
                f"Model features: use_stn={opt.use_stn}, use_residual={opt.use_residual}, use_full_attention={opt.use_full_attention}"
            )

        except Exception as e:
            print(f"Error parsing model parameters from name: {e}")

    return opt


class ExvivoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, max_samples=50):
        from utils.utils import lstFiles
        import os
        import torch

        self.exvivo_files = lstFiles(os.path.join(data_path, "exvivo"))
        self.max_samples = max_samples

        if len(self.exvivo_files) > max_samples:
            indices = (
                torch.linspace(0, len(self.exvivo_files) - 1, max_samples)
                .long()
                .tolist()
            )
            self.exvivo_files = [self.exvivo_files[i] for i in indices]

        print(
            f"Created ExvivoDataset with {len(self.exvivo_files)} files from {data_path}/exvivo"
        )

    def __len__(self):
        return len(self.exvivo_files)

    def __getitem__(self, idx):
        import SimpleITK as sitk
        import torch
        import numpy as np

        try:
            file_path = self.exvivo_files[idx]
            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)
            reader.SetLoadPrivateTags(False)
            image = reader.Execute()

            # Memory optimization: Convert directly to float32 to avoid double conversion
            image_array = sitk.GetArrayFromImage(image).astype(np.float32)
            image_tensor = torch.from_numpy(image_array)

            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            # In-place normalization to [-1, 1] to save memory
            image_min = image_tensor.min()
            image_max = image_tensor.max()

            if image_max - image_min > 1e-5:
                image_tensor.sub_(image_min)  # In-place subtraction
                image_tensor.div_(image_max - image_min)  # In-place division
                image_tensor.mul_(2).sub_(1)  # In-place multiplication and subtraction
            else:
                image_tensor.zero_()  # In-place zeroing

            return image_tensor

        except Exception as e:
            print(f"Error loading file {self.exvivo_files[idx]}: {e}")
            return torch.zeros((1, 64, 64, 64), dtype=torch.float32)


def load_train_exvivo_data(opt):
    """
    Load and preprocess all exvivo data from the training set with caching
    """
    global CACHED_TRAIN_EXVIVO_DATA

    # Return cached data if available
    if CACHED_TRAIN_EXVIVO_DATA is not None and len(CACHED_TRAIN_EXVIVO_DATA) > 0:
        print(f"Using cached training data ({len(CACHED_TRAIN_EXVIVO_DATA)} volumes)")
        return CACHED_TRAIN_EXVIVO_DATA

    import os
    import torch
    import SimpleITK as sitk
    import numpy as np
    from utils.utils import lstFiles
    from tqdm import tqdm

    train_path = opt.data_path
    exvivo_path = os.path.join(train_path, "exvivo")

    if not os.path.exists(exvivo_path):
        print(f"Warning: Training exvivo path does not exist: {exvivo_path}")
        return []

    try:
        # Get all exvivo files
        exvivo_files = lstFiles(exvivo_path)
        print(f"Found {len(exvivo_files)} exvivo files in training set")

        # Use a generator to process files one at a time
        all_tensors = []
        for i, file_path in enumerate(
            tqdm(exvivo_files, desc="Loading training exvivo files")
        ):
            try:
                reader = sitk.ImageFileReader()
                reader.SetFileName(file_path)
                reader.SetLoadPrivateTags(False)
                image = reader.Execute()

                # Get the original image size
                original_size = image.GetSize()

                # Convert to tensor, directly to float32
                image_array = sitk.GetArrayFromImage(image).astype(np.float32)

                # Convert to tensor
                image_tensor = torch.from_numpy(image_array)

                # Add channel dimension if needed
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)

                # Normalize to [-1, 1] range in-place
                image_min = image_tensor.min()
                image_max = image_tensor.max()

                if image_max - image_min > 1e-5:
                    image_tensor.sub_(image_min)
                    image_tensor.div_(image_max - image_min)
                    image_tensor.mul_(2).sub_(1)
                else:
                    image_tensor.zero_()

                # Add batch dimension
                image_tensor = image_tensor.unsqueeze(0)

                all_tensors.append(image_tensor)

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

            # Force garbage collection after each file
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"Successfully loaded {len(all_tensors)} training exvivo volumes")

        # Cache the result
        CACHED_TRAIN_EXVIVO_DATA = all_tensors

        return all_tensors

    except Exception as e:
        print(f"Error loading training exvivo data: {e}")
        import traceback

        traceback.print_exc()
        return []


def evaluate_model(
    model_name, opt, validation_data, device, train_exvivo_data=None, epoch="latest"
):
    """
    Evaluate a model and return FID scores and additional metrics.

    Args:
        model_name: Name of the model to evaluate
        opt: Options
        validation_data: DataLoader for validation samples
        device: Computation device (CPU/GPU)
        train_exvivo_data: Optional list of training exvivo data
        epoch: Which epoch to evaluate

    Returns:
        dict: Dictionary with FID scores and additional metrics
    """
    print(f"\n{'=' * 50}")
    print(f"Evaluating model: {model_name}")
    print(f"Using checkpoint: {epoch}")
    print(f"{'=' * 50}")

    # Reset CUDA cache before each model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    opt.name = model_name
    opt = parse_model_name(model_name, opt)

    # Set which epoch to use
    opt.which_epoch = epoch

    # Set base directory for visualizations
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    opt.use_residual = True
    opt.n_layers_D = 4
    opt.mixed_precision = True
    try:
        # Initialize model with memory optimization settings
        print("Initializing model with memory optimization settings...")

        # Try to enable gradient checkpointing if supported to reduce memory usage
        opt.use_gradient_checkpointing = True

        model = CycleGANModel()
        model.initialize(opt)

        if torch.cuda.is_available():
            opt.gpu_ids = [0]
        else:
            opt.gpu_ids = []

        model.device = device

        # Enable eval mode to reduce memory usage
        model.eval()

        print(f"Loading model from: checkpoints/{model_name}/{epoch}_net_G_A.pth")
        model.load_networks(epoch)

        # Calculate FID scores with visualization
        print("Running FID evaluation with memory optimizations...")
        fid_result = evaluate_slice_based_fid(
            model, validation_data, device, train_exvivo_data, model_name, base_dir
        )

        # Calculate additional metrics
        print("Calculating additional metrics (PSNR, SSIM, LPIPS, NCC)...")
        additional_metrics = calculate_additional_metrics_for_validation(
            model, validation_data, device
        )

        # Combine all metrics
        result = {**fid_result, **additional_metrics}

        # Log results to wandb
        log_fid_to_wandb(model_name, result, opt)

        # Clean up thoroughly after evaluation
        if hasattr(model, "netG_A") and model.netG_A is not None:
            del model.netG_A
        if hasattr(model, "netG_B") and model.netG_B is not None:
            del model.netG_B
        if hasattr(model, "netD_A") and model.netD_A is not None:
            del model.netD_A
        if hasattr(model, "netD_B") and model.netD_B is not None:
            del model.netD_B

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result

    except Exception as e:
        print(f"Error evaluating model {model_name} (epoch {epoch}): {e}")
        import traceback

        traceback.print_exc()

        # Create error result with all metrics
        error_result = {
            "fid_val": float("inf"),
            "fid_train": float("inf"),
            "fid_combined": float("inf"),
            "psnr": 0.0,
            "ssim": 0.0,
            "lpips": 1.0,
            "ncc": 0.0,
            "error": str(e),
        }

        # Still try to log the error to wandb
        log_fid_to_wandb(model_name, error_result, opt)

        return error_result


def save_results_to_csv(results, output_path="fid_evaluation_results.csv"):
    """

    Save evaluation results to a CSV file


    Args:

        results: Dictionary mapping model names to FID scores and additional metrics

        output_path: Path to output CSV file

    """

    try:

        # Make sure the directory exists

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", newline="") as csvfile:

            fieldnames = [
                "model_name",
                "fid_val",
                "fid_train",
                "fid_combined",
                "psnr",
                "ssim",
                "lpips",
                "ncc",
                "error",
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for model_name, scores in results.items():
                row = {
                    "model_name": model_name,
                    "fid_val": scores.get("fid_val", float("inf")),
                    "fid_train": scores.get("fid_train", float("inf")),
                    "fid_combined": scores.get("fid_combined", float("inf")),
                    "psnr": scores.get("psnr", 0.0),
                    "ssim": scores.get("ssim", 0.0),
                    "lpips": scores.get("lpips", 1.0),
                    "ncc": scores.get("ncc", 0.0),
                    "error": scores.get("error", ""),
                }

                writer.writerow(row)

        print(f"Results saved to {os.path.abspath(output_path)}")

    except Exception as e:

        print(f"Error saving results to CSV: {e}")

        import traceback

        traceback.print_exc()


# Update the process_csv_models function to save intermediate results with all metrics


def process_csv_models(csv_path, opt, validation_data, device, train_exvivo_data=None):
    """

    Process models specified in a CSV file


    Args:

        csv_path: Path to CSV file with model configurations

        opt: Options

        validation_data: DataLoader for validation samples

        device: Computation device (CPU/GPU)

        train_exvivo_data: Optional list of training exvivo data


    Returns:

        dict: Dictionary mapping model names to evaluation metrics

    """

    results = {}

    try:

        # Read CSV file

        df = pd.read_csv(csv_path)

        print(f"Loaded {len(df)} model configurations from {csv_path}")

        # Check required columns

        required_columns = ["model_name", "epoch"]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: CSV is missing required columns: {missing_columns}")

            print(f"Required columns: {required_columns}")

            return results

        # Process each model

        for idx, row in df.iterrows():

            model_name = row["model_name"]

            epoch = row["epoch"]

            print(
                f"\nProcessing CSV entry {idx + 1}/{len(df)}: {model_name}, epoch={epoch}"
            )

            # Generate a unique key that includes both model name and epoch

            result_key = f"{model_name}_epoch{epoch}"

            try:

                # Evaluate the model

                model_results = evaluate_model(
                    model_name, opt, validation_data, device, train_exvivo_data, epoch
                )

                # Store results

                results[result_key] = model_results

            except Exception as e:

                # Record the error and continue with next model

                print(f"Error processing model {model_name} (epoch {epoch}): {e}")

                results[result_key] = {
                    "fid_val": float("inf"),
                    "fid_train": float("inf"),
                    "fid_combined": float("inf"),
                    "psnr": 0.0,
                    "ssim": 0.0,
                    "lpips": 1.0,
                    "ncc": 0.0,
                    "error": str(e),
                }

            # Force cleanup after each model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save intermediate results

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            interim_file = f"interim_results_{timestamp}.csv"

            if hasattr(opt, "output_file") and opt.output_file:
                interim_file = (
                    f"{os.path.splitext(opt.output_file)[0]}_interim_{timestamp}.csv"
                )

            save_results_to_csv(results, interim_file)

            print(f"Saved interim results to {interim_file}")

    except Exception as e:

        print(f"Error processing CSV file: {e}")

        import traceback

        traceback.print_exc()

    return results


def main():
    # Set PyTorch to use expandable segments to reduce memory fragmentation
    import os
    import wandb

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    opt = FIDEvalOptions().parse()
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Initialize wandb if enabled
    if opt.use_wandb:
        try:
            from utils.utils import init_wandb

            init_wandb(opt)
            print("Successfully initialized WandB for FID evaluation")
        except Exception as e:
            print(f"Warning: Failed to initialize WandB: {e}")
            print("Continuing without WandB logging")
            opt.use_wandb = False

    # Configure CUDA for memory efficiency
    if torch.cuda.is_available():
        # Empty cache at start
        torch.cuda.empty_cache()

        # Print initial CUDA memory stats
        print(
            f"Initial CUDA memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB"
        )
        print(
            f"Initial CUDA memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB"
        )

        # Set memory efficient options
        torch.backends.cudnn.benchmark = False  # Disable cudnn benchmarking
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = (
                True  # Allow TF32 for faster computation
            )
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True  # Allow TF32 in cudnn

    # Enable memory profiling if requested
    if opt.memory_profiling:
        try:
            import psutil

            print("Memory profiling enabled")
        except ImportError:
            print("Warning: psutil not available for memory profiling")
            opt.memory_profiling = False

    set_seed(42)

    # Memory optimization settings
    opt.use_full_validation = True
    opt.batch_size = 1  # Process one volume at a time

    # Set additional memory optimization parameters
    opt.slice_batch_size = 8  # Process slices in smaller batches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if we're evaluating from CSV or direct model list
    if opt.csv_path is not None:
        print(f"Using CSV file for model evaluation: {opt.csv_path}")
        if not os.path.exists(opt.csv_path):
            print(f"Error: CSV file not found: {opt.csv_path}")
            sys.exit(1)
    else:
        # Find models to evaluate
        if opt.models is None:
            try:
                model_dirs = [
                    d
                    for d in os.listdir(opt.checkpoints_dir)
                    if os.path.isdir(os.path.join(opt.checkpoints_dir, d))
                    and os.path.exists(
                        os.path.join(
                            opt.checkpoints_dir, d, f"{opt.which_epoch}_net_G_A.pth"
                        )
                    )
                ]
            except Exception as e:
                print(f"Error finding models in checkpoints directory: {e}")
                model_dirs = []
        else:
            model_dirs = opt.models

    try:
        opt.name = "temp_for_dataloader"

        # Setup data loaders with memory-efficient options
        print("Setting up data loaders with memory-efficient options...")
        opt.num_threads = 1  # Reduce number of dataloader workers to save memory
        _, validation_loader = setup_dataloaders(opt)
        print(f"Created validation loader with {len(validation_loader)} samples")

        # Load training exvivo data in a memory-efficient way (with caching)
        print("Loading training exvivo data for combined FID reference...")
        train_exvivo_tensors = load_train_exvivo_data(opt)
        if train_exvivo_tensors:
            print(
                f"Successfully loaded {len(train_exvivo_tensors)} training exvivo volumes"
            )
        else:
            print("Warning: No training exvivo data loaded")

        # Force garbage collection after loading data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error setting up data: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a results directory if it doesn't exist
    results_dir = "../results/fid_evaluation"
    os.makedirs(results_dir, exist_ok=True)

    output_file = (
        opt.output_file
        if opt.output_file
        else f"{results_dir}/fid_evaluation_results_{timestamp}.csv"
    )

    print(f"Results will be saved to: {os.path.abspath(output_file)}")

    # Process either CSV models or direct model list
    if opt.csv_path is not None:
        # Process models from CSV file
        results = process_csv_models(
            opt.csv_path, opt, validation_loader, device, train_exvivo_tensors
        )

        # Save results to specified output file
        save_results_to_csv(results, output_file)

    else:
        # Process individual models
        results = {}
        for model_name in model_dirs:
            model_result = evaluate_model(
                model_name,
                opt,
                validation_loader,
                device,
                train_exvivo_tensors,
                opt.which_epoch,
            )
            results[model_name] = model_result

            # Print results summary including all metrics
            print("\n\nEvaluation Metrics Summary:")
            print("=" * 80)

            # Report all metrics per model
            print("\nAll Metrics:")
            sorted_results = sorted(
                [(name, scores) for name, scores in results.items()],
                key=lambda x: x[1].get("fid_combined", float("inf")),
            )

            # Print header
            print(
                f"{'Model':<30} {'FID':<8} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'NCC':<8}"
            )
            print("-" * 80)

            for model_name, scores in sorted_results:
                print(
                    f"{model_name:<30} "
                    f"{scores.get('fid_combined', float('inf')):<8.3f} "
                    f"{scores.get('psnr', 0.0):<8.3f} "
                    f"{scores.get('ssim', 0.0):<8.3f} "
                    f"{scores.get('lpips', 1.0):<8.3f} "
                    f"{scores.get('ncc', 0.0):<8.3f}"
                )

        # Save results to text file
        text_output = f"{results_dir}/fid_evaluation_results_{timestamp}.txt"
        with open(text_output, "w") as f:
            f.write("FID Score Summary:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Evaluation performed using {opt.which_epoch} checkpoint\n")
            f.write(
                f"Combined FID: Weighted average of validation and training FID scores\n"
            )
            f.write("=" * 50 + "\n\n")

            f.write("Combined FID Scores:\n")
            for model_name, fid in sorted_results:
                f.write(f"{model_name}: {fid:.3f}\n")

        # Save detailed results to CSV
        save_results_to_csv(results, output_file)

    # Close WandB
    if opt.use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Error closing WandB: {e}")

    # Clear training data cache at the end
    global CACHED_TRAIN_EXVIVO_DATA
    CACHED_TRAIN_EXVIVO_DATA = None
    gc.collect()


if __name__ == "__main__":
    main()
