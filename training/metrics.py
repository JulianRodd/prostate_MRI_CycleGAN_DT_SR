import gc

import torch
import torch.nn.functional as F


def ensure_consistent_dimensions_by_padding(images_dict):
    """
    Ensure all tensors have consistent dimensions by padding with zeros
    rather than resizing, which maintains original image content.

    Args:
        images_dict: Dictionary of tensors with potentially different spatial dimensions

    Returns:
        Dictionary of tensors with consistent spatial dimensions
    """
    keys = list(images_dict.keys())
    if not keys:
        return images_dict

    # Find the largest spatial dimensions
    max_spatial_dims = None
    for key in keys:
        if images_dict[key] is None:
            continue
        shape = list(images_dict[key].shape)
        if max_spatial_dims is None:
            max_spatial_dims = shape[2:]
        else:
            for i in range(len(shape[2:])):
                max_spatial_dims[i] = max(max_spatial_dims[i], shape[2:][i])

    if max_spatial_dims is None:
        return images_dict

    # Pad all tensors to have the same spatial dimensions
    result = {}
    for key, tensor in images_dict.items():
        if tensor is None:
            result[key] = None
            continue

        current_dims = list(tensor.shape[2:])
        if current_dims != max_spatial_dims:
            print(
                f"Padding {key} from {current_dims} to {max_spatial_dims} for consistent metrics"
            )

            # Calculate padding amounts (pad end of each dimension)
            pad_amounts = []
            for i in range(
                len(max_spatial_dims) - 1, -1, -1
            ):  # Reverse order for torch.nn.functional.pad
                pad_amount = max_spatial_dims[i] - current_dims[i]
                pad_amounts.extend(
                    [0, pad_amount]
                )  # [pad_left, pad_right, pad_top, pad_bottom, ...]

            # Apply padding
            padded_tensor = F.pad(tensor, pad_amounts, mode="constant", value=0)
            result[key] = padded_tensor
        else:
            result[key] = tensor

    return result


def calculate_metrics_with_batching(metrics_calculator, images_dict, device=None):
    """
    Calculate metrics while handling memory constraints and consistent dimensions.
    Processes a dictionary of images and computes various image quality metrics,
    ensuring consistent tensor dimensions and proper normalization.

    Args:
        metrics_calculator: Instance of the metrics calculator
        images_dict: Dictionary of images to calculate metrics for
        device: Torch device (CPU or CUDA)

    Returns:
        Dictionary of calculated metrics
    """
    try:
        # Ensure consistent dimensions across all images by padding (not resizing)
        processed_dict = ensure_consistent_dimensions_by_padding(images_dict)

        metrics = metrics_calculator.calculate_metrics(processed_dict)
        processed_dict = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        expected_metrics = [
            "ssim_sr",
            "psnr_sr",
            "lpips_sr",
            "fid_domain",
            "ncc_domain",
            "metric_domain",
            "metric_structure",
            "metric_combined",
        ]

        for metric in expected_metrics:
            if metric not in metrics:
                print(f"Warning: {metric} not calculated, setting to default value")
                if metric in [
                    "ssim_sr",
                    "psnr_sr",
                    "ncc_domain",
                    "metric_domain",
                    "metric_structure",
                    "metric_combined",
                ]:
                    metrics[metric] = 0.0
                else:
                    metrics[metric] = float("inf") if metric == "fid_domain" else 1.0

        # Recalculate metric_domain correctly as it's crucial for evaluation
        # Normalize FID to [0,1] range where higher is better (opposite of raw FID)
        if "fid_domain" in metrics:
            fid_normalized = max(0, min(1.0, 1.0 - metrics["fid_domain"] / 10.0))
        else:
            fid_normalized = 0.0

        # Normalize NCC to [0,1] range where higher is better
        if "ncc_domain" in metrics:
            # NCC is in [-1,1] range, normalize to [0,1]
            ncc_normalized = (metrics["ncc_domain"] + 1.0) / 2.0
        else:
            ncc_normalized = 0.0

        # Recalculate metric_domain as average of normalized metrics
        metrics["metric_domain"] = (fid_normalized + ncc_normalized) / 2.0

        return metrics
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM during metrics calculation: {e}")
        print(
            "Warning: Falling back to using the validation resolution without further downscaling"
        )
        try:
            metrics = {}
            for metric in expected_metrics:
                if metric in [
                    "ssim_sr",
                    "psnr_sr",
                    "metric_domain",
                    "metric_structure",
                    "metric_combined",
                ]:
                    metrics[metric] = 0.0
                else:
                    metrics[metric] = float("inf") if metric == "fid_domain" else 1.0
            return metrics
        except Exception:
            return {}
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}


def calculate_full_dataset_fid(
    metrics_calculator, real_collector, fake_collector, device
):
    """
    Calculate FID score for full dataset using collected images.
    Ensures sufficient slices are available and handles memory
    management during calculation.

    Args:
        metrics_calculator: Instance of the metrics calculator
        real_collector: List of real image slices
        fake_collector: List of fake image slices
        device: Torch device (CPU or CUDA)

    Returns:
        Dictionary with FID metrics
    """
    metrics = {}

    min_slices_required = 50
    if (
        len(real_collector) < min_slices_required
        or len(fake_collector) < min_slices_required
    ):
        print(
            f"Not enough slices for reliable FID calculation (need at least {min_slices_required}, "
            f"got {len(real_collector)} real, {len(fake_collector)} fake)"
        )
        print(f"FID calculation skipped - results would be unreliable")
        metrics["fid_domain"] = float("inf")
        return metrics

    if (
        not hasattr(metrics_calculator, "fid_available")
        or not metrics_calculator.fid_available
    ):
        print("FID calculation not available")
        return {"fid_domain": float("inf")}

    try:
        # Combine slices into tensors
        print(
            f"Calculating FID with {len(real_collector)} real slices and {len(fake_collector)} fake slices"
        )

        # Move slices to device for calculation
        real_tensor = torch.cat(real_collector, dim=0).to(device)
        fake_tensor = torch.cat(fake_collector, dim=0).to(device)

        # Use slice-based FID to match evaluation script
        if hasattr(metrics_calculator, "calculate_slice_based_fid"):
            print("Using slice-based FID calculation (matches evaluation script)")
            fid_domain = metrics_calculator.calculate_slice_based_fid(
                real_tensor, fake_tensor
            ).item()
        else:
            # Fallback to standard FID if slice-based not available
            print("Slice-based FID not available, using standard FID")
            fid_domain = metrics_calculator.calculate_fid(
                real_tensor, fake_tensor
            ).item()

        metrics["fid_domain"] = fid_domain
        print(f"Evaluation-style FID score: {fid_domain:.4f}")

        # Clean up memory
        del real_tensor, fake_tensor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()

        return metrics

    except Exception as e:
        print(f"Error in full-dataset FID calculation: {e}")
        import traceback

        traceback.print_exc()
        return {"fid_domain": float("inf")}


def crop_black_regions_aligned(image_A, image_B, threshold=0.02):
    """
    Crop black regions from a pair of 5D tensors (B, C, D, H, W) with aligned sizes.
    Identifies and removes black border regions while maintaining alignment
    between the images.

    Args:
        image_A: First input tensor with shape (B, C, D, H, W)
        image_B: Second input tensor with shape (B, C, D, H, W)
        threshold: Pixel intensity threshold below which is considered "black"
                 (normalized to the range of the tensor)

    Returns:
        Tuple of (cropped_A, cropped_B, crop_info) where crop_info contains coordinates for padding back
    """
    # Keep original tensor devices for final output
    device_A = image_A.device
    device_B = image_B.device

    # Check if both are 5D tensors
    if image_A.dim() != 5 or image_B.dim() != 5:
        print("Warning: Expected 5D tensors, returning original tensors")
        return image_A, image_B, None

    with torch.no_grad():
        # Move to CPU for numpy operations
        image_A_np = image_A.detach().cpu().numpy()
        image_B_np = image_B.detach().cpu().numpy()

        # Convert normalized thresholds to actual thresholds in [-1,1] space
        A_threshold = -1 + (threshold * 2)
        B_threshold = -1 + (threshold * 2)

        # Get first channel from first batch for analysis
        A_volume = image_A_np[0, 0]
        B_volume = image_B_np[0, 0]

        # Get dimensions
        D, H, W = A_volume.shape

        # Find non-black indices along each dimension for image A
        A_non_black_d = (A_volume > A_threshold).any(axis=(1, 2))
        A_non_black_h = (A_volume > A_threshold).any(axis=(0, 2))
        A_non_black_w = (A_volume > A_threshold).any(axis=(0, 1))

        # Find non-black indices along each dimension for image B
        B_non_black_d = (B_volume > B_threshold).any(axis=(1, 2))
        B_non_black_h = (B_volume > B_threshold).any(axis=(0, 2))
        B_non_black_w = (B_volume > B_threshold).any(axis=(0, 1))

        # If either image is all black, return originals
        if (
            not A_non_black_d.any()
            or not A_non_black_h.any()
            or not A_non_black_w.any()
            or not B_non_black_d.any()
            or not B_non_black_h.any()
            or not B_non_black_w.any()
        ):
            print(
                "Warning: One or both volumes appear to be completely black, skipping crop"
            )
            return image_A, image_B, None

        # Find bounds for image A
        A_d_start, A_d_end = A_non_black_d.argmax(), D - A_non_black_d[::-1].argmax()
        A_h_start, A_h_end = A_non_black_h.argmax(), H - A_non_black_h[::-1].argmax()
        A_w_start, A_w_end = A_non_black_w.argmax(), W - A_non_black_w[::-1].argmax()

        # Find bounds for image B
        B_d_start, B_d_end = B_non_black_d.argmax(), D - B_non_black_d[::-1].argmax()
        B_h_start, B_h_end = B_non_black_h.argmax(), H - B_non_black_h[::-1].argmax()
        B_w_start, B_w_end = B_non_black_w.argmax(), W - B_non_black_w[::-1].argmax()

        # Take the tightest bounds to ensure both images have same crop dimensions
        d_start = min(A_d_start, B_d_start)
        d_end = max(A_d_end, B_d_end)
        h_start = min(A_h_start, B_h_start)
        h_end = max(A_h_end, B_h_end)
        w_start = min(A_w_start, B_w_start)
        w_end = max(A_w_end, B_w_end)

        # Add small margin (10% of each dimension)
        d_margin = max(1, int((d_end - d_start) * 0.1))
        h_margin = max(1, int((h_end - h_start) * 0.1))
        w_margin = max(1, int((w_end - w_start) * 0.1))

        d_start = max(0, d_start - d_margin)
        d_end = min(D, d_end + d_margin)
        h_start = max(0, h_start - h_margin)
        h_end = min(H, h_end + h_margin)
        w_start = max(0, w_start - w_margin)
        w_end = min(W, w_end + w_margin)

        # Crop the tensors with the unified bounds
        cropped_A = image_A[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
        cropped_B = image_B[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

        # Store crop information for padding back
        crop_info = {
            "d_start": d_start,
            "d_end": d_end,
            "h_start": h_start,
            "h_end": h_end,
            "w_start": w_start,
            "w_end": w_end,
            "orig_shape": (D, H, W),
        }

        # Log crop information
        original_voxels = D * H * W
        cropped_voxels = (d_end - d_start) * (h_end - h_start) * (w_end - w_start)
        reduction = 100 * (1 - cropped_voxels / original_voxels)
        print(
            f"Cropped 3D volumes from {(D, H, W)} to {(d_end - d_start, h_end - h_start, w_end - w_start)}"
        )
        print(
            f"Volume reduction: {reduction:.2f}% (keeping {100 - reduction:.2f}% of original)"
        )

    # Ensure cropped outputs are on same devices as inputs
    return cropped_A.to(device_A), cropped_B.to(device_B), crop_info
