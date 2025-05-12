import SimpleITK as sitk
import numpy as np
from scipy import ndimage


def create_tissue_mask(
    data: np.ndarray,
    threshold: float = 0.1,
    fill_holes: bool = True,
    smooth_sigma: float = 0.5,
    iterations: int = 3,
) -> np.ndarray:
    """Create a binary mask from non-black pixels."""
    # Create initial mask with threshold
    initial_mask = data > threshold

    if not np.any(initial_mask):
        return initial_mask

    # Apply morphological operations to improve mask quality
    if fill_holes:
        # Convert to SimpleITK for better morphological operations
        mask_sitk = sitk.GetImageFromArray(initial_mask.astype(np.uint8))

        # Apply hole filling and closing
        for _ in range(iterations):
            # Fill holes
            mask_sitk = sitk.BinaryFillhole(mask_sitk, fullyConnected=True)
            # Apply binary closing to smooth edges
            mask_sitk = sitk.BinaryMorphologicalClosing(mask_sitk, [3, 3, 3])

        # Convert back to numpy
        filled_mask = sitk.GetArrayFromImage(mask_sitk) > 0
    else:
        filled_mask = initial_mask

    # Apply smoothing for smoother transitions
    if smooth_sigma > 0:
        float_mask = filled_mask.astype(np.float32)
        smoothed_mask = ndimage.gaussian_filter(float_mask, sigma=smooth_sigma)
        final_mask = smoothed_mask > 0.5
    else:
        final_mask = filled_mask

    return final_mask


def process_mask_slice(
    mask_slice, iterations_dilation=3, iterations_erosion=2, iterations_closing=2
):
    """Process a mask slice with dilation, hole filling, erosion and closing operations."""
    # Apply dilation to connect nearby regions
    dilated = ndimage.binary_dilation(mask_slice, iterations=iterations_dilation)

    # Apply multiple hole filling passes
    filled = ndimage.binary_fill_holes(dilated)
    filled = ndimage.binary_fill_holes(filled)  # Second pass for thoroughness

    # Apply erosion to restore approximate size
    eroded = ndimage.binary_erosion(filled, iterations=iterations_erosion)

    # Apply closing for smoothing edges
    smoothed = ndimage.binary_closing(eroded, iterations=iterations_closing)

    return smoothed


def clean_binary_mask(
    mask, iterations_dilation=3, iterations_erosion=2, iterations_closing=2
):
    """Process a 3D mask with slice-by-slice morphological operations."""
    processed_mask = mask.copy()

    if mask.ndim == 3:
        for z in range(mask.shape[2]):
            processed_mask[:, :, z] = process_mask_slice(
                mask[:, :, z],
                iterations_dilation,
                iterations_erosion,
                iterations_closing,
            )
    else:
        # 2D case
        processed_mask = process_mask_slice(
            mask, iterations_dilation, iterations_erosion, iterations_closing
        )

    return processed_mask
