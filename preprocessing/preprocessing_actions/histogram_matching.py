import ants
import numpy as np


def apply_histogram_matching(
    source_image, reference_image, source_mask=None, reference_mask=None, logger=None
):
    """Apply histogram matching using the appropriate ANTs function."""
    try:
        if hasattr(ants, "histogram_match_image2"):
            # Use the newer version that supports masks
            matched_image = ants.histogram_match_image2(
                source_image,
                reference_image,
                source_mask=source_mask,
                reference_mask=reference_mask,
                match_points=64,
            )
            if logger:
                logger.info("Applied histogram_match_image2 with masks")
        else:
            # Use the basic version without masks
            matched_image = ants.histogram_match_image(
                source_image,
                reference_image,
                number_of_histogram_bins=255,
                number_of_match_points=64,
            )
            if logger:
                logger.info("Applied basic histogram_match_image (without masks)")

        # Validate the result
        if matched_image is None or np.any(np.isnan(matched_image.numpy())):
            if logger:
                logger.warning("Histogram matching produced invalid results")
            return None

        return matched_image

    except Exception as e:
        if logger:
            logger.warning(f"Histogram matching failed: {e}")
        return None
