import gc
import logging
import random

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RandomCrop")


class RandomCrop:
    """
    Edge-aware patch extraction with robust content guarantees for prostate MRI data.

    This class extracts patches from 3D prostate MRI images with specific focus on:
    1. Avoiding edges of the prostate to prevent empty/sparse patches
    2. Ensuring content density is above minimum thresholds
    3. Using distance maps to prioritize central prostate regions
    4. Multiple fallback strategies if initial patch selection fails
    """

    FOREGROUND_THRESHOLD = -0.95
    EDGE_MARGIN_PERCENT = 0.05
    MIN_CONTENT_AREA_PERCENT = 15.0
    MIN_CENTRAL_DENSITY = 0.10
    MAX_RETRY_ATTEMPTS = 50

    def __init__(
        self,
        output_size,
        drop_ratio=0.1,
        min_pixel=0.9,
        patches_per_image=8,
        patch_size=(64, 64, 32),
        content_detection_mode="combined",
        strategy_weights=None,
        debug=True,
        edge_aware=True,
        density_check=True,
    ):
        """
        Initialize RandomCrop with configurable parameters.

        Args:
            output_size: Size of output patches (x, y, z) or single int
            drop_ratio: Probability to drop a patch with insufficient content
            min_pixel: Minimum required non-background pixels (absolute count or ratio)
            patches_per_image: Number of patches to extract from each image
            patch_size: Expected patch dimensions (x, y, z)
            content_detection_mode: Method to detect content ("exvivo", "invivo", "combined")
            strategy_weights: Dictionary of weights for different patch extraction strategies
            debug: Enable verbose debug logging
            edge_aware: Enable edge awareness to avoid object boundaries
            density_check: Enable content density checking
        """
        self.name = "RandomCrop"
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.debug = debug
        self.edge_aware = edge_aware
        self.density_check = density_check
        self.drop_ratio = drop_ratio
        self.logger = logger

        if debug:
            logger.setLevel(logging.DEBUG)

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            self.output_size = tuple(output_size)

        assert all(
            i > 0 for i in self.output_size
        ), "All output dimensions must be positive"

        self.total_voxels = (
            self.output_size[0] * self.output_size[1] * self.output_size[2]
        )

        if min_pixel >= 1.0:
            self.min_pixel = int(min_pixel)
        else:
            self.min_pixel = int(self.total_voxels * min_pixel)

        max_reasonable = int(self.total_voxels * 0.25)
        if self.min_pixel > max_reasonable:
            logger.warning(
                f"Very high min_pixel value {self.min_pixel}. Limiting to {max_reasonable}."
            )
            self.min_pixel = max_reasonable

        valid_modes = ["exvivo", "invivo", "combined"]
        if content_detection_mode not in valid_modes:
            logger.warning(
                f"Invalid content_detection_mode '{content_detection_mode}'. Using 'combined'."
            )
            self.content_detection_mode = "combined"
        else:
            self.content_detection_mode = content_detection_mode

        default_weights = {
            "center": 0.4,
            "random": 0.3,
            "edge": 0.2,
            "corner": 0.1,
        }
        self.strategy_weights = (
            strategy_weights if strategy_weights else default_weights
        )

        weight_sum = sum(self.strategy_weights.values())
        if weight_sum > 0:
            self.strategy_weights = {
                k: v / weight_sum for k, v in self.strategy_weights.items()
            }

        self._content_locations_cache = {}
        self._distance_maps_cache = {}

        self._edge_margins = [
            int(dim * self.EDGE_MARGIN_PERCENT) for dim in self.output_size
        ]

        if output_size[2] >= 16:
            self._edge_margins[2] = min(1, self._edge_margins[2])

        if self.debug:
            logger.debug(
                f"Initialized RandomCrop with output_size={self.output_size}, "
                f"min_pixel={self.min_pixel}, content_mode={self.content_detection_mode}"
            )

            if self.edge_aware:
                logger.debug(
                    f"Edge awareness enabled with margins {self._edge_margins}"
                )

    def __call__(self, sample):
        """
        Extract multiple patches with forced diversity.

        Args:
            sample: Dictionary containing image and label or invivo and exvivo SimpleITK images

        Returns:
            Dictionary with combined patches as 4D images
        """
        if "invivo" in sample and "exvivo" in sample:
            invivo, exvivo = sample["invivo"], sample["exvivo"]
            using_new_keys = True
        elif "image" in sample and "label" in sample:
            invivo, exvivo = sample["image"], sample["label"]
            using_new_keys = False
        else:
            raise KeyError(
                "Sample must contain either 'invivo'/'exvivo' or 'image'/'label' keys"
            )

        size_old = invivo.GetSize()
        size_new = self.output_size

        if self.debug:
            logger.debug(
                f"Source image size: {size_old}, target patch size: {size_new}"
            )

        if not self._has_content(invivo, exvivo):
            logger.warning("No content detected in the images - using center patches")
            result = self._handle_empty_image(invivo, exvivo)
        else:
            if self.edge_aware:
                self._ensure_distance_map(invivo, exvivo)

            is_mri_like = (
                size_new[0] / size_old[0] > 0.7
                and size_new[1] / size_old[1] > 0.7
                and size_new[2] / size_old[2] < 0.5
            )

            if is_mri_like and self.patches_per_image > 1:
                patches_result = self._extract_mri_patches(
                    invivo, exvivo, size_old, size_new
                )
            else:
                patches_result = self.extract_diverse_patches(
                    invivo, exvivo, size_old, size_new
                )

            result = self._validate_all_patches(patches_result, invivo, exvivo)

        if not using_new_keys:
            return {"image": result["invivo"], "label": result["exvivo"]}
        return result

    def _validate_all_patches(self, patches_result, original_invivo, original_exvivo):
        """
        Perform final validation on all patches to ensure they meet content requirements.
        Replaces low-content patches with better alternatives.

        Args:
            patches_result: Dictionary with invivo and exvivo patches
            original_invivo: Original SimpleITK invivo for fallback
            original_exvivo: Original SimpleITK exvivo for fallback

        Returns:
            Dictionary with validated patches
        """
        invivo_patches = []
        exvivo_patches = []

        if isinstance(patches_result["invivo"], list):
            existing_invivo_patches = patches_result["invivo"]
            existing_exvivo_patches = patches_result["exvivo"]
        else:
            existing_invivo_patches = self._split_combined_patches(
                patches_result["invivo"]
            )
            existing_exvivo_patches = self._split_combined_patches(
                patches_result["exvivo"]
            )

        min_content_percent = 20.0

        replacement_count = 0
        for i, (invivo_patch, exvivo_patch) in enumerate(
            zip(existing_invivo_patches, existing_exvivo_patches)
        ):
            content_info = self._check_patch_content(invivo_patch, exvivo_patch)

            is_valid = (
                content_info["invivo_percentage"] >= min_content_percent
                and content_info["exvivo_percentage"] >= min_content_percent
            )

            if is_valid:
                invivo_patches.append(invivo_patch)
                exvivo_patches.append(exvivo_patch)
            else:
                replacement_count += 1
                logger.warning(
                    f"Replacing low-content patch {i}: invivo={content_info['invivo_percentage']:.2f}%, "
                    f"exvivo={content_info['exvivo_percentage']:.2f}%"
                )

                replacement = None
                found_good_patch = False

                for retry in range(5):
                    size_old = original_invivo.GetSize()
                    size_new = self.output_size

                    center_x = max(0, (size_old[0] - size_new[0]) // 2)
                    center_y = max(0, (size_old[1] - size_new[1]) // 2)
                    center_z = max(0, (size_old[2] - size_new[2]) // 2)

                    offset_x = random.randint(-size_new[0] // 10, size_new[0] // 10)
                    offset_y = random.randint(-size_new[1] // 10, size_new[1] // 10)
                    offset_z = random.randint(-size_new[2] // 10, size_new[2] // 10)

                    x_start = max(
                        0, min(size_old[0] - size_new[0], center_x + offset_x)
                    )
                    y_start = max(
                        0, min(size_old[1] - size_new[1], center_y + offset_y)
                    )
                    z_start = max(
                        0, min(size_old[2] - size_new[2], center_z + offset_z)
                    )

                    replacement = self._extract_and_validate_patch(
                        original_invivo,
                        original_exvivo,
                        (x_start, y_start, z_start),
                        size_new,
                    )

                    replacement_content = self._check_patch_content(
                        replacement["invivo"], replacement["exvivo"]
                    )

                    if (
                        replacement_content["invivo_percentage"] >= min_content_percent
                        and replacement_content["exvivo_percentage"]
                        >= min_content_percent
                    ):
                        found_good_patch = True
                        break

                    logger.debug(
                        f"Retry {retry + 1}/5 - Found patch with invivo={replacement_content['invivo_percentage']:.2f}%, exvivo={replacement_content['exvivo_percentage']:.2f}%"
                    )

                if not found_good_patch:
                    logger.warning(
                        f"Could not find patch meeting 20% content threshold after 5 attempts. Using center patch."
                    )
                    replacement = {
                        "invivo": self._extract_center_patch(
                            original_invivo, original_exvivo
                        )[0],
                        "exvivo": self._extract_center_patch(
                            original_invivo, original_exvivo
                        )[1],
                    }

                invivo_patches.append(replacement["invivo"])
                exvivo_patches.append(replacement["exvivo"])

        if replacement_count > 0:
            logger.info(
                f"Replaced {replacement_count} patches with insufficient content"
            )

        try:
            combined_invivo = self._combine_patches(invivo_patches)
            combined_exvivo = self._combine_patches(exvivo_patches)

            del invivo_patches, exvivo_patches
            gc.collect()

            if (
                sitk.GetArrayFromImage(combined_invivo).shape[0]
                != self.patches_per_image
            ):
                logger.error(
                    f"Expected {self.patches_per_image} patches but got {sitk.GetArrayFromImage(combined_invivo).shape[0]}"
                )

            return {"invivo": combined_invivo, "exvivo": combined_exvivo}
        except Exception as e:
            logger.error(f"Error combining validated patches: {e}")
            return {"invivo": invivo_patches[0], "exvivo": exvivo_patches[0]}

    def _split_combined_patches(self, combined_image):
        """
        Split a 4D combined image back into a list of 3D patches.

        Args:
            combined_image: SimpleITK 4D image

        Returns:
            List of SimpleITK 3D images
        """
        combined_array = sitk.GetArrayFromImage(combined_image)

        patches = []
        for i in range(combined_array.shape[0]):
            patch_array = combined_array[i]
            patch = sitk.GetImageFromArray(patch_array)
            patch.CopyInformation(combined_image)
            patches.append(patch)

        return patches

    def _has_content(self, invivo, exvivo):
        """
        Check if invivo or exvivo has any content (non-background).

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image

        Returns:
            bool: True if either invivo or exvivo has content
        """
        if self.content_detection_mode in ["exvivo", "combined"]:
            exvivo_array = sitk.GetArrayFromImage(exvivo)
            if np.sum(exvivo_array > 0) > 0:
                return True

        if self.content_detection_mode in ["invivo", "combined"]:
            invivo_size = invivo.GetSize()
            roi_filter = sitk.RegionOfInterestImageFilter()

            regions_to_check = min(3, max(1, int(invivo_size[2] / 10)))
            step_size = max(1, int(invivo_size[2] / regions_to_check))

            for i in range(regions_to_check):
                z_pos = min(i * step_size, invivo_size[2] - 2)

                sample_size = (
                    min(128, invivo_size[0]),
                    min(128, invivo_size[1]),
                    min(2, invivo_size[2] - z_pos),
                )

                roi_filter.SetSize(sample_size)
                roi_filter.SetIndex([0, 0, z_pos])

                try:
                    sample_region = roi_filter.Execute(invivo)
                    sample_array = sitk.GetArrayFromImage(sample_region)

                    if np.sum(sample_array > self.FOREGROUND_THRESHOLD) > 10:
                        return True
                except Exception as e:
                    logger.warning(f"Error sampling invivo for content check: {e}")

        return False

    def _handle_empty_image(self, invivo, exvivo):
        """
        Handle case when image has no detectable content.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image

        Returns:
            Dictionary with invivo/exvivo patches
        """
        invivo_patches = []
        exvivo_patches = []

        for _ in range(self.patches_per_image):
            invivo_crop, exvivo_crop = self._extract_center_patch(invivo, exvivo)
            invivo_patches.append(invivo_crop)
            exvivo_patches.append(exvivo_crop)

        try:
            combined_invivo = self._combine_patches(invivo_patches)
            combined_exvivo = self._combine_patches(exvivo_patches)
            return {"invivo": combined_invivo, "exvivo": combined_exvivo}
        except Exception as e:
            logger.error(f"Error combining patches: {e}")
            return {"invivo": invivo_patches[0], "exvivo": exvivo_patches[0]}

    def _extract_mri_patches(self, invivo, exvivo, size_old, size_new):
        """
        Extract patches from MRI-like data, with focus on different z positions and diversity.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            size_old: Original image size
            size_new: Target patch size

        Returns:
            Dictionary with combined invivo/exvivo patches
        """
        invivo_patches = []
        exvivo_patches = []
        patch_fingerprints = []

        z_positions = self._calculate_z_positions(size_old, size_new)

        for z_idx, z_start in enumerate(z_positions):
            max_retries = 5
            found_good_patch = False

            for retry in range(max_retries):
                use_center = (z_idx % 2 == 0) if retry == 0 else (retry % 2 == 0)

                if use_center:
                    x_start = max(0, (size_old[0] - size_new[0]) // 2)
                    y_start = max(0, (size_old[1] - size_new[1]) // 2)
                else:
                    if self.edge_aware:
                        x_start, y_start = self._get_content_biased_position(
                            invivo, exvivo, size_old, size_new, z_start
                        )
                    else:
                        x_start = random.randint(0, max(0, size_old[0] - size_new[0]))
                        y_start = random.randint(0, max(0, size_old[1] - size_new[1]))

                patch_result = self._extract_and_validate_patch(
                    invivo, exvivo, (x_start, y_start, z_start), size_new
                )

                content_info = self._check_patch_content(
                    patch_result["invivo"], patch_result["exvivo"]
                )

                min_content_percent = 20.0

                has_content = (
                    content_info["invivo_percentage"] >= min_content_percent
                    and content_info["exvivo_percentage"] >= min_content_percent
                )

                if has_content:
                    fingerprint = self._create_patch_fingerprint(patch_result["invivo"])

                    too_similar = False
                    for existing_fp in patch_fingerprints:
                        similarity = self._calculate_fingerprint_similarity(
                            fingerprint, existing_fp
                        )
                        if similarity > 0.7:
                            if self.debug:
                                logger.debug(
                                    f"Z{z_idx} patch too similar (similarity: {similarity:.2f}). Trying another position."
                                )
                            too_similar = True
                            break

                    if not too_similar:
                        found_good_patch = True
                        patch_fingerprints.append(fingerprint)
                        break

                if self.debug and retry < max_retries - 1:
                    logger.debug(
                        f"Retry {retry + 1}/{max_retries} for z-position {z_idx} due to"
                        f"{' insufficient content' if not has_content else ' similarity'}"
                    )

            invivo_patches.append(patch_result["invivo"])
            exvivo_patches.append(patch_result["exvivo"])

            if not found_good_patch:
                logger.warning(
                    f"Could not find diverse patch with sufficient content at z-position {z_idx}"
                )

        try:
            combined_invivo = self._combine_patches(invivo_patches)
            combined_exvivo = self._combine_patches(exvivo_patches)

            del invivo_patches, exvivo_patches, patch_fingerprints
            gc.collect()

            return {"invivo": combined_invivo, "exvivo": combined_exvivo}
        except Exception as e:
            logger.error(f"Error combining MRI patches: {e}")
            return {"invivo": invivo_patches[0], "exvivo": exvivo_patches[0]}

    def extract_diverse_patches(
        self, invivo, exvivo, size_old, size_new, similarity_threshold=0.7
    ):
        """
        Extract diverse patches using multiple strategies (center, random, edge, corner)
        with explicit diversity checking.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            size_old: Original image size
            size_new: Target patch size
            similarity_threshold: Maximum allowed similarity between patches

        Returns:
            Dictionary with combined invivo/exvivo patches
        """
        invivo_patches = []
        exvivo_patches = []
        patch_fingerprints = []

        patch_idx = 0
        attempts = 0
        max_attempts = self.patches_per_image * 5

        while len(invivo_patches) < self.patches_per_image and attempts < max_attempts:
            if patch_idx == 0:
                strategy = "center"
            else:
                strategy = self._choose_patch_strategy()

            offset = attempts * 100
            start_pos = self._get_patch_position(
                strategy, size_old, size_new, patch_idx + offset
            )
            x_start, y_start, z_start = start_pos

            patch_result = self._extract_and_validate_patch(
                invivo, exvivo, (x_start, y_start, z_start), size_new
            )

            if (
                not patch_result
                or "invivo" not in patch_result
                or "exvivo" not in patch_result
            ):
                attempts += 1
                continue

            fingerprint = self._create_patch_fingerprint(patch_result["invivo"])

            too_similar = False
            for existing_fp in patch_fingerprints:
                similarity = self._calculate_fingerprint_similarity(
                    fingerprint, existing_fp
                )
                if similarity > similarity_threshold:
                    if self.debug:
                        logger.debug(
                            f"Patch too similar (similarity: {similarity:.2f}). Trying another position."
                        )
                    too_similar = True
                    break

            if too_similar:
                attempts += 1
                continue

            invivo_patches.append(patch_result["invivo"])
            exvivo_patches.append(patch_result["exvivo"])
            patch_fingerprints.append(fingerprint)
            patch_idx += 1

            if self.debug:
                logger.debug(
                    f"Added diverse patch {patch_idx}/{self.patches_per_image} "
                    f"using strategy '{strategy}' (attempts: {attempts})"
                )

        if len(invivo_patches) < self.patches_per_image:
            logger.warning(
                f"Could only find {len(invivo_patches)} diverse patches "
                f"after {attempts} attempts. Filling with center patches."
            )

            while len(invivo_patches) < self.patches_per_image:
                center_invivo, center_exvivo = self._extract_center_patch(
                    invivo, exvivo
                )
                invivo_patches.append(center_invivo)
                exvivo_patches.append(center_exvivo)

        try:
            combined_invivo = self._combine_patches(invivo_patches)
            combined_exvivo = self._combine_patches(exvivo_patches)

            del invivo_patches, exvivo_patches, patch_fingerprints
            gc.collect()

            return {"invivo": combined_invivo, "exvivo": combined_exvivo}
        except Exception as e:
            logger.error(f"Error combining diverse patches: {e}")
            return {"invivo": invivo_patches[0], "exvivo": exvivo_patches[0]}

    def _ensure_distance_map(self, invivo, exvivo):
        """
        Compute distance map from object boundaries if not already cached.
        This is used for content-biased positioning.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
        """
        cache_key = id(invivo), id(exvivo)

        if cache_key in self._distance_maps_cache:
            return

        if self.content_detection_mode == "exvivo":
            base_image = exvivo
            threshold = 0
        else:
            base_image = invivo
            threshold = self.FOREGROUND_THRESHOLD

        image_size = base_image.GetSize()

        downsample_factor = 1
        if max(image_size) > 256:
            downsample_factor = max(image_size) / 256

        if downsample_factor > 1:
            new_size = [int(s / downsample_factor) for s in image_size]

            if min(new_size) < 16:
                self._distance_maps_cache[cache_key] = None
                return

            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(new_size)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputDirection(base_image.GetDirection())
            resampler.SetOutputOrigin(base_image.GetOrigin())

            new_spacing = [
                base_image.GetSpacing()[0] * (image_size[0] / new_size[0]),
                base_image.GetSpacing()[1] * (image_size[1] / new_size[1]),
                base_image.GetSpacing()[2] * (image_size[2] / new_size[2]),
            ]
            resampler.SetOutputSpacing(new_spacing)

            try:
                downsampled = resampler.Execute(base_image)
                array = sitk.GetArrayFromImage(downsampled)
            except Exception as e:
                logger.warning(f"Error downsampling for distance map: {e}")
                self._distance_maps_cache[cache_key] = None
                return
        else:
            array = sitk.GetArrayFromImage(base_image)

        try:
            mask = array > threshold

            if not np.any(mask):
                self._distance_maps_cache[cache_key] = None
                return

            dt = distance_transform_edt(mask)

            dt_max = dt.max()
            if dt_max > 0:
                dt = dt / dt_max
            else:
                dt = np.zeros_like(dt)

            self._distance_maps_cache[cache_key] = (dt, downsample_factor)

            if self.debug:
                logger.debug(
                    f"Created distance map with range [0, {dt_max}], factor {downsample_factor}"
                )

        except Exception as e:
            logger.warning(f"Error computing distance transform: {e}")
            self._distance_maps_cache[cache_key] = None

    def _get_content_biased_position(
        self, invivo, exvivo, size_old, size_new, z_start=None
    ):
        """
        Get x,y position biased towards content centers using distance map.
        Fixed to handle cases with small dimensions.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            size_old: Original image size
            size_new: Target patch size
            z_start: Fixed z position (if None, will also select z)

        Returns:
            Tuple of (x_start, y_start) or (x_start, y_start, z_start)
        """

        x_space = max(0, size_old[0] - size_new[0])
        y_space = max(0, size_old[1] - size_new[1])
        z_space = max(0, size_old[2] - size_new[2])

        x_margin = min(self._edge_margins[0], max(0, x_space // 2))
        y_margin = min(self._edge_margins[1], max(0, y_space // 2))
        z_margin = min(self._edge_margins[2], max(0, z_space // 2))

        if x_space < 5 or y_space < 5 or (z_start is None and z_space < 5):
            x_start = x_space // 2
            y_start = y_space // 2
            if z_start is None:
                z_start = z_space // 2
                return x_start, y_start, z_start
            return x_start, y_start

        if z_start is None:
            try:
                x_start = random.randint(x_margin, max(x_margin, x_space - x_margin))
                y_start = random.randint(y_margin, max(y_margin, y_space - y_margin))
                z_start = random.randint(z_margin, max(z_margin, z_space - z_margin))
            except ValueError:
                x_start = x_space // 2
                y_start = y_space // 2
                z_start = z_space // 2
                self.logger.warning(
                    "Falling back to center position due to range error"
                )
            return x_start, y_start, z_start
        else:
            try:
                x_start = random.randint(x_margin, max(x_margin, x_space - x_margin))
                y_start = random.randint(y_margin, max(y_margin, y_space - y_margin))
            except ValueError:
                x_start = x_space // 2
                y_start = y_space // 2
                self.logger.warning(
                    "Falling back to center x,y position due to range error"
                )
            return x_start, y_start

        distance_map, downsample_factor = distance_map_data

        if z_start is not None and distance_map.ndim == 3:
            z_idx = min(int(z_start / downsample_factor), distance_map.shape[0] - 1)

            distance_slice = distance_map[z_idx]

            if not np.any(distance_slice > 0):
                for offset in [1, -1, 2, -2, 3, -3]:
                    test_z = z_idx + offset
                    if 0 <= test_z < distance_map.shape[0]:
                        if np.any(distance_map[test_z] > 0):
                            distance_slice = distance_map[test_z]
                            break
        else:
            distance_slice = distance_map

        if np.any(distance_slice > 0.5):
            high_value_coords = np.where(distance_slice > 0.5)

            if len(high_value_coords[0]) > 0:
                idx = random.randint(0, len(high_value_coords[0]) - 1)

                if distance_slice.ndim == 2:
                    y, x = high_value_coords[0][idx], high_value_coords[1][idx]

                    y_orig = int(y * downsample_factor)
                    x_orig = int(x * downsample_factor)

                    x_start = max(
                        0, min(size_old[0] - size_new[0], x_orig - size_new[0] // 2)
                    )
                    y_start = max(
                        0, min(size_old[1] - size_new[1], y_orig - size_new[1] // 2)
                    )

                    return x_start, y_start

                else:
                    z, y, x = (
                        high_value_coords[0][idx],
                        high_value_coords[1][idx],
                        high_value_coords[2][idx],
                    )

                    z_orig = int(z * downsample_factor)
                    y_orig = int(y * downsample_factor)
                    x_orig = int(x * downsample_factor)

                    x_start = max(
                        0, min(size_old[0] - size_new[0], x_orig - size_new[0] // 2)
                    )
                    y_start = max(
                        0, min(size_old[1] - size_new[1], y_orig - size_new[1] // 2)
                    )
                    z_start = max(
                        0, min(size_old[2] - size_new[2], z_orig - size_new[2] // 2)
                    )

                    return x_start, y_start, z_start

        if z_start is None:
            x_start = random.randint(
                self._edge_margins[0],
                max(0, size_old[0] - size_new[0] - self._edge_margins[0]),
            )
            y_start = random.randint(
                self._edge_margins[1],
                max(0, size_old[1] - size_new[1] - self._edge_margins[1]),
            )
            z_start = random.randint(
                self._edge_margins[2],
                max(0, size_old[2] - size_new[2] - self._edge_margins[2]),
            )
            return x_start, y_start, z_start
        else:
            x_start = random.randint(
                self._edge_margins[0],
                max(0, size_old[0] - size_new[0] - self._edge_margins[0]),
            )
            y_start = random.randint(
                self._edge_margins[1],
                max(0, size_old[1] - size_new[1] - self._edge_margins[1]),
            )
            return x_start, y_start

    def _get_content_biased_position_3d(self, invivo, exvivo, size_old, size_new):
        """
        Get 3D position biased towards content centers using distance map.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            size_old: Original image size
            size_new: Target patch size

        Returns:
            Tuple of (x_start, y_start, z_start)
        """
        return self._get_content_biased_position(invivo, exvivo, size_old, size_new)

    def _calculate_z_positions(self, size_old, size_new):
        """
        Calculate optimal z positions to extract patches from.

        Args:
            size_old: Original image size
            size_new: Target patch size

        Returns:
            List of starting z positions for patches
        """
        z_range = max(0, size_old[2] - size_new[2])

        if z_range == 0:
            return [0] * self.patches_per_image

        step_size = max(1, z_range // self.patches_per_image)

        positions = []

        for i in range(self.patches_per_image):
            if i == 0:
                z_pos = 0
            elif i == self.patches_per_image - 1:
                z_pos = z_range
            else:
                base_pos = i * step_size
                offset_range = max(1, step_size // 4)
                random_offset = random.randint(-offset_range, offset_range)
                z_pos = max(0, min(z_range, base_pos + random_offset))

            positions.append(z_pos)

        return positions

    def _choose_patch_strategy(self):
        """
        Choose patch extraction strategy based on weights.

        Returns:
            String: Strategy name ("center", "random", "edge", "corner")
        """
        strategies = list(self.strategy_weights.keys())
        weights = [self.strategy_weights[s] for s in strategies]

        strategy = random.choices(strategies, weights=weights, k=1)[0]

        return strategy

    def _get_patch_position(self, strategy, size_old, size_new, patch_index):
        """
        Get starting position for patch based on strategy.
        Modified to handle cases where image is too small for the requested margins.

        Args:
            strategy: Strategy name
            size_old: Original image size
            size_new: Target patch size
            patch_index: Index of current patch (for determinism)

        Returns:
            Tuple: (x_start, y_start, z_start)
        """
        x_space = max(0, size_old[0] - size_new[0])
        y_space = max(0, size_old[1] - size_new[1])
        z_space = max(0, size_old[2] - size_new[2])

        x_margin = min(self._edge_margins[0], x_space // 2)
        y_margin = min(self._edge_margins[1], y_space // 2)
        z_margin = min(self._edge_margins[2], z_space // 2)

        if z_space < 4:
            z_margin = 0
        if y_space < 10:
            y_margin = 0
        if x_space < 10:
            x_margin = 0

        x_min = min(x_margin, x_space)
        x_max = max(x_min, x_space - x_margin)
        y_min = min(y_margin, y_space)
        y_max = max(y_min, y_space - y_margin)
        z_min = min(z_margin, z_space)
        z_max = max(z_min, z_space - z_margin)

        if strategy == "center":
            x_start = x_space // 2
            y_start = y_space // 2
            z_start = z_space // 2

        elif strategy == "random":
            if x_min >= x_max:
                x_start = x_min
            else:
                x_start = random.randint(x_min, x_max)

            if y_min >= y_max:
                y_start = y_min
            else:
                y_start = random.randint(y_min, y_max)

            if z_min >= z_max:
                z_start = z_min
            else:
                try:
                    z_start = random.randint(z_min, z_max)
                except ValueError:
                    self.logger.warning(
                        f"Invalid z range ({z_min}, {z_max}). Using center."
                    )
                    z_start = z_space // 2

        elif strategy == "edge":
            edge_type = patch_index % 6
            if edge_type == 0:
                x_start = x_min
                y_start = max(0, (size_old[1] - size_new[1]) // 2)
                z_start = max(0, (size_old[2] - size_new[2]) // 2)
            elif edge_type == 1:
                x_start = x_max
                y_start = max(0, (size_old[1] - size_new[1]) // 2)
                z_start = max(0, (size_old[2] - size_new[2]) // 2)
            elif edge_type == 2:
                x_start = max(0, (size_old[0] - size_new[0]) // 2)
                y_start = y_min
                z_start = max(0, (size_old[2] - size_new[2]) // 2)
            elif edge_type == 3:
                x_start = max(0, (size_old[0] - size_new[0]) // 2)
                y_start = y_max
                z_start = max(0, (size_old[2] - size_new[2]) // 2)
            elif edge_type == 4:
                x_start = max(0, (size_old[0] - size_new[0]) // 2)
                y_start = max(0, (size_old[1] - size_new[1]) // 2)
                z_start = z_min
            else:
                x_start = max(0, (size_old[0] - size_new[0]) // 2)
                y_start = max(0, (size_old[1] - size_new[1]) // 2)
                z_start = z_max

        elif strategy == "corner":
            corner = patch_index % 8
            x_corner = (corner & 1) > 0
            y_corner = (corner & 2) > 0
            z_corner = (corner & 4) > 0

            x_start = x_min if not x_corner else x_max
            y_start = y_min if not y_corner else y_max
            z_start = z_min if not z_corner else z_max

        else:
            print(f"Warning: Unknown strategy '{strategy}'. Using center.")
            x_start = max(0, (size_old[0] - size_new[0]) // 2)
            y_start = max(0, (size_old[1] - size_new[1]) // 2)
            z_start = max(0, (size_old[2] - size_new[2]) // 2)

        return (x_start, y_start, z_start)

    def _extract_and_validate_patch(self, invivo, exvivo, start_pos, size):
        """
        Extract a patch and ensure it has sufficient content.
        Strictly enforces the 20% content threshold for visualization.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            start_pos: Starting position for patch
            size: Patch size

        Returns:
            Dictionary with validated invivo and exvivo patches
        """
        invivo_crop, exvivo_crop = self._extract_specific_patch(
            invivo, exvivo, start_pos, size
        )

        content_info = self._check_patch_content(invivo_crop, exvivo_crop)

        min_content_percent = 20.0

        is_valid = (
            content_info["invivo_percentage"] >= min_content_percent
            and content_info["exvivo_percentage"] >= min_content_percent
        )

        if not is_valid:
            for retry in range(3):
                size_old = invivo.GetSize()
                center_x = max(0, (size_old[0] - size[0]) // 2)
                center_y = max(0, (size_old[1] - size[1]) // 2)
                center_z = max(0, (size_old[2] - size[2]) // 2)

                offset_x = random.randint(-size[0] // 5, size[0] // 5)
                offset_y = random.randint(-size[1] // 5, size[1] // 5)
                offset_z = random.randint(-size[2] // 5, size[2] // 5)

                new_x = max(0, min(size_old[0] - size[0], center_x + offset_x))
                new_y = max(0, min(size_old[1] - size[1], center_y + offset_y))
                new_z = max(0, min(size_old[2] - size[2], center_z + offset_z))

                new_invivo, new_exvivo = self._extract_specific_patch(
                    invivo, exvivo, (new_x, new_y, new_z), size
                )

                new_content = self._check_patch_content(new_invivo, new_exvivo)

                if (
                    new_content["invivo_percentage"] >= min_content_percent
                    and new_content["exvivo_percentage"] >= min_content_percent
                ):
                    invivo_crop = new_invivo
                    exvivo_crop = new_exvivo
                    content_info = new_content
                    is_valid = True
                    break

        if not is_valid:
            size_old = invivo.GetSize()
            center_x = max(0, (size_old[0] - size[0]) // 2)
            center_y = max(0, (size_old[1] - size[1]) // 2)
            center_z = max(0, (size_old[2] - size[2]) // 2)

            center_invivo, center_exvivo = self._extract_specific_patch(
                invivo, exvivo, (center_x, center_y, center_z), size
            )

            invivo_crop = center_invivo
            exvivo_crop = center_exvivo

        return {"invivo": invivo_crop, "exvivo": exvivo_crop}

    def _extract_specific_patch(self, invivo, exvivo, start_pos, size):
        """
        Extract a patch at a specific position with boundary safety checks.
        SHIFTS the window instead of resizing when a patch doesn't fit.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            start_pos: Starting position (x, y, z)
            size: Patch size (x, y, z)

        Returns:
            Tuple of extracted patches (invivo_crop, exvivo_crop)
        """
        try:
            invivo_size = invivo.GetSize()

            adjusted_start_pos = list(start_pos)

            adjusted_size = list(size)

            for i in range(3):
                if adjusted_start_pos[i] < 0:
                    adjusted_start_pos[i] = 0

                end_pos = adjusted_start_pos[i] + adjusted_size[i]
                if end_pos > invivo_size[i]:
                    adjusted_start_pos[i] = max(0, invivo_size[i] - adjusted_size[i])

                    if adjusted_start_pos[i] + adjusted_size[i] > invivo_size[i]:
                        logger.warning(
                            f"Cannot fit patch of size {adjusted_size[i]} in dimension {i} (image size: {invivo_size[i]}). Falling back to center patch."
                        )
                        return self._extract_center_patch(invivo, exvivo)

            roi_filter = sitk.RegionOfInterestImageFilter()
            roi_filter.SetSize(adjusted_size)
            roi_filter.SetIndex(
                [
                    int(adjusted_start_pos[0]),
                    int(adjusted_start_pos[1]),
                    int(adjusted_start_pos[2]),
                ]
            )

            invivo_crop = roi_filter.Execute(invivo)
            exvivo_crop = roi_filter.Execute(exvivo)

            invivo_array = sitk.GetArrayFromImage(invivo_crop)
            exvivo_array = sitk.GetArrayFromImage(exvivo_crop)

            invivo_crop_new = sitk.GetImageFromArray(invivo_array.copy())
            exvivo_crop_new = sitk.GetImageFromArray(exvivo_array.copy())

            invivo_crop_new.CopyInformation(invivo_crop)
            exvivo_crop_new.CopyInformation(exvivo_crop)

            invivo_array = None
            exvivo_array = None

            return invivo_crop_new, exvivo_crop_new
        except Exception as e:
            logger.error(f"Error extracting patch at {start_pos}: {e}")
            return self._extract_center_patch(invivo, exvivo)

    def _extract_center_patch(self, invivo, exvivo):
        """
        Extract patch from center of image with guaranteed consistent size.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image

        Returns:
            Tuple of (invivo_crop, exvivo_crop)
        """
        invivo_size = invivo.GetSize()

        size_new = self.output_size

        for i in range(3):
            if size_new[i] > invivo_size[i]:
                logger.warning(
                    f"Cannot extract center patch: requested size {size_new[i]} > image size {invivo_size[i]} in dimension {i}"
                )

                resampler = sitk.ResampleImageFilter()

                resampler.SetSize(size_new)

                new_spacing = [
                    invivo.GetSpacing()[0] * (invivo_size[0] / size_new[0]),
                    invivo.GetSpacing()[1] * (invivo_size[1] / size_new[1]),
                    invivo.GetSpacing()[2] * (invivo_size[2] / size_new[2]),
                ]
                resampler.SetOutputSpacing(new_spacing)

                resampler.SetOutputDirection(invivo.GetDirection())
                resampler.SetOutputOrigin(invivo.GetOrigin())
                resampler.SetDefaultPixelValue(-1)

                resampler.SetInterpolator(sitk.sitkLinear)

                invivo_resampled = resampler.Execute(invivo)
                exvivo_resampled = resampler.Execute(exvivo)

                return invivo_resampled, exvivo_resampled

        center_x = max(0, (invivo_size[0] - size_new[0]) // 2)
        center_y = max(0, (invivo_size[1] - size_new[1]) // 2)
        center_z = max(0, (invivo_size[2] - size_new[2]) // 2)

        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetSize(size_new)
        roi_filter.SetIndex([center_x, center_y, center_z])

        invivo_crop = roi_filter.Execute(invivo)
        exvivo_crop = roi_filter.Execute(exvivo)

        invivo_array = sitk.GetArrayFromImage(invivo_crop)
        exvivo_array = sitk.GetArrayFromImage(exvivo_crop)

        invivo_crop_new = sitk.GetImageFromArray(invivo_array.copy())
        exvivo_crop_new = sitk.GetImageFromArray(exvivo_array.copy())

        invivo_crop_new.CopyInformation(invivo_crop)
        exvivo_crop_new.CopyInformation(exvivo_crop)

        invivo_array = None
        exvivo_array = None

        return invivo_crop_new, exvivo_crop_new

    def _check_patch_content(self, invivo, exvivo):
        """
        Check if a patch contains sufficient and well-distributed content.

        Args:
            invivo: SimpleITK invivo patch
            exvivo: SimpleITK exvivo patch

        Returns:
            Dictionary with content metrics
        """
        invivo_array = sitk.GetArrayFromImage(invivo)
        exvivo_array = sitk.GetArrayFromImage(exvivo)

        exvivo_content = np.sum(exvivo_array > 0)
        invivo_content = np.sum(invivo_array > self.FOREGROUND_THRESHOLD)

        total_pixels = np.prod(invivo_array.shape)
        exvivo_percentage = (
            (exvivo_content / total_pixels) * 100 if total_pixels > 0 else 0
        )
        invivo_percentage = (
            (invivo_content / total_pixels) * 100 if total_pixels > 0 else 0
        )

        central_density = 0
        has_good_density = False

        if self.density_check:
            z_size, y_size, x_size = invivo_array.shape
            z_center_start = z_size // 4
            z_center_end = z_size - z_size // 4
            y_center_start = y_size // 4
            y_center_end = y_size - y_size // 4
            x_center_start = x_size // 4
            x_center_end = x_size - x_size // 4

            central_region = invivo_array[
                z_center_start:z_center_end,
                y_center_start:y_center_end,
                x_center_start:x_center_end,
            ]

            central_content_invivo = np.sum(central_region > self.FOREGROUND_THRESHOLD)
            central_pixels = np.prod(central_region.shape)

            if central_pixels > 0:
                central_density = central_content_invivo / central_pixels

            has_good_density = central_density >= self.MIN_CENTRAL_DENSITY

            if self.content_detection_mode in ["exvivo", "combined"]:
                central_region_exvivo = exvivo_array[
                    z_center_start:z_center_end,
                    y_center_start:y_center_end,
                    x_center_start:x_center_end,
                ]

                central_content_exvivo = np.sum(central_region_exvivo > 0)
                if central_pixels > 0:
                    central_density_exvivo = central_content_exvivo / central_pixels
                    central_density = max(central_density, central_density_exvivo)
                    has_good_density = central_density >= self.MIN_CENTRAL_DENSITY

        if self.content_detection_mode == "exvivo":
            total_content = exvivo_content
        elif self.content_detection_mode == "invivo":
            total_content = invivo_content
        else:
            total_content = max(exvivo_content, invivo_content)

        min_area = self.total_voxels * self.MIN_CONTENT_AREA_PERCENT / 100

        has_sufficient_content = total_content >= min_area

        return {
            "exvivo_content": exvivo_content,
            "invivo_content": invivo_content,
            "total_content": total_content,
            "exvivo_percentage": exvivo_percentage,
            "invivo_percentage": invivo_percentage,
            "has_sufficient_content": has_sufficient_content,
            "central_density": central_density,
            "has_good_density": has_good_density,
        }

    def _combine_patches(self, patches):
        """
        Combine multiple patches into a 4D image with clean memory layout.

        Args:
            patches: List of SimpleITK 3D images

        Returns:
            SimpleITK 4D image with proper memory layout
        """
        arrays = []
        for patch in patches:
            patch_array = sitk.GetArrayFromImage(patch).copy()
            arrays.append(patch_array)

            patch_array = None

        combined = np.stack(arrays, axis=0)

        combined_image = sitk.GetImageFromArray(combined)

        combined_image.SetSpacing(patches[0].GetSpacing())
        combined_image.SetDirection(patches[0].GetDirection())
        combined_image.SetOrigin(patches[0].GetOrigin())

        arrays = None
        combined = None

        return combined_image

    def extract_center_biased_patch(
        self,
        sample,
        patch_index,
        previous_patches=None,
        similarity_threshold=0.7,
        max_attempts=10,
    ):
        """
        Extract a single patch using optimized content-aware strategy with diversity consideration.
        Fixed to prevent NoneType errors when handling previous patches and empty range errors.

        Args:
            sample: Dictionary containing invivo/exvivo or image/label SimpleITK images
            patch_index: Index to use for deterministic extraction
            previous_patches: List of previously extracted patches to check diversity against
            similarity_threshold: Maximum allowed similarity to previous patches (0-1)
            max_attempts: Maximum number of attempts to find a diverse patch

        Returns:
            Dictionary with invivo and exvivo patches
        """
        if "invivo" in sample and "exvivo" in sample:
            invivo, exvivo = sample["invivo"], sample["exvivo"]
            using_new_keys = True
        elif "image" in sample and "label" in sample:
            invivo, exvivo = sample["image"], sample["label"]
            using_new_keys = False
        else:
            raise KeyError(
                "Sample must contain either 'invivo'/'exvivo' or 'image'/'label' keys"
            )

        size_old = invivo.GetSize()
        size_new = self.output_size

        for dim in range(3):
            if size_new[dim] > size_old[dim]:
                print(
                    f"Warning: Patch size {size_new[dim]} is larger than image size {size_old[dim]} in dimension {dim}. "
                    f"This may cause issues with patch extraction."
                )

        should_check_diversity = False
        if (
            previous_patches is not None
            and isinstance(previous_patches, list)
            and len(previous_patches) > 0
        ):
            valid_patches = []
            for prev_patch in previous_patches:
                if not isinstance(prev_patch, dict):
                    continue

                image_key = None
                if using_new_keys and "invivo" in prev_patch:
                    image_key = "invivo"
                elif not using_new_keys and "image" in prev_patch:
                    image_key = "image"

                if image_key and prev_patch[image_key] is not None:
                    try:
                        _ = prev_patch[image_key].GetPixelIDValue()
                        valid_patches.append(prev_patch)
                    except (AttributeError, Exception):
                        continue

            if valid_patches:
                previous_patches = valid_patches
                should_check_diversity = True

        if not should_check_diversity or similarity_threshold >= 1.0:
            if patch_index == 0:
                strategy = "center"
            elif patch_index == 1:
                if self.edge_aware:
                    self._ensure_distance_map(invivo, exvivo)
                    try:
                        x_start, y_start, z_start = (
                            self._get_content_biased_position_3d(
                                invivo, exvivo, size_old, size_new
                            )
                        )
                        result = self._extract_and_validate_patch(
                            invivo, exvivo, (x_start, y_start, z_start), size_new
                        )

                        if not using_new_keys:
                            return {
                                "image": result["invivo"],
                                "label": result["exvivo"],
                            }
                        return result
                    except Exception as e:
                        print(
                            f"Content-biased position failed: {e}. Falling back to center patch."
                        )
                        strategy = "center"
                else:
                    strategy = "random"
            else:
                strategy = self._choose_patch_strategy()

            try:
                if strategy == "random" and self.edge_aware:
                    x_start, y_start, z_start = self._get_content_biased_position_3d(
                        invivo, exvivo, size_old, size_new
                    )
                else:
                    start_pos = self._get_patch_position(
                        strategy, size_old, size_new, patch_index
                    )
                    x_start, y_start, z_start = start_pos

                result = self._extract_and_validate_patch(
                    invivo, exvivo, (x_start, y_start, z_start), size_new
                )

                if not using_new_keys:
                    return {"image": result["invivo"], "label": result["exvivo"]}
                return result
            except Exception as e:
                print(f"Patch extraction failed: {e}. Falling back to center patch.")
                center_invivo, center_exvivo = self._extract_center_patch(
                    invivo, exvivo
                )
                result = {"invivo": center_invivo, "exvivo": center_exvivo}
                if not using_new_keys:
                    return {"image": result["invivo"], "label": result["exvivo"]}
                return result

        candidates = []

        strategies = ["center", "random", "edge"]
        positions_per_strategy = max(1, max_attempts // len(strategies))

        for strategy in strategies:
            for pos_idx in range(positions_per_strategy):
                try:
                    if strategy == "random" and self.edge_aware:
                        x_start, y_start, z_start = (
                            self._get_content_biased_position_3d(
                                invivo, exvivo, size_old, size_new
                            )
                        )
                    else:
                        modified_idx = patch_index + pos_idx * 100
                        start_pos = self._get_patch_position(
                            strategy, size_old, size_new, modified_idx
                        )
                        x_start, y_start, z_start = start_pos

                    try:
                        patch = self._extract_and_validate_patch(
                            invivo, exvivo, (x_start, y_start, z_start), size_new
                        )

                        if not patch or "invivo" not in patch or "exvivo" not in patch:
                            continue

                        content_info = self._check_patch_content(
                            patch["invivo"], patch["exvivo"]
                        )

                        min_content_percent = 20.0
                        has_content = (
                            content_info["invivo_percentage"] >= min_content_percent
                            and content_info["exvivo_percentage"] >= min_content_percent
                        )

                        if not has_content:
                            continue

                        try:
                            patch["fingerprint"] = self._create_patch_fingerprint(
                                patch["invivo"]
                            )
                        except Exception as e:
                            if self.debug:
                                print(f"Debug: Error creating fingerprint: {e}")
                            patch["fingerprint"] = np.zeros((4, 8, 8))

                        max_similarity = 0
                        for prev_patch in previous_patches:
                            try:
                                if "fingerprint" in prev_patch:
                                    prev_fp = prev_patch["fingerprint"]
                                else:
                                    prev_key = (
                                        "invivo" if "invivo" in prev_patch else "image"
                                    )
                                    if prev_patch[prev_key] is None:
                                        continue
                                    prev_fp = self._create_patch_fingerprint(
                                        prev_patch[prev_key]
                                    )

                                similarity = self._calculate_fingerprint_similarity(
                                    patch["fingerprint"], prev_fp
                                )
                                max_similarity = max(max_similarity, similarity)
                            except Exception as e:
                                if self.debug:
                                    print(f"Debug: Error calculating similarity: {e}")
                                continue

                        patch["diversity"] = 1.0 - max_similarity
                        candidates.append(patch)

                        if patch["diversity"] > 1.0 - similarity_threshold:
                            if not using_new_keys:
                                patch["image"] = patch["invivo"]
                                patch["label"] = patch["exvivo"]
                                del patch["invivo"]
                                del patch["exvivo"]
                            return patch
                    except Exception as e:
                        if self.debug:
                            print(f"Debug: Error processing candidate: {e}")
                        continue
                except Exception as e:
                    if self.debug:
                        print(
                            f"Debug: Error generating position for strategy {strategy}: {e}"
                        )
                    continue

        if candidates:
            candidates.sort(key=lambda x: x.get("diversity", 0), reverse=True)
            best_patch = candidates[0]

            if self.debug:
                print(
                    f"Debug: Using best diverse patch (diversity score: {best_patch.get('diversity', 0):.2f})"
                )

            if not using_new_keys:
                best_patch["image"] = best_patch["invivo"]
                best_patch["label"] = best_patch["exvivo"]
                del best_patch["invivo"]
                del best_patch["exvivo"]
            return best_patch

        center_invivo, center_exvivo = self._extract_center_patch(invivo, exvivo)

        if using_new_keys:
            result_dict = {"invivo": center_invivo, "exvivo": center_exvivo}
        else:
            result_dict = {"image": center_invivo, "label": center_exvivo}

        try:
            result_dict["fingerprint"] = self._create_patch_fingerprint(
                center_invivo if center_invivo is not None else np.zeros((20, 20, 20))
            )
        except Exception:
            result_dict["fingerprint"] = np.zeros((4, 8, 8))

        result_dict["diversity"] = 1.0

        return result_dict

    def _create_patch_fingerprint(self, patch):
        """
        Create a compact fingerprint for a patch to enable efficient similarity checking.
        Enhanced with robust error handling.

        Args:
            patch: SimpleITK patch

        Returns:
            numpy.ndarray: Compact fingerprint
        """
        try:
            if patch is None:
                return np.zeros((4, 8, 8))

            try:
                array = sitk.GetArrayFromImage(patch)
            except Exception as e:
                if self.debug:
                    logger.debug(f"Error converting patch to array: {e}")
                return np.zeros((4, 8, 8))

            if array.ndim != 3 or 0 in array.shape:
                return np.zeros((4, 8, 8))

            z_size = min(4, array.shape[0])
            z_indices = np.linspace(0, array.shape[0] - 1, z_size, dtype=int)
            y_indices = np.linspace(0, array.shape[1] - 1, 8, dtype=int)
            x_indices = np.linspace(0, array.shape[2] - 1, 8, dtype=int)

            if len(z_indices) > 0 and len(y_indices) > 0 and len(x_indices) > 0:
                try:
                    fingerprint = array[np.ix_(z_indices, y_indices, x_indices)]
                except Exception:
                    fingerprint = np.zeros(
                        (len(z_indices), len(y_indices), len(x_indices))
                    )
                    for i, z in enumerate(z_indices):
                        for j, y in enumerate(y_indices):
                            for k, x in enumerate(x_indices):
                                if (
                                    z < array.shape[0]
                                    and y < array.shape[1]
                                    and x < array.shape[2]
                                ):
                                    fingerprint[i, j, k] = array[z, y, x]
            else:
                fingerprint = np.zeros((max(1, z_size), 8, 8))

            min_val = fingerprint.min()
            max_val = fingerprint.max()
            range_val = max_val - min_val
            if range_val > 1e-5:
                fingerprint = (fingerprint - min_val) / range_val
            else:
                fingerprint = np.zeros_like(fingerprint)

            return fingerprint
        except Exception as e:
            if self.debug:
                logger.debug(f"Error creating fingerprint: {e}")
            return np.zeros((4, 8, 8))

    def _calculate_fingerprint_similarity(self, fp1, fp2):
        """
        Calculate similarity between two fingerprints.
        Enhanced with robust error handling.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            float: Similarity score (0-1 where 1 is identical)
        """
        try:
            if fp1 is None or fp2 is None:
                return 0.0

            if not isinstance(fp1, np.ndarray) or not isinstance(fp2, np.ndarray):
                try:
                    fp1 = np.array(fp1)
                    fp2 = np.array(fp2)
                except Exception:
                    return 0.0

            if fp1.ndim == 0 or fp2.ndim == 0:
                return 0.0

            try:
                fp1_flat = fp1.flatten()
                fp2_flat = fp2.flatten()
            except Exception:
                return 0.0

            if len(fp1_flat) == 0 or len(fp2_flat) == 0:
                return 0.0

            min_length = min(len(fp1_flat), len(fp2_flat))
            fp1_flat = fp1_flat[:min_length]
            fp2_flat = fp2_flat[:min_length]

            if min_length < 2:
                return 0.0

            try:
                mean1 = np.mean(fp1_flat)
                mean2 = np.mean(fp2_flat)
                std1 = np.std(fp1_flat)
                std2 = np.std(fp2_flat)

                if std1 < 1e-5 or std2 < 1e-5:
                    return 0.0

                correlation = np.mean((fp1_flat - mean1) * (fp2_flat - mean2)) / (
                    std1 * std2
                )

                correlation = max(-1.0, min(1.0, correlation))

                return abs(correlation)
            except Exception:
                return 0.0

        except Exception as e:
            if self.debug:
                logger.debug(f"Error calculating similarity: {e}")
            return 0.0
