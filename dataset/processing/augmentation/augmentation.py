import gc
import random

import SimpleITK as sitk
import numpy as np
import torch
from scipy import ndimage


class Augmentation(object):
    """
    Fixed augmentation pipeline for prostate MRI domain translation.

    Special emphasis on proper background handling for Z-score normalized
    images where background = -1, and slightly stronger augmentations.
    """

    def __init__(self, augmentation_probs=None, random_order=True, max_augmentations=4):
        """
        Args:
            augmentation_probs (dict): Dictionary mapping augmentation function names
                to their probabilities (between 0 and 1).
            random_order (bool): Whether to apply the chosen augmentations in random order.
            max_augmentations (int): Maximum number of augmentations to apply to a single sample
        """
        self.name = "Augmentation"

        default_probs = {
            "apply_rotation": 0.9,
            "apply_shearing": 0.6,
            "apply_zoom": 0.6,
            "apply_contrast_adjustment": 0.6,
            "apply_elastic_deformation": 0.6,
            "apply_local_intensity": 0.6,
            "apply_rician_noise": 0.5,
            "apply_motion_blur": 0.5,
        }

        self.augmentation_probs = (
            augmentation_probs if augmentation_probs is not None else default_probs
        )

        self.random_order = random_order
        self.max_augmentations = max_augmentations

        self.paired_augs = [
            "apply_rotation",
            "apply_shearing",
            "apply_zoom",
            "apply_contrast_adjustment",
            "apply_elastic_deformation",
            "apply_local_intensity",
            "apply_rician_noise",
            "apply_motion_blur",
        ]

        self.bg_value = 0

        self.slice_chunk_size = 8

        self.augmentation_categories = {
            "geometric": [
                "apply_rotation",
                "apply_shearing",
                "apply_zoom",
                "apply_elastic_deformation",
            ],
            "intensity": ["apply_contrast_adjustment", "apply_local_intensity"],
            "noise": ["apply_rician_noise", "apply_motion_blur"],
        }

    def _select_balanced_augmentations(self, all_augs):
        """
        Select a balanced set of augmentations ensuring diversity and preventing over-augmentation.
        Enhanced to provide more diverse combinations and balanced selection.

        Args:
            all_augs: List of all available augmentations

        Returns:
            List of selected augmentations to apply
        """
        selected_augs = []

        max_augs = min(self.max_augmentations + 1, len(all_augs))

        category_augs = {cat: [] for cat in self.augmentation_categories.keys()}

        for aug in all_augs:
            for cat, cat_augs in self.augmentation_categories.items():
                if aug in cat_augs:
                    category_augs[cat].append(aug)
                    break

        if category_augs["geometric"]:
            geom_probs = [
                self.augmentation_probs.get(aug, 0.5)
                for aug in category_augs["geometric"]
            ]
            geom_probs_sum = sum(geom_probs)
            if geom_probs_sum > 0:
                geom_probs = [p / geom_probs_sum for p in geom_probs]
                num_geom = np.random.choice([1, 2], p=[0.3, 0.7])
                num_geom = min(num_geom, len(category_augs["geometric"]))

                if num_geom == 1:
                    selected_geom = np.random.choice(
                        category_augs["geometric"], p=geom_probs
                    )
                    selected_augs.append(selected_geom)
                    all_augs.remove(selected_geom)
                else:
                    selected_geoms = np.random.choice(
                        category_augs["geometric"],
                        size=num_geom,
                        replace=False,
                        p=geom_probs,
                    )
                    for aug in selected_geoms:
                        selected_augs.append(aug)
                        all_augs.remove(aug)

        if category_augs["intensity"]:
            int_probs = [
                self.augmentation_probs.get(aug, 0.5)
                for aug in category_augs["intensity"]
            ]
            int_probs_sum = sum(int_probs)
            if int_probs_sum > 0:
                int_probs = [p / int_probs_sum for p in int_probs]
                selected_int = np.random.choice(category_augs["intensity"], p=int_probs)

                if selected_int in all_augs:
                    selected_augs.append(selected_int)
                    all_augs.remove(selected_int)

        if category_augs["noise"]:
            noise_probs = [
                self.augmentation_probs.get(aug, 0.5) for aug in category_augs["noise"]
            ]
            noise_probs_sum = sum(noise_probs)
            if noise_probs_sum > 0:
                noise_probs = [p / noise_probs_sum for p in noise_probs]
                selected_noise = np.random.choice(category_augs["noise"], p=noise_probs)

                if selected_noise in all_augs:
                    selected_augs.append(selected_noise)
                    all_augs.remove(selected_noise)

        remaining_slots = max_augs - len(selected_augs)
        if remaining_slots > 0 and all_augs:
            remaining_probs = [
                self.augmentation_probs.get(aug, 0.5) for aug in all_augs
            ]
            remaining_probs_sum = sum(remaining_probs)

            if remaining_probs_sum > 0:
                remaining_probs = [p / remaining_probs_sum for p in remaining_probs]

                num_to_select = min(remaining_slots, len(all_augs))
                if num_to_select > 0:
                    additional_augs = np.random.choice(
                        all_augs, size=num_to_select, replace=False, p=remaining_probs
                    )
                    selected_augs.extend(additional_augs)

        return selected_augs

    def __call__(self, sample):
        """
        Apply augmentations with improved background preservation and enhanced diversity.

        Args:
            sample: Dictionary with invivo/exvivo or image/label keys

        Returns:
            Augmented sample with same key convention
        """
        try:
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

            original_invivo, original_exvivo = invivo, exvivo

            all_augs = []

            probability_boost = 0.1

            for aug in self.paired_augs:
                prob = min(
                    1.0, self.augmentation_probs.get(aug, 0.5) + probability_boost
                )
                if np.random.rand() < prob:
                    all_augs.append(aug)

            selected_augs = self._select_balanced_augmentations(all_augs.copy())

            aug_desc = ", ".join(selected_augs) if selected_augs else "none"
            print(f"Selected augmentations: {aug_desc}")

            if self.random_order and len(selected_augs) > 1:
                geom_augs = [
                    aug
                    for aug in selected_augs
                    if aug in self.augmentation_categories["geometric"]
                ]
                other_augs = [aug for aug in selected_augs if aug not in geom_augs]

                if len(geom_augs) > 1:
                    random.shuffle(geom_augs)
                if len(other_augs) > 1:
                    random.shuffle(other_augs)

                if geom_augs and other_augs:
                    selected_augs = []
                    while geom_augs or other_augs:
                        if geom_augs:
                            selected_augs.append(geom_augs.pop(0))
                        if other_augs:
                            selected_augs.append(other_augs.pop(0))
                else:
                    selected_augs = geom_augs + other_augs
                    random.shuffle(selected_augs)

            for aug in selected_augs:
                try:
                    func = getattr(self, aug)
                    invivo, exvivo = func(invivo, exvivo)
                    print(f"Applied {aug}")
                except Exception as e:
                    print(f"Warning: {aug} failed with error {e}")
                    invivo, exvivo = original_invivo, original_exvivo

            invivo_array = sitk.GetArrayFromImage(invivo)
            original_array = sitk.GetArrayFromImage(original_invivo)

            if not self._validate_image_statistics(invivo_array, original_array):
                print(
                    "Warning: Final image validation failed. Reverting to original images."
                )
                invivo, exvivo = original_invivo, original_exvivo

            if using_new_keys:
                return {"invivo": invivo, "exvivo": exvivo}
            else:
                return {"image": invivo, "label": exvivo}

        except Exception as e:
            print(f"Error in augmentation: {e}")
            return sample

    def _cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_background_mask(self, array, epsilon=1e-5):
        """
        Create a precise background mask for Z-score normalized MRI.

        Args:
            array: Image array with background = -1
            epsilon: Tolerance for background identification

        Returns:
            Boolean mask where True indicates background
        """
        return np.isclose(array, self.bg_value, atol=epsilon)

    def _preserve_background_exactly(self, array, mask=None, original=None):
        """
        Precisely preserve background values in the transformed image.

        Args:
            array: Image array that may have interpolation artifacts
            mask: Pre-computed background mask (if available)
            original: Original image to extract background mask from (if mask not provided)

        Returns:
            Array with background exactly preserved
        """
        if mask is None:
            if original is not None:
                mask = self._get_background_mask(original)
            else:
                mask = self._get_background_mask(array)

        array[mask] = self.bg_value

        return array

    def _validate_image_statistics(self, image_array, original_array, epsilon=1e-5):
        """
        Validate image statistics to ensure the augmentation hasn't drastically
        changed the image characteristics. More permissive than previous version.

        Args:
            image_array: Augmented image array
            original_array: Original image array
            epsilon: Tolerance for identifying background

        Returns:
            Boolean indicating if image statistics are valid
        """
        bg_mask_orig = self._get_background_mask(original_array, epsilon)
        fg_mask_orig = ~bg_mask_orig

        bg_mask_new = self._get_background_mask(image_array, epsilon)
        fg_mask_new = ~bg_mask_new

        if not np.any(fg_mask_orig) or not np.any(fg_mask_new):
            return True

        fg_ratio_orig = np.mean(fg_mask_orig)
        fg_ratio_new = np.mean(fg_mask_new)

        if fg_ratio_new < 0.05 * fg_ratio_orig or fg_ratio_new > 5.0 * fg_ratio_orig:
            return False

        fg_orig = original_array[fg_mask_orig]
        fg_new = image_array[fg_mask_new]

        orig_mean = np.mean(fg_orig)
        orig_std = np.std(fg_orig)
        orig_min = np.min(fg_orig)
        orig_max = np.max(fg_orig)
        orig_range = orig_max - orig_min

        new_mean = np.mean(fg_new)
        new_std = np.std(fg_new)
        new_min = np.min(fg_new)
        new_max = np.max(fg_new)
        new_range = new_max - new_min

        if abs(new_mean - orig_mean) > 0.7 * orig_range:
            return False

        if new_std < 0.1 * orig_std or new_std > 5.0 * orig_std:
            return False

        if new_range < 0.2 * orig_range:
            return False

        return True

    def apply_rotation(self, invivo, exvivo, max_angle=60):
        """
        Apply rotation only around the z-axis (in x-y plane) with perfect background preservation.
        Enhanced to support larger rotation angles and ensure diversity.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            max_angle: Maximum rotation angle (default increased to 60 degrees)

        Returns:
            Rotated invivo and exvivo images
        """
        invivo_array = sitk.GetArrayFromImage(invivo)
        exvivo_array = sitk.GetArrayFromImage(exvivo)

        angle_type = np.random.choice(["small", "medium", "large"], p=[0.4, 0.4, 0.2])
        if angle_type == "small":
            angle = np.random.uniform(-15, 15)
        elif angle_type == "medium":
            angle_abs = np.random.uniform(15, 45)
            angle = angle_abs * np.random.choice([-1, 1])
        else:
            angle_abs = np.random.uniform(45, max_angle)
            angle = angle_abs * np.random.choice([-1, 1])

        invivo_result = np.ones_like(invivo_array) * self.bg_value
        exvivo_result = np.ones_like(exvivo_array) * self.bg_value

        for z in range(invivo_array.shape[0]):
            invivo_slice = invivo_array[z]
            exvivo_slice = exvivo_array[z]

            bg_mask = np.isclose(invivo_slice, self.bg_value, atol=1e-5)
            if np.all(bg_mask):
                continue

            fg_mask = ~bg_mask

            fg_invivo = np.copy(invivo_slice)
            fg_exvivo = np.copy(exvivo_slice)

            rotated_invivo = ndimage.rotate(
                fg_invivo,
                angle,
                reshape=False,
                order=1,
                mode="constant",
                cval=self.bg_value,
            )

            rotated_exvivo = ndimage.rotate(
                fg_exvivo,
                angle,
                reshape=False,
                order=1,
                mode="constant",
                cval=self.bg_value,
            )

            rotated_mask = ndimage.rotate(
                fg_mask.astype(np.float32),
                angle,
                reshape=False,
                order=0,
                mode="constant",
                cval=0.0,
            )
            rotated_mask = rotated_mask > 0.5

            slice_result_invivo = np.ones_like(invivo_slice) * self.bg_value
            slice_result_invivo[rotated_mask] = rotated_invivo[rotated_mask]

            invivo_result[z] = slice_result_invivo
            exvivo_result[z] = rotated_exvivo

        invivo_output = sitk.GetImageFromArray(invivo_result)
        invivo_output.CopyInformation(invivo)

        exvivo_output = sitk.GetImageFromArray(exvivo_result)
        exvivo_output.CopyInformation(exvivo)

        return invivo_output, exvivo_output

    def apply_shearing(self, invivo, exvivo, shear_range=(-0.15, 0.15)):
        """
        Apply shearing with perfect background preservation and increased range.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            shear_range: Range of shear factors (increased from ±0.1 to ±0.15)

        Returns:
            Sheared invivo and exvivo images
        """
        invivo_array = sitk.GetArrayFromImage(invivo)
        exvivo_array = sitk.GetArrayFromImage(exvivo)

        if np.random.random() < 0.7:
            shear_level = np.random.choice(["medium", "high"], p=[0.6, 0.4])
            if shear_level == "medium":
                shear_x_abs = np.random.uniform(0.05, 0.1)
                shear_y_abs = np.random.uniform(0.05, 0.1)
            else:
                shear_x_abs = np.random.uniform(0.1, 0.15)
                shear_y_abs = np.random.uniform(0.1, 0.15)

            shear_x = shear_x_abs * np.random.choice([-1, 1])
            shear_y = shear_y_abs * np.random.choice([-1, 1])
        else:
            shear_x = np.random.uniform(-0.05, 0.05)
            shear_y = np.random.uniform(-0.05, 0.05)

        shear_matrix = np.array([[1, shear_x], [shear_y, 1]])

        invivo_result = np.ones_like(invivo_array) * self.bg_value
        exvivo_result = np.ones_like(exvivo_array) * self.bg_value

        for z in range(invivo_array.shape[0]):
            invivo_slice = invivo_array[z]
            exvivo_slice = exvivo_array[z]

            bg_mask = np.isclose(invivo_slice, self.bg_value, atol=1e-5)
            if np.all(bg_mask):
                continue

            fg_mask = ~bg_mask

            sheared_invivo = ndimage.affine_transform(
                invivo_slice,
                shear_matrix,
                order=1,
                mode="constant",
                cval=self.bg_value,
            )

            sheared_exvivo = ndimage.affine_transform(
                exvivo_slice,
                shear_matrix,
                order=1,
                mode="constant",
                cval=self.bg_value,
            )

            sheared_mask = ndimage.affine_transform(
                fg_mask.astype(np.float32),
                shear_matrix,
                order=0,
                mode="constant",
                cval=0.0,
            )
            sheared_mask = sheared_mask > 0.5

            slice_result_invivo = np.ones_like(invivo_slice) * self.bg_value
            slice_result_invivo[sheared_mask] = sheared_invivo[sheared_mask]

            invivo_result[z] = slice_result_invivo
            exvivo_result[z] = sheared_exvivo

        invivo_output = sitk.GetImageFromArray(invivo_result)
        invivo_output.CopyInformation(invivo)

        exvivo_output = sitk.GetImageFromArray(exvivo_result)
        exvivo_output.CopyInformation(exvivo)

        return invivo_output, exvivo_output

    def apply_zoom(self, invivo, exvivo, zoom_range=(0.75, 1.25)):
        """
        Apply zoom to account for variations in prostate size with more diversity.
        With improved background handling and increased zoom range.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            zoom_range: Range of zoom factors (increased to 0.75-1.25)

        Returns:
            Zoomed invivo and exvivo images
        """
        invivo_array = sitk.GetArrayFromImage(invivo)
        exvivo_array = sitk.GetArrayFromImage(exvivo)

        invivo_bg_mask = self._get_background_mask(invivo_array)
        exvivo_bg_mask = self._get_background_mask(exvivo_array)

        zoom_category = np.random.choice(["low", "medium", "high"], p=[0.3, 0.4, 0.3])

        if zoom_category == "low":
            zoom_factor_xy = np.random.uniform(0.75, 0.9)
        elif zoom_category == "medium":
            zoom_factor_xy = np.random.uniform(0.9, 1.1)
        else:
            zoom_factor_xy = np.random.uniform(1.1, 1.25)

        invivo_result = np.ones_like(invivo_array) * self.bg_value
        exvivo_result = np.ones_like(exvivo_array) * self.bg_value

        for z in range(invivo_array.shape[0]):
            invivo_slice = invivo_array[z]
            exvivo_slice = exvivo_array[z]
            slice_invivo_bg = invivo_bg_mask[z]
            slice_exvivo_bg = exvivo_bg_mask[z]

            if np.all(slice_invivo_bg) and np.all(slice_exvivo_bg):
                continue

            zoom_tuple = (zoom_factor_xy, zoom_factor_xy)

            if zoom_factor_xy < 1.0:
                invivo_fg = np.copy(invivo_slice)
                exvivo_fg = np.copy(exvivo_slice)

                zoomed_invivo = ndimage.zoom(
                    invivo_fg,
                    zoom_tuple,
                    order=1,
                    mode="constant",
                    cval=self.bg_value,
                    prefilter=True,
                )

                zoomed_exvivo = ndimage.zoom(
                    exvivo_fg,
                    zoom_tuple,
                    order=1,
                    mode="constant",
                    cval=self.bg_value,
                    prefilter=False,
                )

                pad_y = (invivo_slice.shape[0] - zoomed_invivo.shape[0]) // 2
                pad_x = (invivo_slice.shape[1] - zoomed_invivo.shape[1]) // 2

                temp_invivo = np.ones_like(invivo_slice) * self.bg_value
                temp_exvivo = np.ones_like(exvivo_slice) * self.bg_value

                if pad_y >= 0 and pad_x >= 0:
                    end_y = min(pad_y + zoomed_invivo.shape[0], invivo_slice.shape[0])
                    end_x = min(pad_x + zoomed_invivo.shape[1], invivo_slice.shape[1])

                    temp_invivo[pad_y:end_y, pad_x:end_x] = zoomed_invivo[
                        : end_y - pad_y, : end_x - pad_x
                    ]
                    temp_exvivo[pad_y:end_y, pad_x:end_x] = zoomed_exvivo[
                        : end_y - pad_y, : end_x - pad_x
                    ]

                    invivo_result[z] = temp_invivo
                    exvivo_result[z] = temp_exvivo

            else:
                crop_size_y = int(invivo_slice.shape[0] / zoom_factor_xy)
                crop_size_x = int(invivo_slice.shape[1] / zoom_factor_xy)

                start_y = (invivo_slice.shape[0] - crop_size_y) // 2
                start_x = (invivo_slice.shape[1] - crop_size_x) // 2

                start_y = max(0, start_y)
                start_x = max(0, start_x)
                end_y = min(start_y + crop_size_y, invivo_slice.shape[0])
                end_x = min(start_x + crop_size_x, invivo_slice.shape[1])

                cropped_invivo = invivo_slice[start_y:end_y, start_x:end_x]
                cropped_exvivo = exvivo_slice[start_y:end_y, start_x:end_x]

                cropped_bg_invivo = slice_invivo_bg[start_y:end_y, start_x:end_x]

                zoomed_invivo = ndimage.zoom(
                    cropped_invivo,
                    zoom_tuple,
                    order=1,
                    mode="constant",
                    cval=self.bg_value,
                    prefilter=True,
                )

                zoomed_exvivo = ndimage.zoom(
                    cropped_exvivo,
                    zoom_tuple,
                    order=1,
                    mode="constant",
                    cval=self.bg_value,
                    prefilter=False,
                )

                zoomed_bg = ndimage.zoom(
                    cropped_bg_invivo.astype(np.float32),
                    zoom_tuple,
                    order=0,
                    mode="constant",
                    cval=1.0,
                    prefilter=False,
                )
                zoomed_bg = zoomed_bg > 0.5

                y_size = min(zoomed_invivo.shape[0], invivo_slice.shape[0])
                x_size = min(zoomed_invivo.shape[1], invivo_slice.shape[1])

                if (
                    zoomed_invivo.shape[0] > invivo_slice.shape[0]
                    or zoomed_invivo.shape[1] > invivo_slice.shape[1]
                ):
                    crop_start_y = (zoomed_invivo.shape[0] - invivo_slice.shape[0]) // 2
                    crop_start_x = (zoomed_invivo.shape[1] - invivo_slice.shape[1]) // 2
                    crop_start_y = max(0, crop_start_y)
                    crop_start_x = max(0, crop_start_x)

                    invivo_result[z] = zoomed_invivo[
                        crop_start_y : crop_start_y + y_size,
                        crop_start_x : crop_start_x + x_size,
                    ]

                    exvivo_result[z] = zoomed_exvivo[
                        crop_start_y : crop_start_y + y_size,
                        crop_start_x : crop_start_x + x_size,
                    ]

                    zoomed_bg_cropped = zoomed_bg[
                        crop_start_y : crop_start_y + y_size,
                        crop_start_x : crop_start_x + x_size,
                    ]

                    invivo_result[z][zoomed_bg_cropped] = self.bg_value
                else:
                    temp_invivo = np.ones_like(invivo_slice) * self.bg_value
                    temp_exvivo = np.ones_like(exvivo_slice) * self.bg_value

                    temp_invivo[:y_size, :x_size] = zoomed_invivo[:y_size, :x_size]
                    temp_exvivo[:y_size, :x_size] = zoomed_exvivo[:y_size, :x_size]

                    temp_bg_mask = np.ones_like(slice_invivo_bg)
                    temp_bg_mask[:y_size, :x_size] = zoomed_bg[:y_size, :x_size]
                    temp_invivo[temp_bg_mask] = self.bg_value

                    invivo_result[z] = temp_invivo
                    exvivo_result[z] = temp_exvivo

        invivo_result[invivo_bg_mask] = self.bg_value
        exvivo_result[exvivo_bg_mask] = self.bg_value

        if not self._validate_image_statistics(invivo_result, invivo_array):
            return invivo, exvivo

        invivo_output = sitk.GetImageFromArray(invivo_result)
        invivo_output.CopyInformation(invivo)

        exvivo_output = sitk.GetImageFromArray(exvivo_result)
        exvivo_output.CopyInformation(exvivo)

        del invivo_result, exvivo_result, invivo_array, exvivo_array
        self._cleanup_memory()

        return invivo_output, exvivo_output

    def apply_contrast_adjustment(self, invivo, exvivo, range_min=-35, range_max=35):
        """
        Apply contrast adjustments to simulate differences between scanners.
        Enhanced with increased range and more diverse parameter selection.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            range_min: Minimum adjustment range (increased from -25 to -35)
            range_max: Maximum adjustment range (increased from 25 to 35)

        Returns:
            Contrast-adjusted invivo and unchanged exvivo
        """
        array = sitk.GetArrayFromImage(invivo)

        background_mask = self._get_background_mask(array)
        foreground_mask = ~background_mask

        if np.any(foreground_mask):
            fg_pixels = array[foreground_mask]

            min_val = fg_pixels.min()
            max_val = fg_pixels.max()

            del fg_pixels

            range_val = max_val - min_val + 1e-6

            adjustment_type = np.random.choice(
                ["mild", "moderate", "strong"], p=[0.3, 0.4, 0.3]
            )

            if adjustment_type == "mild":
                gamma = np.random.uniform(0.95, 1.05)
                noise_level = np.random.uniform(0, 0.01)
                brightness_shift = np.random.uniform(-0.02, 0.02) * range_val
            elif adjustment_type == "moderate":
                gamma = np.random.uniform(0.85, 1.15)
                noise_level = np.random.uniform(0.01, 0.02)
                brightness_shift = np.random.uniform(-0.05, 0.05) * range_val
            else:
                gamma = np.random.uniform(0.7, 1.3)
                noise_level = np.random.uniform(0.02, 0.04)
                brightness_shift = np.random.uniform(-0.08, 0.08) * range_val

            result_array = np.copy(array)

            total_slices = array.shape[0]

            for chunk_start in range(0, total_slices, self.slice_chunk_size):
                chunk_end = min(chunk_start + self.slice_chunk_size, total_slices)

                for z in range(chunk_start, chunk_end):
                    slice_fg_mask = foreground_mask[z]
                    if not np.any(slice_fg_mask):
                        continue

                    slice_fg = array[z, slice_fg_mask]

                    normalized = (slice_fg - min_val) / range_val
                    normalized = np.clip(normalized, 0.0, 1.0)

                    corrected = np.power(normalized + 1e-6, gamma)

                    corrected = corrected * range_val
                    corrected = np.clip(corrected, min_val, max_val)

                    corrected += brightness_shift

                    if noise_level > 0:
                        noise = np.random.normal(
                            0, noise_level * (max_val - min_val), slice_fg.shape
                        )
                        corrected += noise

                    result_array[z, slice_fg_mask] = corrected

                    del slice_fg, normalized, corrected
                    if noise_level > 0:
                        del noise

                self._cleanup_memory()

            result_array[background_mask] = self.bg_value

            del background_mask, foreground_mask
        else:
            result_array = array

        if not self._validate_image_statistics(result_array, array):
            return invivo, exvivo

        result = sitk.GetImageFromArray(result_array)
        result.CopyInformation(invivo)

        del result_array, array
        self._cleanup_memory()

        return result, exvivo

    def apply_elastic_deformation(self, invivo, exvivo, alpha=2.5, sigma=5):
        """
        Apply elastic deformation to simulate tissue variations.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            alpha: Deformation magnitude (increased from 1.5 to 2.0)
            sigma: Deformation smoothness

        Returns:
            Deformed invivo and exvivo images
        """
        invivo_array = sitk.GetArrayFromImage(invivo)
        exvivo_array = sitk.GetArrayFromImage(exvivo)

        invivo_bg_mask = self._get_background_mask(invivo_array)
        exvivo_bg_mask = self._get_background_mask(exvivo_array)

        invivo_result = np.ones_like(invivo_array) * self.bg_value
        exvivo_result = np.ones_like(exvivo_array) * self.bg_value

        for z in range(invivo_array.shape[0]):
            invivo_slice = invivo_array[z]
            exvivo_slice = exvivo_array[z]
            slice_invivo_bg = invivo_bg_mask[z]
            slice_exvivo_bg = exvivo_bg_mask[z]

            if np.all(slice_invivo_bg) and np.all(slice_exvivo_bg):
                continue

            dx = (
                ndimage.gaussian_filter(np.random.randn(*invivo_slice.shape), sigma)
                * alpha
            )
            dy = (
                ndimage.gaussian_filter(np.random.randn(*invivo_slice.shape), sigma)
                * alpha
            )

            y, x = np.meshgrid(
                np.arange(invivo_slice.shape[0]),
                np.arange(invivo_slice.shape[1]),
                indexing="ij",
            )
            indices = [y + dy, x + dx]

            deformed_invivo = ndimage.map_coordinates(
                invivo_slice, indices, order=1, mode="constant", cval=self.bg_value
            )
            deformed_exvivo = ndimage.map_coordinates(
                exvivo_slice, indices, order=1, mode="constant", cval=self.bg_value
            )

            deformed_bg = ndimage.map_coordinates(
                slice_invivo_bg.astype(np.float32),
                indices,
                order=0,
                mode="constant",
                cval=1.0,
            )
            deformed_bg = deformed_bg > 0.5

            deformed_invivo[deformed_bg] = self.bg_value

            invivo_result[z] = deformed_invivo
            exvivo_result[z] = deformed_exvivo

        invivo_result[invivo_bg_mask] = self.bg_value
        exvivo_result[exvivo_bg_mask] = self.bg_value

        if not self._validate_image_statistics(invivo_result, invivo_array):
            return invivo, exvivo

        result_invivo = sitk.GetImageFromArray(invivo_result)
        result_invivo.CopyInformation(invivo)

        result_exvivo = sitk.GetImageFromArray(exvivo_result)
        result_exvivo.CopyInformation(exvivo)

        del invivo_array, exvivo_array, invivo_result, exvivo_result
        self._cleanup_memory()

        return result_invivo, result_exvivo

    def apply_local_intensity(self, invivo, exvivo, scale_range=(0.92, 1.08)):
        """
        Apply localized intensity variations to simulate inhomogeneities.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            scale_range: Range of intensity scaling factors (increased from 0.96-1.04 to 0.92-1.08)

        Returns:
            Intensity-modified invivo and unchanged exvivo
        """
        array = sitk.GetArrayFromImage(invivo)

        background_mask = self._get_background_mask(array)
        foreground_mask = ~background_mask

        if np.any(foreground_mask):
            result_array = np.copy(array)

            shape = array.shape
            variation_field = np.ones(shape)

            num_centers = np.random.randint(2, 4)

            for _ in range(num_centers):
                z_center = np.random.randint(0, shape[0])
                y_center = np.random.randint(0, shape[1])
                x_center = np.random.randint(0, shape[2])

                strength = np.random.uniform(*scale_range)

                sigma_z = np.random.uniform(shape[0] / 6, shape[0] / 3)
                sigma_y = np.random.uniform(shape[1] / 6, shape[1] / 3)
                sigma_x = np.random.uniform(shape[2] / 6, shape[2] / 3)

                z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]

                gaussian = np.exp(
                    -(
                        ((z - z_center) / sigma_z) ** 2
                        + ((y - y_center) / sigma_y) ** 2
                        + ((x - x_center) / sigma_x) ** 2
                    )
                    / 2
                )

                gaussian = 1.0 + (gaussian * (strength - 1.0))

                variation_field *= gaussian

            result_array[foreground_mask] = (
                array[foreground_mask] * variation_field[foreground_mask]
            )

            result_array[background_mask] = self.bg_value
        else:
            result_array = array

        if not self._validate_image_statistics(result_array, array):
            return invivo, exvivo

        result = sitk.GetImageFromArray(result_array)
        result.CopyInformation(invivo)

        del array, result_array
        self._cleanup_memory()

        return result, exvivo

    def apply_rician_noise(self, invivo, exvivo, noise_level=(0.008, 0.015)):
        """
        Add Rician noise to simulate MRI acquisition noise.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            noise_level: Range of noise standard deviation (increased from 0.002-0.008 to 0.005-0.015)

        Returns:
            Noise-added invivo and unchanged exvivo
        """
        array = sitk.GetArrayFromImage(invivo)

        background_mask = self._get_background_mask(array)
        foreground_mask = ~background_mask

        if np.any(foreground_mask):
            fg_pixels = array[foreground_mask]
            fg_std = fg_pixels.std()
            noise_std = np.random.uniform(*noise_level) * fg_std

            noise_real = np.random.normal(0, noise_std, array.shape)
            noise_imag = np.random.normal(0, noise_std, array.shape)

            result_array = np.copy(array)
            result_array[foreground_mask] = np.sqrt(
                (array[foreground_mask] + noise_real[foreground_mask]) ** 2
                + noise_imag[foreground_mask] ** 2
            )

            result_array[background_mask] = self.bg_value
        else:
            result_array = array

        if not self._validate_image_statistics(result_array, array):
            return invivo, exvivo

        result = sitk.GetImageFromArray(result_array)
        result.CopyInformation(invivo)

        del array, result_array
        self._cleanup_memory()

        return result, exvivo

    def apply_motion_blur(self, invivo, exvivo, kernel_range=(3, 5)):
        """
        Apply directional blur to random slices to simulate remaining
        motion artifacts in in-vivo scans.

        Args:
            invivo: SimpleITK invivo image
            exvivo: SimpleITK exvivo image
            kernel_range: Range of blur kernel sizes

        Returns:
            Blur-added invivo and unchanged exvivo
        """
        array = sitk.GetArrayFromImage(invivo)

        background_mask = self._get_background_mask(array)

        num_slices = array.shape[0]
        num_to_blur = max(1, int(num_slices * np.random.uniform(0.05, 0.1)))
        blur_slices = np.random.choice(num_slices, size=num_to_blur, replace=False)

        result_array = np.copy(array)

        for z in blur_slices:
            slice_bg_mask = background_mask[z]
            if np.all(slice_bg_mask):
                continue

            kernel_size = np.random.randint(*kernel_range)
            if kernel_size % 2 == 0:
                kernel_size += 1

            angle = np.random.uniform(0, 180)

            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2

            for i in range(kernel_size):
                x = i - center
                y = int(np.round(np.tan(np.radians(angle)) * x))
                if abs(y) <= center:
                    kernel[center + y, i] = 1

            kernel = kernel / np.maximum(kernel.sum(), 1e-8)

            slice_fg = np.copy(array[z])
            slice_fg[slice_bg_mask] = self.bg_value

            blurred = ndimage.convolve(
                slice_fg, kernel, mode="constant", cval=self.bg_value
            )

            blurred[slice_bg_mask] = self.bg_value

            result_array[z] = blurred

        if not self._validate_image_statistics(result_array, array):
            return invivo, exvivo

        result = sitk.GetImageFromArray(result_array)
        result.CopyInformation(invivo)

        del array, result_array
        self._cleanup_memory()

        return result, exvivo

    def _select_balanced_augmentations(self, all_augs):
        """
        Select a balanced set of augmentations ensuring diversity and preventing over-augmentation.

        Args:
            all_augs: List of all available augmentations

        Returns:
            List of selected augmentations to apply
        """
        selected_augs = []

        max_augs = min(self.max_augmentations, len(all_augs))

        geom_augs = self.augmentation_categories["geometric"]
        valid_geom_augs = [aug for aug in geom_augs if aug in all_augs]

        if valid_geom_augs:
            geom_probs = [
                self.augmentation_probs.get(aug, 0.5) for aug in valid_geom_augs
            ]
            geom_probs_sum = sum(geom_probs)
            if geom_probs_sum > 0:
                geom_probs = [p / geom_probs_sum for p in geom_probs]
                selected_geom = np.random.choice(valid_geom_augs, p=geom_probs)
                selected_augs.append(selected_geom)
                all_augs.remove(selected_geom)

        intensity_augs = self.augmentation_categories["intensity"]
        valid_int_augs = [aug for aug in intensity_augs if aug in all_augs]

        if valid_int_augs:
            int_probs = [
                self.augmentation_probs.get(aug, 0.5) for aug in valid_int_augs
            ]
            int_probs_sum = sum(int_probs)
            if int_probs_sum > 0:
                int_probs = [p / int_probs_sum for p in int_probs]
                selected_int = np.random.choice(valid_int_augs, p=int_probs)
                selected_augs.append(selected_int)
                all_augs.remove(selected_int)

        remaining_slots = max_augs - len(selected_augs)
        if remaining_slots > 0 and all_augs:
            remaining_probs = [
                self.augmentation_probs.get(aug, 0.5) for aug in all_augs
            ]
            remaining_probs_sum = sum(remaining_probs)

            if remaining_probs_sum > 0:
                remaining_probs = [p / remaining_probs_sum for p in remaining_probs]

                num_to_select = min(remaining_slots, len(all_augs))
                if num_to_select > 0:
                    additional_augs = np.random.choice(
                        all_augs, size=num_to_select, replace=False, p=remaining_probs
                    )
                    selected_augs.extend(additional_augs)

        return selected_augs
