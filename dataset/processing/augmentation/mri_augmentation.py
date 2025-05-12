import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torchio as tio


class MRIAugmentation:
    """
    Advanced augmentation pipeline for prostate MRI domain translation using TorchIO.

    Specifically designed for in-vivo to ex-vivo prostate domain translation with
    appropriate background handling and MRI-specific transformations.
    """

    def __init__(
        self,
        augmentation_probs: Optional[Dict[str, float]] = None,
        random_order: bool = True,
        max_augmentations: int = 4,
        background_value: float = 0,
    ):
        """
        Initialize the MRI augmentation pipeline with TorchIO transformations.

        Args:
            augmentation_probs: Dictionary mapping transformation names to probabilities
            random_order: Whether to apply transformations in random order
            max_augmentations: Maximum number of augmentations to apply per sample
            background_value: Value representing background in the images
        """
        self.name = "MRIAugmentation"
        self.bg_value = background_value
        self.random_order = random_order
        self.max_augmentations = max_augmentations

        # Default probabilities for each transformation
        default_probs = {
            "affine": 0.9,  # Combines rotation, scaling, shearing
            "elastic": 0.6,  # Elastic deformation
            "bias_field": 0.6,  # Local intensity variations
            "motion": 0.5,  # Motion artifacts
            "ghosting": 0.5,  # MRI ghosting artifacts
            "spike": 0.3,  # MRI spike artifacts
            "noise": 0.5,  # Rician noise
            "gamma": 0.6,  # Contrast adjustment
            "blur": 0.4,  # Blur/PSF variations
            "anisotropy": 0.4,  # Resolution anisotropy
        }

        self.augmentation_probs = (
            augmentation_probs if augmentation_probs else default_probs
        )

        # Group transformations by category for balanced selection
        self.augmentation_categories = {
            "geometric": ["affine", "elastic", "anisotropy"],
            "intensity": ["bias_field", "gamma", "blur"],
            "noise": ["noise", "motion", "ghosting", "spike"],
        }

        # Initialize all possible transformations
        self._init_transformations()

    def _init_transformations(self):
        """Initialize all TorchIO transformations with appropriate parameters for prostate MRI"""
        self.transformations = {
            "affine": tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=30,
                translation=5,
                isotropic=False,
                center="image",
                default_pad_value=self.bg_value,
            ),
            "elastic": tio.RandomElasticDeformation(
                # Reduce control points in z-dimension (previously 8,8,8)
                num_control_points=(8, 8, 5),
                # Reduce maximum displacement (previously 1.5)
                max_displacement=1.0,
                locked_borders=2,
            ),
            "bias_field": tio.RandomBiasField(
                coefficients=0.5,
                order=3,
            ),
            "motion": tio.RandomMotion(
                degrees=10,
                translation=10,
                num_transforms=2,
            ),
            "ghosting": tio.RandomGhosting(
                num_ghosts=(4, 10),
                axes=(0, 1, 2),
                intensity=(0.5, 1.0),
            ),
            "spike": tio.RandomSpike(
                num_spikes=1,
                intensity=(1, 3),
            ),
            "noise": tio.RandomNoise(
                mean=0,
                std=(0, 0.03),
            ),
            "gamma": tio.RandomGamma(
                log_gamma=(-0.3, 0.3),
            ),
            "blur": tio.RandomBlur(
                std=(0, 0.8),
            ),
            "anisotropy": tio.RandomAnisotropy(
                axes=(0, 1, 2),
                downsampling=(1.5, 2.5),
            ),
        }

    def _select_balanced_augmentations(self, available_augs: List[str]) -> List[str]:
        """
        Select a balanced set of augmentations ensuring diversity across categories.

        Args:
            available_augs: List of available augmentation names

        Returns:
            List of selected augmentation names to apply
        """
        selected_augs = []
        max_augs = min(self.max_augmentations, len(available_augs))

        # Group available augmentations by category
        category_augs = {cat: [] for cat in self.augmentation_categories.keys()}

        for aug in available_augs:
            for cat, cat_augs in self.augmentation_categories.items():
                if aug in cat_augs:
                    category_augs[cat].append(aug)
                    break

        # Select from geometric category (1-2 transformations)
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
                    available_augs.remove(selected_geom)
                else:
                    selected_geoms = np.random.choice(
                        category_augs["geometric"],
                        size=num_geom,
                        replace=False,
                        p=geom_probs,
                    )
                    for aug in selected_geoms:
                        selected_augs.append(aug)
                        if aug in available_augs:
                            available_augs.remove(aug)

        # Select from intensity category (1 transformation)
        if category_augs["intensity"]:
            int_probs = [
                self.augmentation_probs.get(aug, 0.5)
                for aug in category_augs["intensity"]
            ]
            int_probs_sum = sum(int_probs)

            if int_probs_sum > 0:
                int_probs = [p / int_probs_sum for p in int_probs]
                selected_int = np.random.choice(category_augs["intensity"], p=int_probs)

                if selected_int in available_augs:
                    selected_augs.append(selected_int)
                    available_augs.remove(selected_int)

        # Select from noise category (1 transformation)
        if category_augs["noise"]:
            noise_probs = [
                self.augmentation_probs.get(aug, 0.5) for aug in category_augs["noise"]
            ]
            noise_probs_sum = sum(noise_probs)

            if noise_probs_sum > 0:
                noise_probs = [p / noise_probs_sum for p in noise_probs]
                selected_noise = np.random.choice(category_augs["noise"], p=noise_probs)

                if selected_noise in available_augs:
                    selected_augs.append(selected_noise)
                    available_augs.remove(selected_noise)

        # Fill remaining slots if needed
        remaining_slots = max_augs - len(selected_augs)
        if remaining_slots > 0 and available_augs:
            remaining_probs = [
                self.augmentation_probs.get(aug, 0.5) for aug in available_augs
            ]
            remaining_probs_sum = sum(remaining_probs)

            if remaining_probs_sum > 0:
                remaining_probs = [p / remaining_probs_sum for p in remaining_probs]

                num_to_select = min(remaining_slots, len(available_augs))
                if num_to_select > 0:
                    additional_augs = np.random.choice(
                        available_augs,
                        size=num_to_select,
                        replace=False,
                        p=remaining_probs,
                    )
                    selected_augs.extend(additional_augs)

        return selected_augs

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply the selected augmentations to the given sample.
        Modified to implement adaptive elastic deformation parameters.

        Args:
            sample: Dictionary with invivo/exvivo or image/label keys

        Returns:
            Augmented sample with same key convention
        """
        try:
            # Determine input key format
            if "invivo" in sample and "exvivo" in sample:
                input_key, output_key = "invivo", "exvivo"
                using_new_keys = True
            elif "image" in sample and "label" in sample:
                input_key, output_key = "image", "label"
                using_new_keys = False
            else:
                raise KeyError(
                    "Sample must contain either 'invivo'/'exvivo' or 'image'/'label' keys"
                )

            # Store original images for validation
            invivo_img = sample[input_key]
            exvivo_img = sample[output_key]

            # Adapt elastic deformation parameters based on image size if needed
            if "elastic" in self.transformations:
                # Get image dimensions
                size = invivo_img.GetSize()
                z_size = size[2]

                # Adjust elastic deformation parameters for small z dimensions
                if z_size < 16:
                    # For very small images, use fewer control points and smaller displacement
                    self.transformations["elastic"] = tio.RandomElasticDeformation(
                        num_control_points=(6, 6, 3),
                        max_displacement=0.5,
                        locked_borders=1,
                    )
                elif z_size < 32:
                    # For small images, use modest parameters
                    self.transformations["elastic"] = tio.RandomElasticDeformation(
                        num_control_points=(7, 7, 4),
                        max_displacement=0.8,
                        locked_borders=2,
                    )
                else:
                    # For larger images, use default parameters but still safe
                    self.transformations["elastic"] = tio.RandomElasticDeformation(
                        num_control_points=(8, 8, 5),
                        max_displacement=1.0,
                        locked_borders=2,
                    )

            # Select augmentations to apply
            all_augs = []
            for aug_name, prob in self.augmentation_probs.items():
                if np.random.rand() < prob:
                    all_augs.append(aug_name)

            selected_augs = self._select_balanced_augmentations(all_augs.copy())
            aug_desc = ", ".join(selected_augs) if selected_augs else "none"
            print(f"Selected augmentations: {aug_desc}")

            if not selected_augs:
                return sample

            # Create TorchIO subject from the images
            subject_dict = {
                "invivo": tio.ScalarImage.from_sitk(invivo_img),
                "exvivo": tio.ScalarImage.from_sitk(
                    exvivo_img
                ),  # Both are scalar images
            }
            subject = tio.Subject(subject_dict)

            # Apply transformations in sequence or random order
            if self.random_order and len(selected_augs) > 1:
                # Ensure geometric transforms are applied first for better results
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

                ordered_augs = geom_augs + other_augs
            else:
                ordered_augs = selected_augs

            # Apply each transformation
            for aug_name in ordered_augs:
                try:
                    # Some augmentations should only apply to the input image (invivo)
                    if aug_name in [
                        "bias_field",
                        "noise",
                        "gamma",
                        "blur",
                        "motion",
                        "ghosting",
                        "spike",
                    ]:
                        # Create a temporary subject with only the invivo image
                        temp_subject = tio.Subject({"invivo": subject["invivo"]})

                        # Apply transform
                        transform = self.transformations[aug_name]
                        transformed_temp = transform(temp_subject)

                        # Update only the invivo image in the original subject
                        subject["invivo"] = transformed_temp["invivo"]
                    else:
                        # Apply transform to both images
                        transform = self.transformations[aug_name]
                        subject = transform(subject)

                    print(f"Applied {aug_name}")
                except Exception as e:
                    print(f"Warning: {aug_name} failed with error: {e}")

                    # If elastic transformation failed, retry with more conservative parameters
                    if aug_name == "elastic":
                        try:
                            print(
                                "Retrying elastic deformation with safer parameters..."
                            )
                            safer_transform = tio.RandomElasticDeformation(
                                num_control_points=(4, 4, 3),
                                max_displacement=0.3,
                                locked_borders=2,
                            )

                            if "bias_field" in [
                                "bias_field",
                                "noise",
                                "gamma",
                                "blur",
                                "motion",
                                "ghosting",
                                "spike",
                            ]:
                                temp_subject = tio.Subject(
                                    {"invivo": subject["invivo"]}
                                )
                                transformed_temp = safer_transform(temp_subject)
                                subject["invivo"] = transformed_temp["invivo"]
                            else:
                                subject = safer_transform(subject)

                            print("Applied elastic deformation with safer parameters")
                        except Exception as retry_error:
                            print(f"Retry also failed: {retry_error}")

            # Validate the transformation (check if the statistics are reasonable)
            original_invivo_array = np.asarray(
                tio.ScalarImage.from_sitk(invivo_img).data
            )
            transformed_invivo_array = np.asarray(subject["invivo"].data)

            if not self._validate_image_statistics(
                transformed_invivo_array, original_invivo_array
            ):
                print(
                    "Warning: Final image validation failed. Reverting to original images."
                )
                return sample

            # Convert back to SimpleITK
            result = {
                input_key: subject["invivo"].as_sitk(),
                output_key: subject["exvivo"].as_sitk(),
            }

            return result

        except Exception as e:
            print(f"Error in augmentation: {e}")
            return sample

    def _validate_image_statistics(
        self, image_array: np.ndarray, original_array: np.ndarray, epsilon: float = 1e-5
    ) -> bool:
        """
        Validate image statistics to ensure augmentation hasn't drastically
        changed image characteristics. Modified to be more permissive.

        Args:
            image_array: Augmented image array
            original_array: Original image array
            epsilon: Tolerance for identifying background

        Returns:
            Boolean indicating if image statistics are valid
        """
        # Get background masks
        bg_mask_orig = np.isclose(original_array, self.bg_value, atol=epsilon)
        fg_mask_orig = ~bg_mask_orig

        bg_mask_new = np.isclose(image_array, self.bg_value, atol=epsilon)
        fg_mask_new = ~bg_mask_new

        if not np.any(fg_mask_orig) or not np.any(fg_mask_new):
            return True

        # Check foreground ratio (more permissive)
        fg_ratio_orig = np.mean(fg_mask_orig)
        fg_ratio_new = np.mean(fg_mask_new)

        # Changed from 0.05/5.0 to 0.01/10.0
        if fg_ratio_new < 0.01 * fg_ratio_orig or fg_ratio_new > 10.0 * fg_ratio_orig:
            return False

        # Compare foreground statistics
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

        # Check if statistics are within reasonable bounds (more permissive)
        # Changed from 0.7 to 0.9
        if abs(new_mean - orig_mean) > 0.9 * orig_range:
            return False

        # Changed from 0.1/5.0 to 0.05/10.0
        if new_std < 0.05 * orig_std or new_std > 10.0 * orig_std:
            return False

        # Changed from 0.2 to 0.1
        if new_range < 0.1 * orig_range:
            return False

        return True


class DomainSpecificAugmentation:
    """
    Domain-specific augmentation pipeline that applies different transforms
    to in-vivo and ex-vivo MRI data to better simulate domain characteristics.
    """

    def __init__(self, background_value: float = 0):
        """
        Initialize domain-specific augmentation pipelines.

        Args:
            background_value: Value representing background in the images
        """
        self.bg_value = background_value

        # In-vivo specific transformations (more noise, motion, etc.)
        self.invivo_transforms = tio.Compose(
            [
                tio.RandomBiasField(coefficients=0.3, p=0.7),
                tio.OneOf(
                    {
                        tio.RandomMotion(degrees=5, translation=5): 0.5,
                        tio.RandomGhosting(axes=1, intensity=(0.5, 0.8)): 0.3,
                        tio.RandomSpike(num_spikes=1, intensity=(1, 2)): 0.2,
                    },
                    p=0.7,
                ),
                tio.RandomNoise(std=(0, 0.1), p=0.6),
            ]
        )

        # Ex-vivo specific transformations (different contrast, less noise)
        self.exvivo_transforms = tio.Compose(
            [
                tio.RandomBiasField(coefficients=0.2, p=0.5),
                tio.RandomBlur(std=(0, 1.0), p=0.4),
                tio.RandomNoise(std=(0, 0.05), p=0.4),
            ]
        )

        # Spatial transforms that should be applied consistently to both domains
        self.spatial_transforms = tio.Compose(
            [
                tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.8),
                tio.RandomElasticDeformation(
                    num_control_points=7, max_displacement=4, p=0.5
                ),
                tio.OneOf(
                    {
                        tio.RandomAnisotropy(
                            axes=(0, 1, 2), downsampling=(1.5, 2.5)
                        ): 0.5,
                        tio.RandomFlip(axes=(0, 1)): 0.5,
                    },
                    p=0.5,
                ),
            ]
        )

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply domain-specific augmentations to the sample.

        Args:
            sample: Dictionary with invivo/exvivo or image/label keys

        Returns:
            Augmented sample with same key convention
        """
        try:
            # Determine input key format
            if "invivo" in sample and "exvivo" in sample:
                input_key, output_key = "invivo", "exvivo"
            elif "image" in sample and "label" in sample:
                input_key, output_key = "image", "label"
            else:
                raise KeyError(
                    "Sample must contain either 'invivo'/'exvivo' or 'image'/'label' keys"
                )

            # Convert to TorchIO format
            subject_dict = {
                "invivo": tio.ScalarImage.from_sitk(sample[input_key]),
                "exvivo": tio.ScalarImage.from_sitk(
                    sample[output_key]
                ),  # Both are scalar images
            }
            subject = tio.Subject(subject_dict)

            # Create a shared random seed for consistent spatial transforms
            seed = np.random.randint(0, 2147483647)

            # Apply spatial transforms to both images
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            subject = self.spatial_transforms(subject)

            # Apply in-vivo specific transforms
            torch.manual_seed(np.random.randint(0, 2147483647))
            invivo_subject = tio.Subject({"image": subject["invivo"]})
            invivo_subject = self.invivo_transforms(invivo_subject)
            subject["invivo"] = invivo_subject["image"]

            # Apply ex-vivo specific transforms
            torch.manual_seed(np.random.randint(0, 2147483647))
            exvivo_subject = tio.Subject({"image": subject["exvivo"]})
            exvivo_subject = self.exvivo_transforms(exvivo_subject)
            subject["exvivo"] = exvivo_subject["image"]

            # Convert back to SimpleITK and return
            return {
                input_key: subject["invivo"].as_sitk(),
                output_key: subject["exvivo"].as_sitk(),
            }

        except Exception as e:
            print(f"Error in domain-specific augmentation: {e}")
            return sample
