import math
from typing import Tuple, Dict, Any, List

import torch
import torch.nn.functional as F

from memory_utils import aggressive_memory_cleanup
from metrics import (
    calculate_metrics_with_batching,
    calculate_full_dataset_fid,
    crop_black_regions_aligned,
)
from visualization.plot_model_images import plot_full_validation_images


class ValidationHandler:
    """
    Handler for validation operations in CycleGAN training.

    Manages full image validation, sliding window validation for large volumes,
    metrics calculation, and visualization. Includes memory-optimized operations
    for handling large medical imaging data.
    """

    def __init__(self, model, device):
        """
        Initialize validation handler.

        Args:
            model: CycleGAN model for validation
            device: Device to run validation on (CPU or CUDA)
        """
        self.model = model
        self.device = device
        self.metrics_calculator = None
        self.using_cpu = device.type == "cpu"
        self.enable_full_metrics = True
        self.downscale_factor = 1.0

        self.use_sliding_window = True
        self.patch_size = [64, 64, 32]
        self.stride_inplane = 32
        self.stride_layer = 16
        self.min_patch_size = [16, 16, 8]

        self.real_A_collector = []
        self.fake_A_collector = []
        self.real_B_collector = []
        self.fake_B_collector = []
        self.collected_samples = 0

        self.enable_full_dataset_fid = False

    def reset_collectors(self):
        """
        Reset image collectors but preserve train_exvivo_collector.
        Used to clear accumulated images between validation runs.
        """
        train_exvivo_collector = None
        if hasattr(self, "train_exvivo_collector"):
            train_exvivo_collector = self.train_exvivo_collector

        self.real_A_collector = []
        self.fake_A_collector = []
        self.real_B_collector = []
        self.fake_B_collector = []
        self.collected_samples = 0

        if train_exvivo_collector is not None:
            self.train_exvivo_collector = train_exvivo_collector

        aggressive_memory_cleanup()

    def _generate_patch_indices(
        self, image_shape, patch_size, stride_inplane, stride_layer
    ) -> List[List[int]]:
        """
        Generate indices for patches in a sliding window approach.

        Args:
            image_shape: Shape of the image to generate patches for (D, H, W)
            patch_size: Size of each patch (D, H, W)
            stride_inplane: Stride for in-plane dimensions (H, W)
            stride_layer: Stride for layer dimension (D)

        Returns:
            List of patch indices [istart, iend, jstart, jend, kstart, kend]
        """
        inum = max(
            1,
            int(math.ceil((image_shape[0] - patch_size[0]) / float(stride_inplane)))
            + 1,
        )
        jnum = max(
            1,
            int(math.ceil((image_shape[1] - patch_size[1]) / float(stride_inplane)))
            + 1,
        )
        knum = max(
            1,
            int(math.ceil((image_shape[2] - patch_size[2]) / float(stride_layer))) + 1,
        )

        patch_indices = []

        for k in range(knum):
            for i in range(inum):
                for j in range(jnum):
                    istart = min(
                        i * stride_inplane, max(0, image_shape[0] - patch_size[0])
                    )
                    iend = min(istart + patch_size[0], image_shape[0])

                    if iend - istart < self.min_patch_size[0]:
                        continue

                    jstart = min(
                        j * stride_inplane, max(0, image_shape[1] - patch_size[1])
                    )
                    jend = min(jstart + patch_size[1], image_shape[1])

                    if jend - jstart < self.min_patch_size[1]:
                        continue

                    kstart = min(
                        k * stride_layer, max(0, image_shape[2] - patch_size[2])
                    )
                    kend = min(kstart + patch_size[2], image_shape[2])

                    if kend - kstart < self.min_patch_size[2]:
                        continue

                    if (
                        (iend - istart) >= self.min_patch_size[0]
                        and (jend - jstart) >= self.min_patch_size[1]
                        and (kend - kstart) >= self.min_patch_size[2]
                    ):
                        patch_indices.append([istart, iend, jstart, jend, kstart, kend])

        if not patch_indices:
            print(f"Warning: No valid patches found for image of shape {image_shape}.")
            print(
                f"Creating a single central patch with minimum sizes {self.min_patch_size}"
            )

            istart = max(0, image_shape[0] // 2 - self.min_patch_size[0] // 2)
            iend = min(image_shape[0], istart + self.min_patch_size[0])

            jstart = max(0, image_shape[1] // 2 - self.min_patch_size[1] // 2)
            jend = min(image_shape[1], jstart + self.min_patch_size[1])

            kstart = max(0, image_shape[2] // 2 - self.min_patch_size[2] // 2)
            kend = min(image_shape[2], kstart + self.min_patch_size[2])

            if (iend - istart) >= 4 and (jend - jstart) >= 4 and (kend - kstart) >= 4:
                patch_indices.append([istart, iend, jstart, jend, kstart, kend])
            else:
                print(
                    f"Cannot create valid patches for this image - dimensions too small"
                )

        return patch_indices

    def _initialize_metrics_calculator(self):
        """
        Initialize metrics calculator with enhanced error handling and compatibility checks.
        Creates and configures the metrics calculator with appropriate settings based on
        available memory and device capabilities.
        """
        if self.metrics_calculator is None:
            from metrics.val_metrics import MetricsCalculator

            enable_fid = True
            if torch.cuda.is_available():
                free_memory = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                )
                free_memory_gb = free_memory / (1024**3)

                if free_memory_gb < 2:
                    print(
                        f"Warning: Very low GPU memory ({free_memory_gb:.2f}GB free), disabling FID calculation"
                    )
                    enable_fid = False

            try:
                # Create metrics calculator
                self.metrics_calculator = MetricsCalculator(
                    self.device, enable_fid=enable_fid
                )

                # Ensure required methods are available
                required_methods = [
                    "calculate_metrics",
                    "calculate_fid",
                    "calculate_ncc",
                    "calculate_slice_based_fid",
                ]

                missing_methods = []
                for method in required_methods:
                    if not hasattr(self.metrics_calculator, method):
                        missing_methods.append(method)

                if missing_methods:
                    print(
                        f"Warning: MetricsCalculator missing methods: {missing_methods}"
                    )
                    # Try to add compatibility methods
                    for method in missing_methods:
                        if method == "calculate_slice_based_fid" and hasattr(
                            self.metrics_calculator, "calculate_fid"
                        ):
                            setattr(
                                self.metrics_calculator,
                                "calculate_slice_based_fid",
                                self.metrics_calculator.calculate_fid,
                            )

                # Verify FID calculator is available
                if not hasattr(self.metrics_calculator, "fid_available"):
                    setattr(self.metrics_calculator, "fid_available", enable_fid)

            except Exception as e:
                print(f"Error initializing metrics calculator: {e}")
                import traceback

                traceback.print_exc()

                # Create fallback metrics calculator without FID
                self.metrics_calculator = MetricsCalculator(
                    self.device, enable_fid=False
                )
                print("Created fallback metrics calculator without FID")

    def _has_invalid_values(self, tensor: torch.Tensor) -> bool:
        """
        Check if tensor has invalid values (NaN, Inf, all zeros, etc.).

        Args:
            tensor: Tensor to check

        Returns:
            Boolean indicating if tensor has invalid values
        """
        with torch.no_grad():
            if tensor is None:
                return True
            if torch.isnan(tensor).any():
                return True
            if torch.isinf(tensor).any():
                return True
            if torch.all(tensor == 0):
                return True
            if self.using_cpu and torch.abs(tensor).max() < 1e-6:
                return True
        return False

    def _downscale_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Downscale tensor if downscale factor is less than 1.0.

        Args:
            tensor: Tensor to downscale

        Returns:
            Downscaled tensor if needed, original tensor otherwise
        """
        if self.downscale_factor < 1.0:
            if tensor.dim() == 4:
                return F.interpolate(
                    tensor,
                    scale_factor=self.downscale_factor,
                    mode="bilinear",
                    align_corners=False,
                )
            elif tensor.dim() == 5:
                return F.interpolate(
                    tensor,
                    scale_factor=self.downscale_factor,
                    mode="trilinear",
                    align_corners=False,
                )
        return tensor

    def _collect_images_for_fid(self):
        """
        Collect image slices for FID calculation using consistent
        slice extraction along the smallest dimension (matching evaluation script).
        Extracts and stores 2D slices from 3D volumes for more accurate FID calculation.
        """
        if not self.enable_full_dataset_fid:
            return

        try:
            with torch.no_grad():

                def get_smallest_dimension_info(volume):
                    """Identify the smallest dimension for slice extraction"""
                    if volume.dim() != 5:
                        return None, None, None

                    B, C, D, H, W = volume.shape
                    dims = [(D, 2, "D"), (H, 3, "H"), (W, 4, "W")]
                    dims.sort(key=lambda x: x[0])  # Sort by size
                    return dims[0]

                # Extract all slices along the smallest dimension for more consistent FID
                for vol_key, vol_collector in [
                    ("real_B", self.real_B_collector),
                    ("fake_B", self.fake_B_collector),
                ]:
                    if not hasattr(self.model, vol_key) or self._has_invalid_values(
                        getattr(self.model, vol_key)
                    ):
                        continue

                    volume = getattr(self.model, vol_key)

                    if volume.dim() != 5:  # Skip non-3D volumes
                        continue

                    # Find the smallest dimension
                    smallest_dim_size, smallest_dim_idx, smallest_dim_name = (
                        get_smallest_dimension_info(volume)
                    )

                    if smallest_dim_size is None:
                        continue

                    # Extract all slices along the smallest dimension
                    for slice_idx in range(smallest_dim_size):
                        if smallest_dim_idx == 2:  # D dimension
                            slice_tensor = volume[
                                0:1, :, slice_idx : slice_idx + 1, :, :
                            ].squeeze(2)
                        elif smallest_dim_idx == 3:  # H dimension
                            slice_tensor = volume[
                                0:1, :, :, slice_idx : slice_idx + 1, :
                            ].squeeze(3)
                            slice_tensor = slice_tensor.transpose(
                                2, 3
                            )  # Consistent orientation
                        else:  # W dimension
                            slice_tensor = volume[
                                0:1, :, :, :, slice_idx : slice_idx + 1
                            ].squeeze(4)
                            slice_tensor = slice_tensor.transpose(
                                2, 3
                            )  # Consistent orientation

                        # Skip slices with invalid values
                        if (
                            torch.isnan(slice_tensor).any()
                            or torch.isinf(slice_tensor).any()
                        ):
                            continue

                        # Skip slices with almost no content (use same threshold as evaluation)
                        non_background = ((slice_tensor > -0.95).float().mean()).item()
                        if (
                            non_background < 0.001
                        ):  # Very loose threshold matching evaluation script
                            continue

                        # Add to collector
                        vol_collector.append(slice_tensor.detach().cpu())

                self.collected_samples += 1

                # Provide feedback on collection progress
                if self.collected_samples % 2 == 0:
                    min_slices_required = 50
                    if (
                        len(self.real_B_collector) < min_slices_required
                        or len(self.fake_B_collector) < min_slices_required
                    ):
                        remaining = min_slices_required - min(
                            len(self.real_B_collector), len(self.fake_B_collector)
                        )
                        print(
                            f"Collecting slices for evaluation-style FID: have {min(len(self.real_B_collector), len(self.fake_B_collector))}, need {remaining} more"
                        )

        except Exception as e:
            print(f"Error collecting images for FID: {e}")
            import traceback

            traceback.print_exc()

    def _preprocess_data_safely(self, data):
        """
        Safely preprocess input data for validation.

        Args:
            data: Input data to preprocess

        Returns:
            Preprocessed data
        """
        try:
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                processed_data = []
                for item in data:
                    if isinstance(item, torch.Tensor):
                        processed_data.append(item.clone())
                    else:
                        processed_data.append(item)
                return processed_data

            return data
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return data

    def validate_full_image(self, data):
        """
        Validate full image with sliding window approach.

        Handles large medical images by processing patches and then
        combining the results. Includes black region cropping and
        memory optimization for efficient processing.

        Args:
            data: Input data for validation

        Returns:
            Tuple of (losses, metrics, success_flag)
        """
        aggressive_memory_cleanup()

        try:
            processed_data = self._preprocess_data_safely(data)
            if processed_data is None:
                print("Failed to preprocess data")
                return {}, {}, False

            if not isinstance(processed_data, (list, tuple)) or len(processed_data) < 2:
                print("Invalid data format")
                return {}, {}, False

            image_A = processed_data[0]
            image_B = processed_data[1]

            if image_A.dim() < 5:
                print(f"Input dimension too low: {image_A.shape}")
                return {}, {}, False

            # Store original images before cropping for reference
            original_image_A = image_A.clone()
            original_image_B = image_B.clone()

            # Apply black region cropping with aligned dimensions
            print("Cropping black regions from validation images...")
            original_shapes = (image_A.shape, image_B.shape)

            image_A, image_B, crop_info = crop_black_regions_aligned(
                image_A, image_B, threshold=0.02
            )
            cropped_shapes = (image_A.shape, image_B.shape)
            print(f"Cropped A: {original_shapes[0]} → {cropped_shapes[0]}")
            print(f"Cropped B: {original_shapes[1]} → {cropped_shapes[1]}")

            min_dim_A = min(image_A.shape[2:])
            min_dim_B = min(image_B.shape[2:])

            if min_dim_A < self.min_patch_size[0] or min_dim_B < self.min_patch_size[0]:
                print(
                    f"Warning: Cropped image dimensions {image_A.shape[2:]} too small for minimum patch size {self.min_patch_size}"
                )
                scale_factor = (
                    max(
                        self.min_patch_size[0] / min_dim_A,
                        self.min_patch_size[1] / min_dim_B,
                    )
                    * 1.5
                )
                image_A = F.interpolate(
                    image_A,
                    scale_factor=scale_factor,
                    mode="trilinear",
                    align_corners=False,
                )
                image_B = F.interpolate(
                    image_B,
                    scale_factor=scale_factor,
                    mode="trilinear",
                    align_corners=False,
                )
                upscaled_min_dim = min(min(image_A.shape[2:]), min(image_B.shape[2:]))
                if upscaled_min_dim < 16:
                    print(
                        "Warning: Images still too small after upscaling, using traditional validation"
                    )
                    self.use_sliding_window = False
                    return self.validate_batch_regular(data)

            if self.downscale_factor < 1.0:
                image_A = self._downscale_if_needed(image_A)
                image_B = self._downscale_if_needed(image_B)

            # Generate patch indices
            patch_indices = self._generate_patch_indices(
                image_A.shape[2:],
                self.patch_size,
                self.stride_inplane,
                self.stride_layer,
            )
            print(f"Generated {len(patch_indices)} patches after black region cropping")

            # Get batch size for patch processing
            batch_size = (
                getattr(self.model.opt, "batch_size", 1)
                if hasattr(self.model, "opt")
                else 1
            )

            # Process patches in batches
            output_device = self.device
            model_results = self._process_patches_in_batches(
                image_A, image_B, patch_indices, batch_size, crop_info=crop_info
            )

            # Check if any valid patches were processed
            if (
                torch.sum(torch.abs(model_results["fake_A"])) < 1e-6
                or torch.sum(torch.abs(model_results["fake_B"])) < 1e-6
            ):
                print(
                    "Warning: No patches were processed successfully, falling back to regular validation"
                )
                self.use_sliding_window = False
                return self.validate_batch_regular(data)

            # Check if we have padded real_A and real_B in model_results
            if "real_A" not in model_results or "real_B" not in model_results:
                # If not, create padded versions manually
                if crop_info:
                    orig_D, orig_H, orig_W = crop_info["orig_shape"]
                    d_start = int(crop_info["d_start"])
                    d_end = int(crop_info["d_end"])
                    h_start = int(crop_info["h_start"])
                    h_end = int(crop_info["h_end"])
                    w_start = int(crop_info["w_start"])
                    w_end = int(crop_info["w_end"])

                    # Create padded original tensors
                    padded_real_A = (
                        torch.ones(
                            (
                                original_image_A.shape[0],
                                original_image_A.shape[1],
                                orig_D,
                                orig_H,
                                orig_W,
                            ),
                            device=original_image_A.device,
                        )
                        * -1.0
                    )
                    padded_real_A[:, :, d_start:d_end, h_start:h_end, w_start:w_end] = (
                        image_A
                    )

                    padded_real_B = (
                        torch.ones(
                            (
                                original_image_B.shape[0],
                                original_image_B.shape[1],
                                orig_D,
                                orig_H,
                                orig_W,
                            ),
                            device=original_image_B.device,
                        )
                        * -1.0
                    )
                    padded_real_B[:, :, d_start:d_end, h_start:h_end, w_start:w_end] = (
                        image_B
                    )

                    model_results["real_A"] = padded_real_A
                    model_results["real_B"] = padded_real_B

                    print(
                        f"Created padded real_A and real_B with dimensions {padded_real_A.shape[2:]}"
                    )
                else:
                    # If no crop info, just use the original tensors
                    model_results["real_A"] = original_image_A
                    model_results["real_B"] = original_image_B

            # Create images dictionary for metrics calculation
            images_dict = {
                "real_A": model_results["real_A"].to(output_device),
                "real_B": model_results["real_B"].to(output_device),
                "fake_A": model_results["fake_A"].to(output_device),
                "fake_B": model_results["fake_B"].to(output_device),
            }

            if "rec_A" in model_results:
                images_dict["rec_A"] = model_results["rec_A"].to(output_device)
            if "rec_B" in model_results:
                images_dict["rec_B"] = model_results["rec_B"].to(output_device)

            # Check that all images have the same dimensions
            shapes = {k: v.shape[2:] for k, v in images_dict.items()}
            print(f"Image shapes for visualization: {shapes}")

            # Plot validation images
            self.plot_full_reconstructed_images(images_dict, 0)

            # Set model attributes for FID calculation
            if self.enable_full_dataset_fid:
                self.model.real_A = images_dict["real_A"].to(self.device)
                self.model.real_B = images_dict["real_B"].to(self.device)
                self.model.fake_A = model_results["fake_A"].to(self.device)
                self.model.fake_B = model_results["fake_B"].to(self.device)
                self._collect_images_for_fid()

            # Calculate metrics
            self._initialize_metrics_calculator()
            metrics = calculate_metrics_with_batching(
                self.metrics_calculator, images_dict, self.device
            )
            losses = (
                self.model.get_current_losses()
                if hasattr(self.model, "get_current_losses")
                else {}
            )

            aggressive_memory_cleanup()
            return losses, metrics, True

        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM in sliding window validation: {e}")
            if self.downscale_factor > 0.1:
                self.downscale_factor *= 0.5
                print(
                    f"Warning: Reducing validation resolution to {self.downscale_factor * 100:.1f}%"
                )
                return {}, {}, False
            print("Warning: Cannot recover from OOM")
            aggressive_memory_cleanup()
            return {}, {}, False

        except Exception as e:
            print(f"Error in sliding window validation: {e}")
            import traceback

            traceback.print_exc()
            aggressive_memory_cleanup()
            return {}, {}, False

    def _process_patches_in_batches(
        self, image_A, image_B, patch_indices, batch_size=2, crop_info=None
    ):
        """
        Process patches in batches with improved memory efficiency.

        Extracts and processes patches from input volumes, combining the
        results into a complete output volume with proper weighting.

        Args:
            image_A: Input image A
            image_B: Input image B
            patch_indices: List of patch indices
            batch_size: Number of patches to process in each batch
            crop_info: Information about cropped regions

        Returns:
            Dictionary of processed results
        """
        from tqdm import tqdm
        import time
        import gc

        device = self.device
        num_patches = len(patch_indices)

        # Get cropped shapes but don't create full tensors yet
        cropped_shape_A = image_A.shape
        cropped_shape_B = image_B.shape

        # Reduce batch size for memory savings
        if torch.cuda.is_available():
            free_mem = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            )
            free_mem_gb = free_mem / (1024**3)

            # Adaptively set batch size based on available memory
            if free_mem_gb < 2.0:
                batch_size = 1
                print(
                    f"Low memory detected ({free_mem_gb:.2f}GB free), using batch_size=1"
                )
            elif free_mem_gb < 4.0:
                batch_size = min(batch_size, 2)
                print(
                    f"Limited memory ({free_mem_gb:.2f}GB free), using batch_size={batch_size}"
                )

        # Initialize output tensors - use 16-bit floating point if possible for memory savings
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Create accumulators as empty tensors initially
        # We'll convert to full tensors after processing
        B, C = image_A.shape[:2]
        patch_results = {
            "fake_A_patches": [],  # List of (patch_idx, tensor) tuples
            "fake_B_patches": [],
            "rec_A_patches": [],
            "rec_B_patches": [],
        }

        # Process patches in batches
        start_time = time.time()
        processed_patches = 0
        valid_patches = 0

        # Main processing loop
        for batch_start in tqdm(
            range(0, num_patches, batch_size), desc="Processing patch batches"
        ):
            # Clear cache at the start of each batch
            torch.cuda.empty_cache()
            gc.collect()

            batch_end = min(batch_start + batch_size, num_patches)
            batch_indices = patch_indices[batch_start:batch_end]

            # Create batch tensors
            batch_A = []
            batch_B = []
            batch_info = []  # Store patch coordinates for each batch item

            # Extract patches for the batch
            for patch_idx in batch_indices:
                istart, iend, jstart, jend, kstart, kend = patch_idx
                patch_shape = (iend - istart, jend - jstart, kend - kstart)

                # Skip invalid patches
                if min(patch_shape) < 8 or any(dim == 1 for dim in patch_shape):
                    continue

                # Extract patches
                patch_A = image_A[:, :, istart:iend, jstart:jend, kstart:kend]
                patch_B = image_B[:, :, istart:iend, jstart:jend, kstart:kend]

                batch_A.append(patch_A)
                batch_B.append(patch_B)
                batch_info.append(patch_idx)

            processed_patches += len(batch_indices)

            # Skip if no valid patches in this batch
            if not batch_A:
                continue

            # Combine into batch tensors
            try:
                batch_A_tensor = torch.cat(batch_A, dim=0)
                batch_B_tensor = torch.cat(batch_B, dim=0)

                # Clear individual patch tensors
                batch_A = None
                batch_B = None

                # Process the batch
                self.model.set_input([batch_A_tensor, batch_B_tensor])
                # Free input tensors
                batch_A_tensor = None
                batch_B_tensor = None
                torch.cuda.empty_cache()

                self.model.test()

                # Get outputs
                if not hasattr(self.model, "fake_A") or not hasattr(
                    self.model, "fake_B"
                ):
                    continue

                # Get references to model outputs
                fake_A_batch = self.model.fake_A
                fake_B_batch = self.model.fake_B
                rec_A_batch = getattr(self.model, "rec_A", None)
                rec_B_batch = getattr(self.model, "rec_B", None)

                # Skip if outputs have invalid values
                if self._has_invalid_values(fake_A_batch) or self._has_invalid_values(
                    fake_B_batch
                ):
                    # Explicitly clear model outputs
                    self.model.fake_A = None
                    self.model.fake_B = None
                    if rec_A_batch is not None:
                        self.model.rec_A = None
                    if rec_B_batch is not None:
                        self.model.rec_B = None
                    continue

                # Store results for each patch
                for i, patch_idx in enumerate(batch_info):
                    istart, iend, jstart, jend, kstart, kend = patch_idx

                    # Store the results along with their coordinates
                    fake_A_patch = fake_A_batch[i : i + 1].detach().clone()
                    fake_B_patch = fake_B_batch[i : i + 1].detach().clone()

                    patch_results["fake_A_patches"].append(
                        (patch_idx, fake_A_patch.cpu())
                    )
                    patch_results["fake_B_patches"].append(
                        (patch_idx, fake_B_patch.cpu())
                    )

                    if rec_A_batch is not None and not self._has_invalid_values(
                        rec_A_batch
                    ):
                        rec_A_patch = rec_A_batch[i : i + 1].detach().clone()
                        patch_results["rec_A_patches"].append(
                            (patch_idx, rec_A_patch.cpu())
                        )

                    if rec_B_batch is not None and not self._has_invalid_values(
                        rec_B_batch
                    ):
                        rec_B_patch = rec_B_batch[i : i + 1].detach().clone()
                        patch_results["rec_B_patches"].append(
                            (patch_idx, rec_B_patch.cpu())
                        )

                    valid_patches += 1

                # Explicitly clear model outputs
                self.model.fake_A = None
                self.model.fake_B = None
                if rec_A_batch is not None:
                    self.model.rec_A = None
                if rec_B_batch is not None:
                    self.model.rec_B = None

                # Clear local references
                fake_A_batch = None
                fake_B_batch = None
                rec_A_batch = None
                rec_B_batch = None

                # Aggressive memory cleanup
                aggressive_memory_cleanup()

            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                aggressive_memory_cleanup()
                continue

        # Calculate elapsed time and processing rate
        elapsed_time = time.time() - start_time
        patches_per_second = processed_patches / elapsed_time if elapsed_time > 0 else 0

        print(
            f"Processed {processed_patches} patches ({valid_patches} valid) in {elapsed_time:.2f}s ({patches_per_second:.2f} patches/sec)"
        )

        # Now create full output tensors and place patches
        fake_A = torch.ones_like(image_A, device=device) * -1.0
        fake_B = torch.ones_like(image_B, device=device) * -1.0
        weight_map_A = torch.zeros_like(image_A, device=device)
        weight_map_B = torch.zeros_like(image_B, device=device)

        # Only create rec tensors if we have valid patches
        has_rec_A = len(patch_results["rec_A_patches"]) > 0
        has_rec_B = len(patch_results["rec_B_patches"]) > 0

        rec_A = torch.ones_like(image_A, device=device) * -1.0 if has_rec_A else None
        rec_B = torch.ones_like(image_B, device=device) * -1.0 if has_rec_B else None
        weight_map_rec_A = (
            torch.zeros_like(image_A, device=device) if has_rec_A else None
        )
        weight_map_rec_B = (
            torch.zeros_like(image_B, device=device) if has_rec_B else None
        )

        # Add patches to output tensors (with batch efficiency)
        print("Compositing fake_A patches...")
        for patch_idx, patch_tensor in patch_results["fake_A_patches"]:
            istart, iend, jstart, jend, kstart, kend = patch_idx
            fake_A[:, :, istart:iend, jstart:jend, kstart:kend] += patch_tensor.to(
                device
            )
            weight_map_A[:, :, istart:iend, jstart:jend, kstart:kend] += 1.0

        print("Compositing fake_B patches...")
        for patch_idx, patch_tensor in patch_results["fake_B_patches"]:
            istart, iend, jstart, jend, kstart, kend = patch_idx
            fake_B[:, :, istart:iend, jstart:jend, kstart:kend] += patch_tensor.to(
                device
            )
            weight_map_B[:, :, istart:iend, jstart:jend, kstart:kend] += 1.0

        if has_rec_A:
            print("Compositing rec_A patches...")
            for patch_idx, patch_tensor in patch_results["rec_A_patches"]:
                istart, iend, jstart, jend, kstart, kend = patch_idx
                rec_A[:, :, istart:iend, jstart:jend, kstart:kend] += patch_tensor.to(
                    device
                )
                weight_map_rec_A[:, :, istart:iend, jstart:jend, kstart:kend] += 1.0

        if has_rec_B:
            print("Compositing rec_B patches...")
            for patch_idx, patch_tensor in patch_results["rec_B_patches"]:
                istart, iend, jstart, jend, kstart, kend = patch_idx
                rec_B[:, :, istart:iend, jstart:jend, kstart:kend] += patch_tensor.to(
                    device
                )
                weight_map_rec_B[:, :, istart:iend, jstart:jend, kstart:kend] += 1.0

        # Free memory from patch collections
        patch_results = None
        torch.cuda.empty_cache()
        gc.collect()

        # Normalize output tensors by weight maps
        epsilon = 1e-8
        weight_map_A = torch.where(
            weight_map_A > 0,
            weight_map_A,
            torch.tensor(epsilon, device=weight_map_A.device),
        )
        weight_map_B = torch.where(
            weight_map_B > 0,
            weight_map_B,
            torch.tensor(epsilon, device=weight_map_B.device),
        )

        fake_A = fake_A / weight_map_A
        fake_B = fake_B / weight_map_B

        # Free weight maps as soon as possible
        weight_map_A = None
        weight_map_B = None
        torch.cuda.empty_cache()

        if rec_A is not None and weight_map_rec_A is not None:
            weight_map_rec_A = torch.where(
                weight_map_rec_A > 0,
                weight_map_rec_A,
                torch.tensor(epsilon, device=weight_map_rec_A.device),
            )
            rec_A = rec_A / weight_map_rec_A
            weight_map_rec_A = None  # Free memory

        if rec_B is not None and weight_map_rec_B is not None:
            weight_map_rec_B = torch.where(
                weight_map_rec_B > 0,
                weight_map_rec_B,
                torch.tensor(epsilon, device=weight_map_rec_B.device),
            )
            rec_B = rec_B / weight_map_rec_B
            weight_map_rec_B = None  # Free memory

        # Padding results back to original dimensions if crop_info is provided
        results = {"fake_A": fake_A, "fake_B": fake_B}

        if rec_A is not None:
            results["rec_A"] = rec_A
        if rec_B is not None:
            results["rec_B"] = rec_B

        # Fix indentation issue - this block should not be part of the if statement
        if crop_info:
            print("Padding results back to original dimensions...")

            # Get original dimensions
            orig_D, orig_H, orig_W = crop_info["orig_shape"]

            # Get crop coordinates
            d_start = int(crop_info["d_start"])
            d_end = int(crop_info["d_end"])
            h_start = int(crop_info["h_start"])
            h_end = int(crop_info["h_end"])
            w_start = int(crop_info["w_start"])
            w_end = int(crop_info["w_end"])

            # Create padded tensors one at a time to save memory
            for key in list(
                results.keys()
            ):  # Use list() to avoid mutation during iteration
                tensor = results[key]
                B, C, D, H, W = tensor.shape

                # Create empty tensor of original size with black (-1) background
                padded = (
                    torch.ones((B, C, orig_D, orig_H, orig_W), device=tensor.device)
                    * -1.0
                )

                # Copy data from result to the correct position in padded tensor
                padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end] = tensor

                # Update the result and free the original tensor
                results[key] = padded
                tensor = None
                torch.cuda.empty_cache()

            # Also pad the original images (add to results) one at a time
            real_A_padded = (
                torch.ones(
                    (image_A.shape[0], image_A.shape[1], orig_D, orig_H, orig_W),
                    device=image_A.device,
                )
                * -1.0
            )
            real_A_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end] = image_A
            results["real_A"] = real_A_padded
            real_A_padded = None

            real_B_padded = (
                torch.ones(
                    (image_B.shape[0], image_B.shape[1], orig_D, orig_H, orig_W),
                    device=image_B.device,
                )
                * -1.0
            )
            real_B_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end] = image_B
            results["real_B"] = real_B_padded
            real_B_padded = None

            print(
                f"Padded tensors from {image_A.shape[2:]} back to {(orig_D, orig_H, orig_W)}"
            )

        # Clear any lingering references
        torch.cuda.empty_cache()
        gc.collect()

        return results

    def crop_black_regions(self, image, threshold=0.02):
        """
        Crop black regions from a tensor.

        Args:
            image: Input tensor to crop
            threshold: Threshold for considering a pixel as non-black

        Returns:
            Cropped tensor
        """
        # Just a wrapper around crop_black_regions_aligned for a single image
        if image.dim() != 5:
            print("Warning: Expected 5D tensor, returning original tensor")
            return image

        dummy = torch.ones_like(image)
        cropped, _, crop_info = crop_black_regions_aligned(image, dummy, threshold)
        return cropped

    def validate_batch_regular(self, data):
        """
        Regular validation for a single batch.

        Used when sliding window validation is not needed or not possible.

        Args:
            data: Input data for validation

        Returns:
            Tuple of (losses, metrics, success_flag)
        """
        try:
            aggressive_memory_cleanup()

            if data is None:
                print("Warning: Received None data in validate_batch")
                return {}, {}, False

            processed_data = self._preprocess_data_safely(data)
            if processed_data is None:
                print("Warning: Failed to preprocess data, skipping validation sample")
                return {}, {}, False

            # Apply black region cropping before validation
            if isinstance(processed_data, (list, tuple)) and len(processed_data) >= 2:
                if processed_data[0].dim() >= 4:  # Has spatial dimensions
                    print(
                        "Applying black region cropping to regular validation data..."
                    )
                    original_shapes = [item.shape for item in processed_data[:2]]

                    processed_data[0] = self.crop_black_regions(
                        processed_data[0], threshold=0.02
                    )
                    processed_data[1] = self.crop_black_regions(
                        processed_data[1], threshold=0.02
                    )

                    cropped_shapes = [item.shape for item in processed_data[:2]]
                    print(f"Cropped shapes: {original_shapes} → {cropped_shapes}")

            data = processed_data
            processed_data = None

            try:
                self.model.set_input(data)
            except Exception as e:
                print(f"Error in set_input: {e}")
                return {}, {}, False

            data = None
            aggressive_memory_cleanup()

            with torch.no_grad():
                try:
                    self.model.test()
                except Exception as e:
                    print(f"Error in model.test(): {e}")
                    return {}, {}, False

            required_tensors = ["real_A", "real_B", "fake_A", "fake_B"]
            for tensor_name in required_tensors:
                tensor = getattr(self.model, tensor_name, None)
                if tensor is None:
                    print(f"Error: {tensor_name} is None after forward pass")
                    return {}, {}, False
                if self._has_invalid_values(tensor):
                    print(f"Error: {tensor_name} has invalid values")
                    return {}, {}, False

            if self.enable_full_dataset_fid:
                self._collect_images_for_fid()

            with torch.no_grad():
                losses = self.model.get_current_losses()

            all_metrics = {}

            with torch.no_grad():
                images_dict = {}
                tensor_shapes = {}

                if hasattr(self.model, "real_A") and hasattr(self.model, "fake_A"):
                    images_dict["real_A"] = self.model.real_A.detach()
                    images_dict["fake_A"] = self.model.fake_A.detach()
                    tensor_shapes["real_A"] = self.model.real_A.shape
                    tensor_shapes["fake_A"] = self.model.fake_A.shape

                if hasattr(self.model, "real_B") and hasattr(self.model, "fake_B"):
                    images_dict["real_B"] = self.model.real_B.detach()
                    images_dict["fake_B"] = self.model.fake_B.detach()
                    tensor_shapes["real_B"] = self.model.real_B.shape

                print(f"Model tensor shapes: {tensor_shapes}")

                if len(images_dict) == 4:
                    self._initialize_metrics_calculator()
                    all_metrics = calculate_metrics_with_batching(
                        self.metrics_calculator, images_dict, self.device
                    )

                images_dict = None

            for attr_name in required_tensors:
                if hasattr(self.model, attr_name):
                    setattr(self.model, attr_name, None)

            aggressive_memory_cleanup()
            return losses, all_metrics, True

        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM in validation: {e}")
            if self.downscale_factor > 0.1:
                self.downscale_factor *= 0.5
                print(
                    f"Reducing validation resolution to {self.downscale_factor * 100:.1f}%"
                )
                aggressive_memory_cleanup()
                return self.validate_batch_regular(data)
            elif self.enable_full_metrics:
                self.enable_full_metrics = False
                print("Disabling full metrics calculation due to memory constraints")
                aggressive_memory_cleanup()
                return self.validate_batch_regular(data)
            else:
                aggressive_memory_cleanup()
                return {}, {}, False

        except Exception as e:
            print(f"Error in validation: {e}")
            aggressive_memory_cleanup()
            return {}, {}, False

    def validate_batch(self, data: Any) -> Tuple[Dict, Dict, bool]:
        """
        Top-level validation method that chooses the appropriate validation approach.

        Selects between sliding window validation and regular validation
        based on configuration and input data characteristics.

        Args:
            data: Input data for validation

        Returns:
            Tuple of (losses, metrics, success_flag)
        """
        # Check if evaluation-style metrics are enabled in options
        eval_style_enabled = False
        if hasattr(self.model, "opt") and hasattr(
            self.model.opt, "run_eval_style_metrics"
        ):
            eval_style_enabled = self.model.opt.run_eval_style_metrics

        # For regular validation (still needed for loss calculation and model updates)
        result = (
            self.validate_full_image(data)
            if self.use_sliding_window
            else self.validate_batch_regular(data)
        )

        # Extract losses, metrics, and success flag
        losses, metrics, success = result

        # If validation was successful, ensure the FID collector is used
        if success and eval_style_enabled:
            # Make sure to collect slices for FID even with regular validation
            self._collect_images_for_fid()

            # Check if we should report the collected FID
            # In a real implementation, you might check epoch count or steps
            # For now, we'll check if we have collected enough slices
            min_slices_required = 50
            if (
                len(self.real_B_collector) >= min_slices_required
                and len(self.fake_B_collector) >= min_slices_required
            ):

                # Calculate and report evaluation-style FID
                fid_metrics = self.calculate_full_dataset_fid()

                # Update metrics with evaluation-style FID
                if "fid_domain" in fid_metrics:
                    # Add a prefix to distinguish from regular metrics
                    metrics["eval_fid_domain"] = fid_metrics["fid_domain"]
                    if "fid_domain_smoothed" in fid_metrics:
                        metrics["eval_fid_domain_smoothed"] = fid_metrics[
                            "fid_domain_smoothed"
                        ]

                    # Log to console for visibility
                    print(
                        f"\n=== Evaluation-Style FID: {fid_metrics['fid_domain']:.4f} ===\n"
                    )

        return losses, metrics, success

    def setup_full_validation_plotting(
        self, max_plots=8, epoch=None, run_name=None, dataset_prefix=None
    ):
        """
        Setup for full validation image plotting.

        Args:
            max_plots: Maximum number of plots to generate
            epoch: Current epoch number
            run_name: Name of the training run
            dataset_prefix: Prefix for dataset naming
        """
        self.plot_full_validation = True
        self.max_full_plots = max_plots  # Increased to 8
        self.full_images_plotted = 0
        self.current_epoch = epoch
        self.run_name = run_name
        self.dataset_prefix = dataset_prefix
        print(f"Full validation plotting enabled (max {max_plots} plots)")

    def plot_full_reconstructed_images(self, images_dict, sample_idx):
        """
        Plot full reconstructed images for visualization.

        Args:
            images_dict: Dictionary of images to plot
            sample_idx: Index of the current sample
        """
        if not hasattr(self, "plot_full_validation") or not self.plot_full_validation:
            return

        if self.full_images_plotted >= self.max_full_plots:
            return

        if not hasattr(self, "current_epoch") or not hasattr(self, "run_name"):
            print("Warning: Missing epoch or run_name for full validation plotting")
            return

        try:
            dataset_name = None
            if hasattr(self, "dataset_prefix") and self.dataset_prefix:
                dataset_name = f"{self.dataset_prefix}_{sample_idx}"

            plot_full_validation_images(
                images_dict,
                self.current_epoch,
                self.run_name,
                self.full_images_plotted,
                dataset_name,
            )

            self.full_images_plotted += 1

        except Exception as e:
            print(f"Warning: Failed to plot full validation image: {e}")
            import traceback

            traceback.print_exc()

    def calculate_full_dataset_fid(self):
        """
        Calculate FID score for the full dataset.

        Uses the collected real and fake image slices to calculate FID,
        matching the approach used in the evaluation script.

        Returns:
            Dictionary with FID metrics
        """
        self._initialize_metrics_calculator()
        return calculate_full_dataset_fid(
            self.metrics_calculator,
            self.real_B_collector,
            self.fake_B_collector,
            self.device,
        )

    def _collect_train_exvivo_images(self):
        """
        Collect all exvivo images from training set for FID calculation.

        Loads and processes ex-vivo training data for use in evaluation metrics.
        """
        if not hasattr(self, "train_exvivo_collector"):
            self.train_exvivo_collector = []

        if len(self.train_exvivo_collector) > 50:
            print(
                f"Already collected {len(self.train_exvivo_collector)} training exvivo images"
            )
            return

        try:
            import os
            from utils.utils import lstFiles

            # Get path to training exvivo data
            if hasattr(self.model, "opt") and hasattr(self.model.opt, "data_path"):
                train_path = self.model.opt.data_path
            else:
                print(
                    "Cannot determine training data path, skipping train exvivo collection"
                )
                return

            train_exvivo_path = os.path.join(train_path, "exvivo")

            if not os.path.exists(train_exvivo_path):
                print(f"Training exvivo path does not exist: {train_exvivo_path}")
                return

            import SimpleITK as sitk
            import torch
            import torch.nn.functional as F

            exvivo_files = lstFiles(train_exvivo_path)

            if not exvivo_files:
                print("No exvivo files found in training set")
                return

            # Use all available files instead of sampling
            collection_factor = 1

            for i, file_path in enumerate(exvivo_files):
                print(f"Processing training exvivo file {i + 1}/{len(exvivo_files)}")

                try:
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(file_path)
                    reader.SetLoadPrivateTags(False)
                    image = reader.Execute()

                    # Convert to tensor
                    image_array = sitk.GetArrayFromImage(image)
                    image_tensor = torch.from_numpy(image_array).float()

                    # Add channel dimension if needed
                    if image_tensor.dim() == 3:
                        image_tensor = image_tensor.unsqueeze(0)

                    # Add batch dimension
                    image_tensor = image_tensor.unsqueeze(0)

                    # Normalize to [-1, 1] range
                    image_min = image_tensor.min()
                    image_max = image_tensor.max()
                    if image_max - image_min > 1e-5:
                        image_tensor = (image_tensor - image_min) / (
                            image_max - image_min
                        ) * 2 - 1
                    else:
                        image_tensor = torch.zeros_like(image_tensor)

                    # Resize for memory efficiency
                    if image_tensor.dim() == 5:  # 3D volume
                        D, H, W = image_tensor.shape[2:]
                        target_d = max(16, int(D * collection_factor))
                        target_h = max(64, int(H * collection_factor))
                        target_w = max(64, int(W * collection_factor))

                        image_tensor = F.interpolate(
                            image_tensor,
                            size=(target_d, target_h, target_w),
                            mode="trilinear",
                            align_corners=False,
                        )

                        # Extract all slices rather than keeping the volume
                        # This gives us more real samples for the FID calculation
                        for d in range(target_d):
                            slice_tensor = image_tensor[:, :, d : d + 1, :, :].squeeze(
                                2
                            )
                            self.train_exvivo_collector.append(slice_tensor.cpu())

                    else:  # 2D image
                        H, W = image_tensor.shape[2:]
                        target_h = max(64, int(H * collection_factor))
                        target_w = max(64, int(W * collection_factor))

                        image_tensor = F.interpolate(
                            image_tensor,
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        )

                        # Add 2D image directly
                        self.train_exvivo_collector.append(image_tensor.cpu())

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue

        except Exception as e:
            print(f"Error collecting training exvivo images: {e}")
            import traceback

            traceback.print_exc()
