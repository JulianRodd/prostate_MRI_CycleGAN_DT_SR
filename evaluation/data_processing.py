import torch
import os
import gc
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Import from other modules
from visualization import get_smallest_dimension_info

# Global cache for training data
# This will be properly defined in config.py, but we reference it here
from config import CACHED_TRAIN_EXVIVO_DATA


def extract_slices_from_volumes(volumes, device, min_content_percentage=0.001):
    """
    Extract 2D slices from a list of 3D volumes in a memory-efficient way.
    Always uses the smallest dimension for consistent slice extraction.
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


def combine_real_datasets_memory_efficient(val_real, train_real, device):
    """
    Combine validation and training real data in a memory-efficient way.
    When data dimensions don't match, we use batch processing rather than
    trying to manipulate the entire tensor at once.
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


def ensure_compatible_dimensions(tensor_a, tensor_b, device):
    """
    Ensure that two tensors have compatible dimensions for concatenation
    by adding or removing dimensions as needed.
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
