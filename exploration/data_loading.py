import numpy as np
import nibabel as nib
import SimpleITK as sitk
import traceback


def print_debug_info(directory):
    """Print debug information about the dataset structure."""
    import os

    print(f"\nDebug: Exploring directory structure of {directory}")
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for f in files[:5]:
            print(f"{sub_indent}{f}")
        if len(files) > 5:
            print(f"{sub_indent}... and {len(files) - 5} more files")


def load_scan(filepath):
    """Load and return a scan and its data."""
    try:
        print(f"Loading file: {filepath}")

        if filepath.lower().endswith(".mha"):
            sitk_img = sitk.ReadImage(filepath)
            data = sitk.GetArrayFromImage(sitk_img)
            data = np.transpose(data, (2, 1, 0))

            class MhaWrapper:
                def __init__(self, sitk_img):
                    self.sitk_img = sitk_img
                    self.header = type(
                        "obj", (object,), {"get_zooms": lambda: sitk_img.GetSpacing()}
                    )

            img = MhaWrapper(sitk_img)
        else:
            img = nib.load(filepath)
            data = img.get_fdata()

        data = np.nan_to_num(data)
        print(f"  Loaded data shape: {data.shape}")
        return img, data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        traceback.print_exc()
        return None, None


def preprocess_scan(data, mask=None, normalize=True, clahe=False):
    """Preprocess scan data with optional masking, normalization, and CLAHE."""
    if data is None:
        return None

    if mask is not None:
        data = data * (mask > 0)

    if np.sum(data > 0) == 0:
        return data

    if normalize:
        data_min = np.min(data[data > 0])
        data_max = np.max(data[data > 0])
        if data_max > data_min:
            data_norm = np.zeros_like(data)
            data_norm[data > 0] = (data[data > 0] - data_min) / (data_max - data_min)
            data = data_norm

    if clahe:
        try:
            from skimage import exposure

            middle_slice_idx = data.shape[2] // 2
            middle_slice = data[:, :, middle_slice_idx].copy()

            if np.sum(middle_slice > 0) > 0:
                slice_min = np.min(middle_slice[middle_slice > 0])
                slice_max = np.max(middle_slice[middle_slice > 0])
                middle_slice_uint8 = np.zeros_like(middle_slice, dtype=np.uint8)
                if slice_max > slice_min:
                    middle_slice_uint8[middle_slice > 0] = (
                        (middle_slice[middle_slice > 0] - slice_min)
                        / (slice_max - slice_min)
                        * 255
                    ).astype(np.uint8)

                middle_slice_clahe = exposure.equalize_adapthist(
                    middle_slice_uint8, kernel_size=8, clip_limit=0.02
                )

                middle_slice_float = np.zeros_like(middle_slice)
                middle_slice_float[middle_slice > 0] = (
                    middle_slice_clahe[middle_slice > 0] * (slice_max - slice_min)
                    + slice_min
                )

                data[:, :, middle_slice_idx] = middle_slice_float
        except Exception as e:
            print(f"CLAHE preprocessing error: {e}")

    return data
