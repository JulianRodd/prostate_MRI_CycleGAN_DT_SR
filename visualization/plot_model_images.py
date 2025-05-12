import gc
import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


class CycleGANVisualizer:
    class VolumeNavigator:
        """
        Interactive visualizer for navigating through volume slices.

        Allows scrolling through the slices of multiple 3D volumes side by side.
        """

        def __init__(self, fig, axes, volumes, titles):
            self.axes = axes
            self.volumes = volumes
            self.titles = titles
            _, _, self.slices = volumes[0].shape
            self.current_slice = self.slices // 2
            self.imshow_objects = []

            for ax, vol, title in zip(self.axes, volumes, titles):
                ax.set_title(title)
                vol_rotated = np.rot90(vol, k=-1)
                im = ax.imshow(vol_rotated[:, :, self.current_slice], cmap="gray")
                self.imshow_objects.append(im)
                ax.set_ylabel(f"Slice {self.current_slice}/{self.slices-1}")

            fig.suptitle("Use scroll wheel to navigate through slices", y=1.02)
            self.update()

        def onscroll(self, event):
            if event.button == "up":
                self.current_slice = min(self.slices - 1, self.current_slice + 1)
            else:
                self.current_slice = max(0, self.current_slice - 1)
            self.update()

        def update(self):
            for im, vol in zip(self.imshow_objects, self.volumes):
                vol_rotated = np.rot90(vol, k=-1)
                im.set_data(vol_rotated[:, :, self.current_slice])

            for ax in self.axes:
                ax.set_ylabel(f"Slice {self.current_slice}/{self.slices-1}")

            self.axes[0].figure.canvas.draw()


def plot_full_validation_images(
    images_dict, epoch, run_name, image_idx, dataset_name=None
):
    """
    Plot validation images with consistent dimensions.

    This function ensures all images have the same dimensions by checking
    the dimensions of all tensors and resizing smaller ones to match.

    Args:
        images_dict (dict): Dictionary containing tensors for various image types
        epoch (int): Current epoch number
        run_name (str): Name of the current run
        image_idx (int): Index of the image in the validation set
        dataset_name (str, optional): Name of the dataset
    """
    output_dir = os.path.join("validation_images", run_name, f"epoch_{epoch}")
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name:
        filename = f"{dataset_name}_full_validation_{image_idx}.png"
    else:
        filename = f"full_validation_{image_idx}.png"

    output_path = os.path.join(output_dir, filename)

    plot_keys = [
        ("real_A", "Real A (in-vivo)"),
        ("fake_B", "Fake B (ex-vivo)"),
        ("rec_A", "Cycle A (rec)"),
        ("real_B", "Real B (ex-vivo)"),
        ("fake_A", "Fake A (in-vivo)"),
        ("rec_B", "Cycle B (rec)"),
    ]

    available_plots = [
        (k, t) for k, t in plot_keys if k in images_dict and images_dict[k] is not None
    ]

    if not available_plots:
        print(f"No valid images to plot for {filename}")
        return

    max_dims = None
    for key, _ in available_plots:
        tensor = images_dict[key]
        if tensor is None:
            continue

        if tensor.dim() == 5:
            dims = tensor.shape[2:]
        elif tensor.dim() == 4:
            dims = tensor.shape[2:] + (1,)
        else:
            continue

        if max_dims is None:
            max_dims = dims
        else:
            max_dims = tuple(max(a, b) for a, b in zip(max_dims, dims))

    print(f"Maximum image dimensions for plot: {max_dims}")

    consistent_images = {}
    for key, _ in available_plots:
        tensor = images_dict[key]
        if tensor is None:
            continue

        if tensor.dim() == 5:
            D, H, W = tensor.shape[2:]
            if (D, H, W) != max_dims:
                print(f"Padding {key} from {(D, H, W)} to {max_dims}")

                padded = (
                    torch.ones(
                        (
                            tensor.shape[0],
                            tensor.shape[1],
                            max_dims[0],
                            max_dims[1],
                            max_dims[2],
                        ),
                        device=tensor.device,
                    )
                    * -1.0
                )

                padded[:, :, :D, :H, :W] = tensor
                consistent_images[key] = padded
            else:
                consistent_images[key] = tensor
        elif tensor.dim() == 4:
            H, W = tensor.shape[2:]
            if (H, W, 1) != max_dims:
                print(f"Padding 2D {key} from {(H, W)} to {max_dims[:2]}")

                padded = (
                    torch.ones(
                        (tensor.shape[0], tensor.shape[1], max_dims[0], max_dims[1]),
                        device=tensor.device,
                    )
                    * -1.0
                )

                padded[:, :, :H, :W] = tensor
                consistent_images[key] = padded
            else:
                consistent_images[key] = tensor

    try:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.flatten()

        slices = []
        titles = []

        def extract_axial_slice(tensor):
            with torch.no_grad():
                tensor_cpu = tensor.detach().cpu()
                vol = tensor_cpu.numpy()

                tensor_cpu = None

                if len(vol.shape) == 5:
                    middle_z = vol.shape[4] // 2
                    slice_2d = vol[0, 0, :, :, middle_z]
                    print(
                        f"Extracted axial slice at z={middle_z}, shape={slice_2d.shape}"
                    )
                elif len(vol.shape) == 4:
                    slice_2d = vol[0, 0, :, :]
                else:
                    print(f"Unexpected tensor shape: {vol.shape}")
                    return None

                min_val = np.min(slice_2d)
                max_val = np.max(slice_2d)
                mean_val = np.mean(slice_2d)
                print(
                    f"Slice value range: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}"
                )

                foreground_mask = slice_2d > -0.95

                if foreground_mask.sum() > 0:
                    foreground_values = slice_2d[foreground_mask]

                    p_low, p_high = np.percentile(foreground_values, [1, 99.5])

                    if p_high - p_low < 0.1:
                        p_low = min(p_low, np.median(foreground_values) - 0.1)
                        p_high = max(p_high, np.median(foreground_values) + 0.1)
                        print(
                            f"Adjusting percentile range to prevent white-out: [{p_low:.4f}, {p_high:.4f}]"
                        )

                    slice_2d_clipped = np.copy(slice_2d)
                    slice_2d_clipped[foreground_mask] = np.clip(
                        foreground_values, p_low, p_high
                    )

                    slice_2d_norm = np.zeros_like(slice_2d_clipped)
                    if p_high > p_low:
                        slice_2d_norm[foreground_mask] = (
                            slice_2d_clipped[foreground_mask] - p_low
                        ) / (p_high - p_low)

                    slice_2d_norm[~foreground_mask] = 0.0

                    if np.mean(slice_2d_norm) > 0.9:
                        print(
                            "WARNING: Normalized image appears too bright, rescaling..."
                        )
                        non_zero_mask = slice_2d_norm > 0
                        if np.any(non_zero_mask):
                            slice_2d_norm[non_zero_mask] *= 0.8

                    return slice_2d_norm
                else:
                    print(
                        "Warning: No foreground pixels detected, using simple normalization"
                    )

                slice_2d = (slice_2d + 1) / 2.0
                return slice_2d

        for i, (key, title) in enumerate(available_plots[:6]):
            try:
                tensor_to_plot = consistent_images.get(key, images_dict[key])

                slice_2d = extract_axial_slice(tensor_to_plot)
                if slice_2d is not None:
                    slices.append(slice_2d)
                    titles.append(title)

                    ax = axes[i]
                    im = ax.imshow(slice_2d, cmap="gray", aspect="equal")
                    ax.set_title(title, fontsize=12)
                    ax.axis("off")

                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                    print(
                        f"Plotted axial slice for {title} with shape {slice_2d.shape}"
                    )
            except Exception as e:
                print(f"Error processing {key}: {e}")

        for i in range(len(slices), len(axes)):
            axes[i].axis("off")

        fig.suptitle(f"Validation Results - Epoch {epoch} - Axial View", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        print(
            f"Saved validation image with {len(slices)} axial slices to {output_path}"
        )

        slices = None
        gc.collect()

    except Exception as e:
        print(f"Error creating validation image: {e}")
        import traceback

        traceback.print_exc()


def plot_model_images(model, data, epoch=0, run_name=""):
    """
    Create visualizations of model inputs and outputs.

    Args:
        model: CycleGAN model
        data: Input data (typically a batch of images)
        epoch (int): Current epoch number
        run_name (str): Name of the current run
    """
    print(f"plot_model_images called with epoch={epoch}, run_name={run_name}")

    was_training = getattr(model, "isTrain", True)
    model.eval()

    image_folder = os.path.join("model_images", run_name)
    nifti_folder = os.path.join("model_nifti", run_name, f"epoch_{epoch}")

    try:
        os.makedirs(image_folder, exist_ok=True)
        print(f"Image folder created/verified: {image_folder}")
        os.makedirs(nifti_folder, exist_ok=True)
        print(f"NIfTI folder created/verified: {nifti_folder}")
    except Exception as e:
        print(f"Error creating output directories: {e}")

    original_data = data

    with torch.no_grad():
        try:
            torch.cuda.empty_cache()
            gc.collect()

            if not isinstance(data, (list, tuple)) or len(data) < 2:
                print(
                    f"Warning: Invalid data format for visualization, data: {type(data)}"
                )
                return

            for i, item in enumerate(data):
                if not isinstance(item, torch.Tensor):
                    print(f"Warning: Data item {i} is not a tensor, it's {type(item)}")
                    return

                print(f"Data tensor {i} has shape {item.shape}")

                if item.dim() < 3 or any(dim <= 3 for dim in item.shape[2:]):
                    print(
                        f"Warning: Data item {i} has dimensions {item.shape} which are too small"
                    )
                    return

            if torch.cuda.is_available():
                free_memory = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                )
                free_memory_gb = free_memory / (1024**3)
                print(f"Available GPU memory: {free_memory_gb:.2f}GB")
                if free_memory_gb < 2:
                    print(
                        f"Only {free_memory_gb:.2f}GB memory available, using reduced resolution for visualization"
                    )

            print("Setting model input data...")
            try:
                model.set_input(data)
                print("Running model.test()...")
                model.test()
                print("Model.test() completed successfully")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"CUDA OOM during model.test(): {e}")
                    print("Trying with reduced resolution...")

                    if isinstance(data, (list, tuple)) and len(data) >= 2:
                        reduced_scale = 0.3
                        reduced_data = []

                        for i, item in enumerate(data):
                            if item.dim() == 5:
                                downsampled = F.interpolate(
                                    item,
                                    scale_factor=reduced_scale,
                                    mode="trilinear",
                                    align_corners=True,
                                )
                            else:
                                downsampled = F.interpolate(
                                    item,
                                    scale_factor=reduced_scale,
                                    mode="bilinear",
                                    align_corners=True,
                                )
                            reduced_data.append(downsampled)

                        try:
                            model.set_input(reduced_data)
                            model.test()
                            print("Model.test() successful with reduced resolution")
                        except Exception as e2:
                            print(f"Error with reduced resolution: {e2}")
                            return
                    else:
                        return
                else:
                    raise e
            except Exception as e:
                print(f"Error running model.test(): {e}")
                return

            outputs = [
                "real_A",
                "fake_B",
                "rec_A",
                "real_B",
                "fake_A",
                "rec_B",
                "idt_A",
                "idt_B",
            ]
            missing_outputs = []

            for output in outputs:
                if not hasattr(model, output) or getattr(model, output) is None:
                    missing_outputs.append(output)

            if missing_outputs:
                print(f"Warning: Missing outputs: {missing_outputs}")

            print("Preparing visualization data...")
            volumes = []

            def process_tensor(tensor_name):
                try:
                    if (
                        hasattr(model, tensor_name)
                        and getattr(model, tensor_name) is not None
                    ):
                        tensor = getattr(model, tensor_name)
                        return prepare_volume_cpu(tensor)
                    else:
                        print(f"{tensor_name} not available, using zeros")
                        if "A" in tensor_name and hasattr(model, "real_A"):
                            shape = prepare_volume_cpu(model.real_A).shape
                        elif "B" in tensor_name and hasattr(model, "real_B"):
                            shape = prepare_volume_cpu(model.real_B).shape
                        else:
                            shape = (10, 10, 10)
                        return np.zeros(shape)
                except Exception as e:
                    print(f"Error processing {tensor_name}: {e}")
                    return np.zeros((10, 10, 10))

            volumes.append(process_tensor("real_A"))
            volumes.append(process_tensor("fake_B"))
            volumes.append(process_tensor("rec_A"))
            volumes.append(process_tensor("idt_B"))

            volumes.append(process_tensor("real_B"))
            volumes.append(process_tensor("fake_A"))
            volumes.append(process_tensor("rec_B"))
            volumes.append(process_tensor("idt_A"))

            print(f"Created {len(volumes)} visualization volumes")

            for i, vol in enumerate(volumes):
                print(f"Volume {i} shape: {vol.shape}")

            titles = [
                "Real A (in-vivo)",
                "Fake B (ex-vivo)",
                "Cycle A (rec)",
                "Idt B (A→B→A)",
                "Real B (ex-vivo)",
                "Fake A (in-vivo)",
                "Cycle B (rec)",
                "Idt A (B→A→B)",
            ]

            print("Creating visualization figure...")
            try:
                fig, axes = plt.subplots(2, 4, figsize=(24, 12), dpi=150)
                axes = axes.flatten()

                if volumes[0].ndim == 3:
                    max_depth = max(vol.shape[2] for vol in volumes if vol.ndim == 3)

                    middle_slices = []
                    for vol in volumes:
                        if vol.ndim == 3:
                            middle_slice = vol.shape[2] // 2
                            middle_slices.append(middle_slice)
                        else:
                            middle_slices.append(0)

                    from collections import Counter

                    most_common_slice = Counter(middle_slices).most_common(1)[0][0]

                    for i, (vol, title, ax) in enumerate(zip(volumes, titles, axes)):
                        if vol.ndim == 3 and vol.shape[2] > most_common_slice:
                            ax.imshow(
                                np.rot90(vol[:, :, most_common_slice]), cmap="gray"
                            )
                        elif vol.ndim == 3:
                            ax.imshow(np.rot90(vol[:, :, -1]), cmap="gray")
                        else:
                            ax.imshow(np.rot90(vol), cmap="gray")

                        ax.set_title(f"{title} (Slice {most_common_slice})")
                        ax.axis("off")

                else:
                    for i, (vol, title, ax) in enumerate(zip(volumes, titles, axes)):
                        ax.imshow(np.rot90(vol), cmap="gray")
                        ax.set_title(title)
                        ax.axis("off")

                plt.tight_layout()

                output_path = os.path.join(image_folder, f"epoch_{epoch}.png")
                print(f"Saving plot to: {output_path}")
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Successfully saved visualization to {output_path}")

            except Exception as e:
                print(f"Error creating or saving figure: {e}")
                import traceback

                traceback.print_exc()

        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback

            traceback.print_exc()

            for name in [
                "real_A",
                "real_B",
                "fake_A",
                "fake_B",
                "rec_A",
                "rec_B",
                "idt_A",
                "idt_B",
            ]:
                if hasattr(model, name):
                    setattr(model, name, None)
            torch.cuda.empty_cache()
            gc.collect()

    if was_training:
        model.train()
    else:
        model.eval()

    torch.cuda.empty_cache()
    gc.collect()


def prepare_volume_cpu(tensor):
    """
    Prepare a volume tensor for visualization using robust normalization.

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        np.ndarray: Processed volume ready for visualization
    """
    if tensor is None:
        return np.zeros((10, 10, 10))

    with torch.no_grad():
        tensor_cpu = tensor.detach().cpu()
        vol = tensor_cpu.numpy()
        tensor_cpu = None

        if len(vol.shape) == 5:
            vol = vol[0, 0]
        elif len(vol.shape) == 4:
            vol = vol[0, 0]

        foreground_mask = vol > -0.95

        if foreground_mask.sum() > 0:
            foreground_values = vol[foreground_mask]

            p_low, p_high = np.percentile(foreground_values, [1, 99.5])

            vol_normalized = np.copy(vol)

            vol_normalized[foreground_mask] = np.clip(
                vol[foreground_mask], p_low, p_high
            )

            if p_high > p_low:
                vol_normalized[foreground_mask] = (
                    vol_normalized[foreground_mask] - p_low
                ) / (p_high - p_low)

            vol_normalized[~foreground_mask] = 0.0

            if np.mean(vol_normalized) > 0.9:
                print(
                    "WARNING: Normalized volume appears too bright, reducing intensity"
                )
                non_zero_mask = vol_normalized > 0
                if np.any(non_zero_mask):
                    vol_normalized[non_zero_mask] *= 0.8

            return vol_normalized

        else:
            if np.max(vol) - np.min(vol) > 1e-6:
                vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
            else:
                vol = np.zeros_like(vol)

        return vol


def save_as_nifti(tensor, reference_data, output_path):
    """
    Save a tensor as a NIfTI file, preserving metadata from reference if available.

    Args:
        tensor (torch.Tensor): Tensor to save
        reference_data: Reference data with metadata (optional)
        output_path (str): Path to save the NIfTI file
    """
    if tensor is None:
        print(f"Warning: Cannot save {output_path} - tensor is None")
        return

    try:
        with torch.no_grad():
            array = tensor.detach().cpu().numpy()
            tensor = None
            torch.cuda.empty_cache()

            if len(array.shape) == 5:
                array = array[0, 0]
            elif len(array.shape) == 4:
                array = array[0]

            if array.min() < 0 or array.max() > 1:
                array = (array + 1) / 2.0

            image = sitk.GetImageFromArray(array)
            array = None

            if hasattr(reference_data, "GetSpacing"):
                image.SetSpacing(reference_data.GetSpacing())
                image.SetOrigin(reference_data.GetOrigin())
                image.SetDirection(reference_data.GetDirection())

            sitk.WriteImage(image, output_path, useCompression=True)
            print(f"Saved NIfTI file: {output_path}")

            image = None
            gc.collect()

    except Exception as e:
        print(f"Error saving NIfTI file: {str(e)}")
        gc.collect()
