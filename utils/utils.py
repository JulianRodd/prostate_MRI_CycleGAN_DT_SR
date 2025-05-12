import gc
import glob
import logging
import math
import os
import re
from collections import OrderedDict
from typing import Tuple

import SimpleITK as sitk
import numpy as np
import torch
import wandb


def resample_to_physical_spacing(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolator: int = sitk.sitkBSpline,
) -> sitk.Image:
    """
    Resample an image to a target physical spacing.

    Args:
        image (sitk.Image): The input image to resample
        target_spacing (Tuple[float, float, float]): Target voxel spacing (x, y, z)
        interpolator (int): SimpleITK interpolator to use

    Returns:
        sitk.Image: Resampled image with the target spacing
    """
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    physical_size = original_size * original_spacing
    new_spacing = np.array(target_spacing)
    new_size = physical_size / new_spacing
    new_size = np.ceil(new_size).astype(int)

    reference = sitk.Image(new_size.tolist(), image.GetPixelID())
    reference.SetSpacing(target_spacing)
    reference.SetDirection(image.GetDirection())
    reference.SetOrigin(image.GetOrigin())

    resampled = sitk.Resample(
        image, reference, sitk.Transform(), interpolator, 0.0, image.GetPixelID()
    )

    return resampled


def mkdirs(paths):
    """
    Create directories for the given paths.

    Args:
        paths: A list of paths or a single path to create
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): Path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def check_dir(path):
    """
    Check if a directory exists and create it if it doesn't.

    Args:
        path (str): Path to check and potentially create
    """
    if not os.path.exists(path):
        os.mkdir(path)


def debug_image_stats(image: sitk.Image, name: str, stage: str = "") -> None:
    """
    Print debug statistics for an image.

    Args:
        image (sitk.Image): The image to analyze
        name (str): Name identifier for the image
        stage (str, optional): Processing stage identifier
    """
    try:
        print(f"\n=== {stage} {name} Stats ===")
        print(f"Size: {image.GetSize()}")
        print(f"Spacing: {image.GetSpacing()}")
        print(f"Origin: {image.GetOrigin()}")
        print(f"Direction: {image.GetDirection()}")
        print(f"Pixel Type: {image.GetPixelID()}")

        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        print(f"Min/Max: {stats.GetMinimum():.2f}/{stats.GetMaximum():.2f}")
        print(f"Mean/Std: {stats.GetMean():.2f}/{stats.GetVariance() ** .5:.2f}")
        print("=" * 40)
    except Exception as e:
        print(f"Failed to get stats for {name}: {str(e)}")


def new_state_dict(file_name):
    """
    Create a new state dict from a checkpoint file, removing 'module.' prefix if present.

    Args:
        file_name (str): Path to the checkpoint file

    Returns:
        OrderedDict: Processed state dictionary
    """
    state_dict = torch.load(file_name)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == "module":
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def numericalSort(value):
    """
    Sort strings that contain numbers naturally.

    Args:
        value (str): String to be sorted

    Returns:
        list: Parts of the string with numbers converted to integers
    """
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def lstFiles(Path):
    """
    List all medical image files in a directory.

    Args:
        Path (str): Directory path to search

    Returns:
        list: Sorted list of medical image file paths
    """
    images_list = []
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii.gz" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".mhd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".mha" in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)
    return images_list


def create_list(data_path):
    """
    Create source and target data lists from a data path.

    Args:
        data_path (str): Path to the data directory

    Returns:
        tuple: Lists of source and target data dictionaries
    """
    data_list = glob.glob(os.path.join(data_path, "*"))

    label_name = "label.nii"
    data_name = "image.nii"

    data_list.sort()

    list_source = [{"data": os.path.join(path, data_name)} for path in data_list]
    list_target = [{"label": os.path.join(path, label_name)} for path in data_list]

    return list_source, list_target


def matrix_from_axis_angle(a):
    """
    Create a rotation matrix from an axis-angle representation.

    Args:
        a (tuple): Axis-angle (ux, uy, uz, theta)

    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array(
        [
            [ci * ux * ux + c, ci * ux * uy - uz * s, ci * ux * uz + uy * s],
            [ci * uy * ux + uz * s, ci * uy * uy + c, ci * uy * uz - ux * s],
            [ci * uz * ux - uy * s, ci * uz * uy + ux * s, ci * uz * uz + c],
        ]
    )
    return R


def get_center(img):
    """
    Get the physical center point of an image.

    Args:
        img (sitk.Image): Input image

    Returns:
        tuple: Physical coordinates of the center point
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint(
        (int(np.ceil(width / 2)), int(np.ceil(height / 2)), int(np.ceil(depth / 2)))
    )


def make_latest_G(opt):
    """
    Create a copy of the latest generator model checkpoint.

    Args:
        opt: Options containing paths and model names
    """
    source_path = f"{opt.checkpoints_dir}/{opt.name}/latest_net_G_A.pth"
    target_path = f"{opt.checkpoints_dir}/{opt.name}/latest_net_G.pth"
    if os.path.exists(source_path):
        try:
            import shutil

            try:
                shutil.copy2(source_path, target_path)
            except PermissionError:
                shutil.copy(source_path, target_path)
        except Exception as e:
            try:
                with open(source_path, "rb") as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
                print(f"Warning: Used basic file copy due to permissions: {str(e)}")
            except Exception as e2:
                print(f"Warning: Could not copy checkpoint file: {str(e2)}")
                print(
                    "Training completed but checkpoint copy failed - manual copy may be needed"
                )


def setup_logging():
    """
    Set up logging configuration.

    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("test_debug.log")],
    )
    return logging.getLogger(__name__)


def cleanup():
    """
    Clean up GPU memory and garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def set_seed(seed=0):
    """
    Set random seed for reproducibility.

    Args:
        seed (int, optional): Random seed value
    """
    torch.manual_seed(seed)


def init_wandb(opt):
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        opt: Options containing WandB configuration
    """
    if opt.use_wandb:
        try:
            os.environ["WANDB_START_METHOD"] = "thread"
            wandb.login(key="cde9483f01d3d4c883d033dbde93150f7d5b22d5", timeout=60)
        except Exception as e:
            print(f"Warning: WandB initialization failed: {str(e)}")
            print("Continuing without WandB logging...")
            opt.use_wandb = False


def set_starting_epoch(opt):
    """
    Set the starting epoch for training continuation.

    Args:
        opt: Options containing training configuration
    """
    if not opt.continue_train:
        return

    if opt.which_epoch.isdigit():
        starting_epoch = int(opt.which_epoch)
        opt.epoch_count = starting_epoch + 1
        print(
            f"Continuing training from epoch {opt.epoch_count} (after epoch {starting_epoch})"
        )
        return

    if opt.which_epoch == "latest":
        import os
        import re
        import glob

        from dataset.data_loader import generate_experiment_name

        patch_size_str = "_".join(str(x) for x in opt.patch_size[:2])
        if len(opt.patch_size) > 2:
            patch_size_str += f"_{opt.patch_size[2]}"
        else:
            patch_size_str += "_10"

        experiment_name = generate_experiment_name(opt, patch_size_str)
        print(f"Generated experiment name: {experiment_name}")

        log_path = os.path.join(opt.checkpoints_dir, experiment_name, "loss_log.txt")

        if os.path.exists(log_path):
            print(f"Found log file at: {log_path}")
        else:
            print(f"WARNING: No loss_log.txt found in {experiment_name}")
            print(f"Looking for log file at: {log_path}")

            log_path = None
            patterns = [
                os.path.join(
                    opt.checkpoints_dir,
                    f"*_{opt.name}_ngf{opt.ngf}_ndf{opt.ndf}_*",
                    "loss_log.txt",
                ),
                os.path.join(
                    "checkpoints",
                    f"*_{opt.name}_ngf{opt.ngf}_ndf{opt.ndf}_*",
                    "loss_log.txt",
                ),
            ]

            for pattern in patterns:
                print(f"Searching with pattern: {pattern}")
                matching_files = glob.glob(pattern, recursive=True)
                if matching_files:
                    log_path = matching_files[0]
                    experiment_dir = os.path.basename(os.path.dirname(log_path))
                    print(f"Found log file in directory: {experiment_dir}")
                    break

        if log_path is None:
            print(
                f"Warning: Could not find loss_log.txt to determine latest epoch. Starting from epoch 1."
            )
            return

        print(f"Determining last completed epoch from {log_path}...")

        last_epoch = 0
        try:
            with open(log_path, "r") as f:
                log_content = f.read()

            model_name_pattern = r"Full name: ([^\n]+)"
            model_names = re.findall(model_name_pattern, log_content)
            if model_names:
                log_model_name = model_names[0]
                print(f"DEBUG - Log file is for model: {log_model_name}")

            val_pattern = r"\[Run: .*?\] \(epoch: (\d+), iters: 0,.*?\) val_"
            val_epochs = re.findall(val_pattern, log_content)
            print(
                f"DEBUG - Found {len(val_epochs)} validation epochs: {val_epochs[:5]}..."
            )

            if not val_epochs:
                alt_val_pattern = r"\[Run:.*?\].*?\(epoch:\s*(\d+).*?\).*?val_"
                val_epochs = re.findall(alt_val_pattern, log_content)
                print(
                    f"DEBUG - Using alternative pattern, found {len(val_epochs)} validation epochs: {val_epochs[:5]}..."
                )

            epoch_pattern = r"\[Run: .*?\] \(epoch: (\d+), iters: \d+,"
            epoch_headers = re.findall(epoch_pattern, log_content)
            print(
                f"DEBUG - Found {len(epoch_headers)} epoch headers: {epoch_headers[:5]}..."
            )

            if not epoch_headers:
                alt_epoch_pattern = r"\[Run:.*?\].*?\(epoch:\s*(\d+).*?iters:"
                epoch_headers = re.findall(alt_epoch_pattern, log_content)
                print(
                    f"DEBUG - Using alternative pattern, found {len(epoch_headers)} epoch headers: {epoch_headers[:5]}..."
                )

            all_epochs = [int(e) for e in val_epochs + epoch_headers]
            if all_epochs:
                last_epoch = max(all_epochs)

            if last_epoch > 0:
                opt.epoch_count = last_epoch + 1
                print(
                    f"Determined from logs: Continuing training from epoch {opt.epoch_count} (after epoch {last_epoch})"
                )
            else:
                print(
                    "Warning: Could not determine last epoch from logs. Starting from epoch 1."
                )
                print("Log file exists but could not find epoch numbers in it.")

        except Exception as e:
            print(f"Error parsing log file to determine epoch: {e}")
            print("Starting from epoch 1.")


def log_to_wandb(opt, metrics, epoch):
    """
    Log metrics to Weights & Biases.

    Args:
        opt: Options containing WandB configuration
        metrics (dict): Metrics to log
        epoch (int): Current epoch
    """
    if not hasattr(opt, "use_wandb") or not opt.use_wandb:
        return

    try:
        if wandb.run is not None:
            if "epoch" not in metrics:
                metrics["epoch"] = epoch
            wandb.log(metrics)
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")


class WandbStepTracker:
    """
    Tracker for ensuring monotonically increasing step values in WandB.
    """

    _instance = None
    _last_step = 0

    @classmethod
    def get_instance(cls):
        """
        Get singleton instance of the tracker.

        Returns:
            WandbStepTracker: Singleton instance
        """
        if cls._instance is None:
            cls._instance = WandbStepTracker()
        return cls._instance

    def get_safe_step(self, proposed_step):
        """
        Get a safe step value that is guaranteed to be monotonically increasing.

        Args:
            proposed_step: Proposed step value

        Returns:
            int: Safe step value
        """
        if proposed_step is None:
            safe_step = self._last_step + 1
            return safe_step

        if not isinstance(proposed_step, int):
            try:
                proposed_step = int(proposed_step)
            except (ValueError, TypeError):
                safe_step = self._last_step + 1
                self._last_step = safe_step
                return safe_step

        if proposed_step <= self._last_step:
            safe_step = self._last_step + 1
        else:
            safe_step = proposed_step

        self._last_step = safe_step
        return safe_step


def setup_sliding_window_validation(trainer, mini=False):
    """
    Configure sliding window validation for a trainer.

    Args:
        trainer: Trainer object to configure
        mini (bool, optional): Whether to use mini (denser) sliding window

    Returns:
        Trainer: Configured trainer object
    """
    val_handler = trainer.validation_handler
    val_handler.use_sliding_window = True

    if hasattr(trainer.opt, "patch_size"):
        patch_size = (
            trainer.opt.patch_size.copy()
            if isinstance(trainer.opt.patch_size, list)
            else list(trainer.opt.patch_size)
        )
        if len(patch_size) < 3:
            patch_size.append(10)
        print(f"Using original patch size from training options: {patch_size}")
    else:
        patch_size = [64, 64, 32]
        print(f"Using default patch size: {patch_size}")

    val_handler.patch_size = patch_size
    val_handler.min_patch_size = [8, 8, 4]

    if not mini:
        val_handler.stride_inplane = max(8, patch_size[0] // 2)
        val_handler.stride_layer = max(2, patch_size[2] // 2)
    else:
        val_handler.stride_inplane = max(8, patch_size[0] // 1 - 1)
        val_handler.stride_layer = max(2, patch_size[2] // 1 - 1)
    print(f"Configured sliding window validation with:")
    print(f"  Patch size: {val_handler.patch_size}")
    print(f"  Minimum patch size: {val_handler.min_patch_size}")
    print(f"  Stride in-plane: {val_handler.stride_inplane}")
    print(f"  Stride layer: {val_handler.stride_layer}")

    return trainer


def _sanitize_wandb_values(log_dict):
    """
    Sanitize values for WandB logging by handling None, non-numeric, and invalid values.

    Args:
        log_dict (dict): Dictionary of values to log

    Returns:
        dict: Sanitized dictionary
    """
    sanitized_dict = {}
    for k, v in log_dict.items():
        if v is None:
            sanitized_dict[k] = 0.0
            continue

        if not isinstance(v, (int, float)):
            try:
                v = float(v)
            except (ValueError, TypeError):
                sanitized_dict[k] = 0.0
                continue

        if math.isnan(v) or math.isinf(v):
            sanitized_dict[k] = 0.0
        else:
            sanitized_dict[k] = v

    return sanitized_dict
