import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import traceback
from torch.utils.data import DataLoader

from dataset.processing.augmentation.mri_augmentation import MRIAugmentation
from dataset.processing.random_crop import RandomCrop
from dataset.processing.padding import Padding
from dataset.niftiDataset import NifitDataSet
from dataset.precomputed_dataset import PrecomputedPatchDataset
from dataset.preprocess_patches import prepare_dataset_patches


class VisualizeSlices:
    class MultiIndexTracker:
        def __init__(self, fig, axes, volumes, titles):
            self.axes = axes
            self.volumes = volumes
            if not volumes:
                return

            rows, cols, self.slices = volumes[0].shape
            self.current_slice = self.slices // 2
            self.imshow_objects = []
            self.colorbar_objects = []

            for i, (ax, vol, title) in enumerate(zip(self.axes, volumes, titles)):
                ax.set_title(title, fontsize=10)

                cmap = "gray"
                p1, p99 = np.percentile(vol, (1, 99))
                vmin, vmax = p1, p99

                im = ax.imshow(
                    vol[:, :, self.current_slice],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                self.imshow_objects.append(im)

                ax.set_ylabel(f"Slice: {self.current_slice}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

                if i % 2 == 0:
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=7)
                    self.colorbar_objects.append(cbar)

            self.update()

        def onscroll(self, event):
            if event.button == "up":
                self.current_slice = min(self.slices - 1, self.current_slice + 1)
            else:
                self.current_slice = max(0, self.current_slice - 1)
            self.update()

        def update(self):
            for im, vol in zip(self.imshow_objects, self.volumes):
                im.set_data(vol[:, :, self.current_slice])

            for ax in self.axes:
                if ax.get_visible():
                    ax.set_ylabel(f"Slice: {self.current_slice}", fontsize=8)

            self.axes[0].figure.canvas.draw_idle()

    @staticmethod
    def plot_volumes(volumes_list, titles_list):
        if not volumes_list or not titles_list:
            print("Error: Empty volume or title list")
            return

        try:
            num_images = len(volumes_list)
            num_pairs = num_images // 2

            if num_pairs <= 3:
                grid_rows = 1
                grid_cols = num_pairs
            elif num_pairs <= 8:
                grid_rows = 2
                grid_cols = (num_pairs + 1) // 2
            else:
                import math

                grid_size = math.ceil(math.sqrt(num_pairs))
                grid_rows = grid_size
                grid_cols = (num_pairs + grid_size - 1) // grid_size

            fig, axes = plt.subplots(
                grid_rows, grid_cols * 2, figsize=(grid_cols * 5, grid_rows * 4)
            )

            if grid_rows == 1 and grid_cols == 1:
                axes = axes.reshape(1, 2)
            elif grid_rows == 1:
                axes = axes.reshape(1, -1)
            elif grid_cols == 1:
                axes = axes.reshape(-1, 2)

            axes_flat = axes.flatten()

            for i in range(num_images, len(axes_flat)):
                axes_flat[i].set_visible(False)

            used_axes = axes_flat[:num_images]
            rotated_volumes = [np.rot90(vol, k=-1) for vol in volumes_list]

            tracker = VisualizeSlices.MultiIndexTracker(
                fig, used_axes, rotated_volumes, titles_list
            )
            fig.canvas.mpl_connect("scroll_event", tracker.onscroll)

            fig.suptitle("Use scroll wheel to navigate slices simultaneously", y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()
        except Exception as e:
            print(f"Error in plot_volumes: {str(e)}")
            traceback.print_exc()


class Options:
    def __init__(self, args):
        self.patch_size = args.patch_size
        self.data_path = args.data_path
        self.patches_per_image = args.patches_per_image
        self.drop_ratio = args.drop_ratio
        self.min_pixel = args.min_pixel
        self.batch_size = args.batch_size
        self.val_path = args.data_path


def setup_data_loader(args):
    try:
        opt = Options(args)

        if args.use_precomputed_patches:
            print("Using precomputed patch dataset...")
            transforms = get_transforms(args, is_train=True)
            temp_dir, manifest_path = prepare_dataset_patches(opt, is_train=True)

            dataset = PrecomputedPatchDataset(
                manifest_path=manifest_path,
                is_train=True,
                max_cache_size_mb=512,
                content_threshold=args.min_pixel / 100,
                skip_empty_patches=True,
            )

            print(f"Loaded precomputed patch dataset with {len(dataset)} patches")
            stats = dataset.get_bad_patches_stats()
            print(f"Bad patch statistics: {stats}")
        else:
            print("Using on-the-fly patch generation...")
            transforms = get_transforms(args, is_train=True)

            dataset = NifitDataSet(
                args.data_path,
                which_direction="AtoB",
                transforms=transforms,
                shuffle_labels=False,
                train=True,
            )

            print(f"Created on-the-fly dataset with {len(dataset)} possible patches")

        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True), dataset
    except Exception as e:
        print(f"Error in setup_data_loader: {str(e)}")
        traceback.print_exc()
        return None, None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Visualization Tool")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training data",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        required=True,
        help="Input dimension for patches [X Y Z]",
    )
    parser.add_argument(
        "--min_pixel",
        type=float,
        required=True,
        help="Minimum percentage of non-zero pixels in label",
    )
    parser.add_argument(
        "--patches_per_image",
        type=int,
        default=1,
        help="Number of patches to extract per image",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (currently supports 1)"
    )
    parser.add_argument(
        "--drop_ratio",
        type=float,
        default=0,
        help="Probability to drop empty label patches",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        default=False,
        help="Whether to resample images",
    )
    parser.add_argument(
        "--new_resolution",
        type=float,
        nargs=3,
        default=(0.5, 0.5, 0.5),
        help="New resolution for resampling",
    )
    parser.add_argument(
        "--registration_type",
        type=str,
        default="rigid",
        help="Type of registration to apply",
    )
    parser.add_argument(
        "--use_precomputed_patches",
        action="store_true",
        help="Use precomputed patches instead of on-the-fly generation",
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        help="Specific patch directory to check (bypasses preprocessing)",
    )

    return parser.parse_args()


def get_transforms(args, is_train=True):
    min_pixel = float(args.min_pixel)
    if min_pixel >= 1.0:
        min_pixel = int(min_pixel)
    else:
        total_voxels = args.patch_size[0] * args.patch_size[1] * args.patch_size[2]
        min_pixel = int(total_voxels * min_pixel / 100)

    base_transforms = [
        Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
    ]

    if is_train:
        augmentation_probs = {
            "affine": 0.9,
            "elastic": 0.6,
            "bias_field": 0.6,
            "gamma": 0.6,
            "motion": 0.5,
            "noise": 0.5,
            "blur": 0.4,
        }

        base_transforms.append(
            MRIAugmentation(
                augmentation_probs=augmentation_probs,
                random_order=True,
                max_augmentations=4,
                background_value=0,
            )
        )

        crop_transform = RandomCrop(
            (args.patch_size[0], args.patch_size[1], args.patch_size[2]),
            float(args.drop_ratio),
            min_pixel,
            args.patches_per_image if hasattr(args, "patches_per_image") else 1,
            args.patch_size,
        )
        crop_transform.always_apply_zoom = True
        base_transforms.append(crop_transform)
    else:
        crop_transform = RandomCrop(
            (args.patch_size[0], args.patch_size[1], args.patch_size[2]),
            0,
            min_pixel,
            args.patches_per_image if hasattr(args, "patches_per_image") else 1,
            args.patch_size,
        )
        crop_transform.always_apply_zoom = True
        base_transforms.append(crop_transform)

    return base_transforms


def visualize_random_samples(dataset, loader, num_samples):
    num_samples = min(16, num_samples)
    print(f"\nShowing {num_samples} random patches...")

    volumes = []
    titles = []

    empty_patches = 0
    attempts = 0
    max_attempts_total = num_samples * 5

    for i in range(num_samples):
        success = False
        max_attempts_per_sample = 5
        sample_attempts = 0

        while (
            not success
            and sample_attempts < max_attempts_per_sample
            and attempts < max_attempts_total
        ):
            try:
                rand_int = random.randint(0, len(dataset) - 1)
                print(f"Getting sample {i + 1} (index {rand_int})")

                volume, label = loader.dataset[rand_int]

                volume_content_pct = (
                    (volume > -0.95).sum().item() / volume.numel() * 100
                )
                label_content_pct = (label > 0).sum().item() / label.numel() * 100

                if volume_content_pct < 5 or label_content_pct < 5:
                    print(
                        f"Sample {i + 1} appears empty (content: {volume_content_pct:.1f}%/{label_content_pct:.1f}%), trying again..."
                    )
                    empty_patches += 1
                    attempts += 1
                    sample_attempts += 1
                    continue

                volume = np.squeeze(volume.numpy(), axis=0)
                label = np.squeeze(label.numpy(), axis=0)

                volume_title = f"Image {i + 1} ({volume_content_pct:.1f}%)"
                label_title = f"Label {i + 1} ({label_content_pct:.1f}%)"

                volumes.extend([volume, label])
                titles.extend([volume_title, label_title])

                print(f"\nPatch {i + 1}:")
                print(f"Shape: {volume.shape}")
                print(f"Image range: [{volume.min():.2f}, {volume.max():.2f}]")
                print(f"Label range: [{label.min():.2f}, {label.max():.2f}]")
                print(f"Image content: {volume_content_pct:.2f}%")
                print(f"Label content: {label_content_pct:.2f}%")

                success = True
            except Exception as e:
                print(f"Error getting sample: {e}")
                traceback.print_exc()
                attempts += 1
                sample_attempts += 1

        if not success:
            print(
                f"Failed to get valid sample after {sample_attempts} attempts for sample {i + 1}"
            )

    if empty_patches > 0:
        print(
            f"Warning: Encountered {empty_patches} empty patches during visualization"
        )

    if volumes:
        VisualizeSlices.plot_volumes(volumes, titles)
    else:
        print("Error: No valid patches could be visualized")


def check_direct_patches(patch_dir):
    print(f"Checking patches in directory: {patch_dir}")

    manifest_path = os.path.join(patch_dir, "patch_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: No manifest file found at {manifest_path}")
        return None, None

    dataset = PrecomputedPatchDataset(
        manifest_path=manifest_path,
        is_train=True,
        max_cache_size_mb=512,
        content_threshold=0.01,
        skip_empty_patches=True,
    )

    print(f"Loaded patch dataset with {len(dataset)} patches")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    return loader, dataset


def main():
    args = parse_arguments()

    print("\nConfiguration:")
    print(f"Data path: {args.data_path}")
    print(f"Patch size: {args.patch_size}")
    print(f"Min pixel: {args.min_pixel}")
    print(f"Patches per image: {args.patches_per_image}")
    print(f"Drop ratio: {args.drop_ratio}")
    print(f"Using precomputed patches: {args.use_precomputed_patches}")

    if args.patch_dir and os.path.exists(args.patch_dir):
        loader, dataset = check_direct_patches(args.patch_dir)
    else:
        loader, dataset = setup_data_loader(args)

    if loader is None or dataset is None:
        print("Error setting up data loader. Exiting.")
        return

    print(f"Dataset size: {len(dataset)}")
    visualize_random_samples(dataset, loader, args.patches_per_image)


if __name__ == "__main__":
    main()
