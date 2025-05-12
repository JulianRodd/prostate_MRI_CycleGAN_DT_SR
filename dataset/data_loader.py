import os
import random
import string

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.precomputed_dataset import PrecomputedPatchDataset
from dataset.preprocess_patches import prepare_dataset_patches

from dataset.processing.augmentation.mri_augmentation import MRIAugmentation
from dataset.processing.padding import Padding
from dataset.processing.random_crop import RandomCrop


def setup_full_image_validation(opt):
    data_path = opt.val_path
    print(f"Determining full image dimensions from validation data at: {data_path}")

    import os
    from utils.utils import lstFiles

    img_path = os.path.join(data_path, "invivo")
    images_list = lstFiles(img_path)

    if not images_list:
        print("Warning: No validation images found. Using default full image size.")
        return [400, 400, 70]

    import SimpleITK as sitk

    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(images_list[0])
        reader.SetLoadPrivateTags(False)
        image = reader.Execute()

        full_size = list(image.GetSize())
        full_size = [full_size[2], full_size[1], full_size[0]]

        print(f"Detected full image size: {full_size}")
        return full_size
    except Exception as e:
        print(f"Error determining image size: {e}")
        return [400, 400, 70]


def simple_collate_fn(batch):
    try:
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.stack([item[1] for item in batch], dim=0)
        return images, labels
    except RuntimeError as e:
        if "resize" in str(e).lower():
            print("Falling back to contiguous copy collation due to resize error")
            images = torch.stack(
                [item[0].clone().contiguous() for item in batch], dim=0
            )
            labels = torch.stack(
                [item[1].clone().contiguous() for item in batch], dim=0
            )
            return images, labels
        else:
            raise


def generate_experiment_name(opt, patch_size_str):
    if hasattr(opt, "continue_training_run") and opt.continue_training_run:
        if len(opt.continue_training_run) != 3:
            raise ValueError("continue_training_run must be exactly 3 characters")
        random_string = opt.continue_training_run
    else:
        random_string = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=3)
        )

    return f"{random_string}_{opt.name}_ngf{opt.ngf}_ndf{opt.ndf}_patch{patch_size_str}"


def get_transforms(opt, is_train=True):
    base_transforms = [
        Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
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

        augmentation = MRIAugmentation(
            augmentation_probs=augmentation_probs,
            random_order=True,
            max_augmentations=4,
            background_value=0,
        )
        base_transforms.append(augmentation)

    if is_train:
        crop_transform = RandomCrop(
            (opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            drop_ratio=float(opt.drop_ratio),
            min_pixel=opt.min_pixel,
            patches_per_image=opt.patches_per_image,
            patch_size=opt.patch_size,
        )

        crop_transform.always_apply_zoom = True
        crop_transform.debug_mode = True

        print(
            f"Training crop: patches_per_image={opt.patches_per_image}, min_pixel={opt.min_pixel}, forced_zoom=True"
        )
    else:
        crop_transform = RandomCrop(
            (opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            drop_ratio=0.0,
            min_pixel=opt.min_pixel,
            patches_per_image=opt.patches_per_image,
            patch_size=opt.patch_size,
        )

        crop_transform.always_apply_zoom = True
        crop_transform.debug_mode = False

        print(
            f"Validation crop: patches_per_image={opt.patches_per_image}, min_pixel={opt.min_pixel}, forced_zoom=True"
        )

    base_transforms.append(crop_transform)
    return base_transforms


def worker_init_fn(worker_id, seed=None):
    worker_seed = seed + worker_id if seed is not None else 42 + worker_id

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    if hasattr(torch, "utils") and hasattr(torch.utils, "data"):
        if hasattr(torch.utils.data, "_utils") and hasattr(
            torch.utils.data._utils, "pin_memory"
        ):
            if hasattr(torch.utils.data._utils.pin_memory, "DEFAULT_PIN_MEMORY_DEVICE"):
                torch.utils.data._utils.pin_memory.DEFAULT_PIN_MEMORY_DEVICE = ""

    if hasattr(torch.utils.data, "_utils") and hasattr(
        torch.utils.data._utils, "worker"
    ):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"


def setup_dataloaders(opt):
    print("Setting up dataloaders...")

    if not hasattr(opt, "patches_per_image"):
        opt.patches_per_image = 8
        print(f"Setting default patches_per_image={opt.patches_per_image}")

    patch_size_str = (
        str(opt.patch_size)
        .replace("[", "")
        .replace("]", "")
        .replace(",", "")
        .replace(" ", "_")
    )
    opt.name = generate_experiment_name(opt, patch_size_str)

    train_transforms = get_transforms(opt, is_train=True)

    train_patch_dir, train_manifest_path = prepare_dataset_patches(opt, is_train=True)

    train_set = PrecomputedPatchDataset(
        manifest_path=train_manifest_path,
        is_train=True,
        max_cache_size_mb=512,
        skip_empty_patches=False,
    )

    print(f"Training configuration:")
    print(f"  Patch size: {opt.patch_size}")
    print(f"  Min non-background pixels: {opt.min_pixel}")
    print(f"  Patches per image: {opt.patches_per_image}")
    print(f"  Training patches directory: {train_patch_dir}")

    use_full_validation = getattr(opt, "use_full_validation", False)

    if use_full_validation:
        print("\nSetting up validation with FULL IMAGES")

        full_image_size = setup_full_image_validation(opt)

        if hasattr(opt, "full_image_size") and opt.full_image_size:
            full_image_size = opt.full_image_size
            print(f"Using manually specified full image size: {full_image_size}")

        print(f"Validation will use FULL images with size: {full_image_size}")

        val_transforms = get_transforms(opt, is_train=False)

        for t in val_transforms:
            if isinstance(t, RandomCrop):
                t.original_patch_size = t.patch_size.copy()
                t.patch_size = full_image_size
                t.patches_per_image = 1
                original_extract = t.extract_center_biased_patch
                t.extract_center_biased_patch = lambda sample, idx, **kwargs: sample
                print(
                    f"Modified validation crop to use full images of size {full_image_size}"
                )
                break

        from dataset.niftiDataset import NifitDataSet

        val_set = NifitDataSet(
            data_path=opt.val_path,
            which_direction="AtoB",
            transforms=val_transforms,
            shuffle_labels=False,
            train=False,
            test=False,
            is_inference=False,
            max_cache_size_mb=256,
        )

        print(f"Validation will use {len(val_set)} full images")
    else:
        print("\nSetting up validation with precomputed patches")
        val_patch_dir, val_manifest_path = prepare_dataset_patches(opt, is_train=False)

        val_set = PrecomputedPatchDataset(
            manifest_path=val_manifest_path,
            is_train=False,
            max_cache_size_mb=512,
        )

        print(f"Validation patches directory: {val_patch_dir}")
        print(f"Validation will use {len(val_set)} patches")

    worker_seed = opt.seed if hasattr(opt, "seed") else 42
    using_cpu = not torch.cuda.is_available()

    if using_cpu:
        batch_size = 1
        num_workers = 0
        pin_memory = False
        persistent_workers = False

        if not hasattr(opt, "accumulation_steps") or opt.accumulation_steps < 1:
            original_batch = getattr(opt, "batch_size", 4)
            opt.accumulation_steps = original_batch

        print(
            f"CPU configuration: batch_size={batch_size}, accumulation_steps={opt.accumulation_steps}"
        )
    else:
        batch_size = opt.batch_size
        num_workers = min(4, os.cpu_count() or 4)
        pin_memory = True
        persistent_workers = True

        print(f"GPU configuration: batch_size={batch_size}, num_workers={num_workers}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, worker_seed),
        drop_last=True,
        collate_fn=simple_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0 if using_cpu else 1,
        pin_memory=pin_memory,
        collate_fn=simple_collate_fn,
    )

    return train_loader, val_loader
