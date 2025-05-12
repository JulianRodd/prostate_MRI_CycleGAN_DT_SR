import argparse
import gc
import json
import os
import shutil

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from dataset.niftiDataset import NifitDataSet
from dataset.processing.augmentation.mri_augmentation import MRIAugmentation
from dataset.processing.padding import Padding
from dataset.processing.random_crop import RandomCrop


def validate_patch_content(patch_sample):
    if (
        patch_sample is None
        or "image" not in patch_sample
        or "label" not in patch_sample
    ):
        return False, 0.0

    image_array = sitk.GetArrayFromImage(patch_sample["image"])
    label_array = sitk.GetArrayFromImage(patch_sample["label"])

    if np.isnan(image_array).any() or np.isinf(image_array).any():
        return False, 0.0

    image_content = np.sum(image_array > -0.95)
    label_content = np.sum(label_array > 0)
    total_pixels = np.prod(image_array.shape)
    min_content_threshold = total_pixels * 0.01
    content_ratio = max(image_content, label_content) / total_pixels * 100
    image_variance = np.var(image_array)
    low_variance = image_variance < 1e-5

    is_valid = (
        image_content > min_content_threshold or label_content > min_content_threshold
    ) and not low_variance

    return is_valid, content_ratio


def preprocess_patches(
    data_path,
    output_dir,
    which_direction="AtoB",
    transform_pipeline=None,
    is_train=True,
    patches_per_image=8,
    batch_size=4,
    num_workers=4,
    max_cache_size_mb=512,
    similarity_threshold=0.7,
):
    print(f"{'Training' if is_train else 'Validation'} patch preprocessing started...")

    dataset = NifitDataSet(
        data_path,
        which_direction=which_direction,
        transforms=None,
        shuffle_labels=False,
        train=is_train,
        max_cache_size_mb=max_cache_size_mb,
    )

    os.makedirs(output_dir, exist_ok=True)

    padding_transform = None
    crop_transform = None
    augmentation_transform = None

    for transform in transform_pipeline:
        if isinstance(transform, Padding):
            padding_transform = transform
        elif isinstance(transform, RandomCrop):
            crop_transform = transform
        elif isinstance(transform, MRIAugmentation):
            augmentation_transform = transform

    if crop_transform is None:
        raise ValueError("RandomCrop transform must be provided")

    original_patches_per_image = crop_transform.patches_per_image
    crop_transform.patches_per_image = patches_per_image

    manifest = {}
    patch_count = 0
    total_attempts = 0
    empty_patches = 0
    similar_patches = 0
    saved_patches = 0
    required_patches_per_image = patches_per_image

    for idx in tqdm(range(len(dataset.images_list)), desc="Processing images"):
        image_path = dataset.images_list[idx]
        label_path = dataset.labels_list[idx]

        try:
            image = dataset.read_image(image_path)
            label = dataset.read_image(label_path)
            sample = {"image": image, "label": label}

            if padding_transform:
                sample = padding_transform(sample)

            if is_train and augmentation_transform:
                try:
                    sample = augmentation_transform(sample)
                except Exception as e:
                    print(f"Warning: Augmentation failed: {e}")

            extracted_patches = []
            patch_idx = 0
            attempts = 0
            max_attempts = patches_per_image * 5
            max_history_patches = min(5, patches_per_image)

            while (
                len(extracted_patches) < patches_per_image and attempts < max_attempts
            ):
                try:
                    recent_patches = (
                        extracted_patches[-max_history_patches:]
                        if extracted_patches
                        else []
                    )

                    patch_sample = crop_transform.extract_center_biased_patch(
                        sample,
                        patch_idx + attempts,
                        previous_patches=recent_patches,
                        similarity_threshold=similarity_threshold,
                    )

                    total_attempts += 1

                    if patch_sample is None:
                        print(
                            f"Warning: Got None patch on attempt {attempts + 1}, retrying..."
                        )
                        attempts += 1
                        continue

                    image_key = "invivo" if "invivo" in patch_sample else "image"
                    label_key = "exvivo" if "exvivo" in patch_sample else "label"

                    if (
                        image_key not in patch_sample
                        or patch_sample[image_key] is None
                        or label_key not in patch_sample
                        or patch_sample[label_key] is None
                    ):
                        print(
                            f"Warning: Invalid patch keys on attempt {attempts + 1}, retrying..."
                        )
                        attempts += 1
                        continue

                    try:
                        is_valid, content_ratio = validate_patch_content(patch_sample)
                    except Exception as e:
                        print(f"Warning: Error validating content: {e}, retrying...")
                        attempts += 1
                        continue

                    if not is_valid:
                        empty_patches += 1
                        print(
                            f"Skipping empty patch (attempt {attempts + 1}, content ratio: {content_ratio:.2f}%)"
                        )
                        attempts += 1
                        continue

                    patch_id = f"{idx:04d}_{patch_idx:02d}"
                    patch_image_path = os.path.join(
                        output_dir, f"img_{patch_id}.nii.gz"
                    )
                    patch_label_path = os.path.join(
                        output_dir, f"lbl_{patch_id}.nii.gz"
                    )

                    try:
                        sitk.WriteImage(patch_sample[image_key], patch_image_path)
                        sitk.WriteImage(patch_sample[label_key], patch_label_path)
                    except Exception as e:
                        print(f"Warning: Error saving patch: {e}, retrying...")
                        attempts += 1
                        continue

                    manifest[patch_count] = {
                        "original_image": image_path,
                        "original_label": label_path,
                        "patch_image": patch_image_path,
                        "patch_label": patch_label_path,
                        "image_idx": idx,
                        "patch_idx": patch_idx,
                        "content_ratio": float(content_ratio),
                        "is_valid": True,
                    }

                    patch_count += 1
                    saved_patches += 1
                    patch_idx += 1

                    diversity_info = {
                        image_key: patch_sample[image_key],
                        label_key: patch_sample[label_key],
                    }

                    if "fingerprint" in patch_sample:
                        diversity_info["fingerprint"] = patch_sample["fingerprint"]

                    extracted_patches.append(diversity_info)

                    diversity_score = patch_sample.get("diversity", 1.0)
                    print(
                        f"Saved patch {patch_idx}/{required_patches_per_image} for image {idx} "
                        f"(content: {content_ratio:.2f}%, diversity: {diversity_score:.2f})"
                    )

                    patch_sample[image_key] = None
                    patch_sample[label_key] = None
                    patch_sample = None
                    gc.collect()

                except Exception as e:
                    print(f"Error processing patch {patch_idx} for image {idx}: {e}")
                    attempts += 1
                    gc.collect()
                    continue

            if len(extracted_patches) < patches_per_image:
                print(
                    f"Warning: Could only extract {len(extracted_patches)} patches from image {idx}"
                )

            for patch in extracted_patches:
                for key in list(patch.keys()):
                    patch[key] = None
            extracted_patches.clear()
            extracted_patches = None
            gc.collect()

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue

        image = None
        label = None
        sample = None
        gc.collect()

    manifest_path = os.path.join(output_dir, "patch_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    crop_transform.patches_per_image = original_patches_per_image

    print(f"Patch preprocessing statistics:")
    print(f"  Total patch attempts: {total_attempts}")
    print(f"  Empty patches rejected: {empty_patches}")
    print(
        f"  Similar patches rejected: {total_attempts - empty_patches - saved_patches}"
    )
    print(f"  Successfully saved patches: {saved_patches}")
    print(f"  Success rate: {saved_patches / max(1, total_attempts) * 100:.2f}%")

    print(f"Preprocessed {patch_count} patches from {len(dataset.images_list)} images")
    print(f"Patches saved to {output_dir}")
    print(f"Manifest saved to {manifest_path}")

    return manifest


def prepare_dataset_patches(opt, is_train=True):
    data_path = opt.data_path if is_train else opt.val_path

    patch_size_str = f"{opt.patch_size[0]}x{opt.patch_size[1]}x{opt.patch_size[2]}"
    patches_per_image = (
        opt.patches_per_image if hasattr(opt, "patches_per_image") else 8
    )
    min_pixel = opt.min_pixel if hasattr(opt, "min_pixel") else 0.5

    min_pixel_str = f"{min_pixel:.2f}".replace(".", "_")

    param_dir_name = f"p{patch_size_str}_ppi{patches_per_image}_mp{min_pixel_str}"

    parent_dir = os.path.dirname(data_path)
    dataset_name = os.path.basename(data_path)
    temp_dir = os.path.join(
        parent_dir,
        f"patches_{dataset_name}_{'train' if is_train else 'val'}_{param_dir_name}",
    )

    manifest_path = os.path.join(temp_dir, "patch_manifest.json")
    if os.path.exists(manifest_path):
        patch_count = len([f for f in os.listdir(temp_dir) if f.startswith("img_")])

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            if len(manifest) > 0:
                print(f"Found existing patch directory with {patch_count} patches")
                print(f"Using existing patches from {temp_dir}")
                return temp_dir, manifest_path
        except:
            print("Existing patch manifest is invalid, recreating patches")

    if os.path.exists(temp_dir):
        print(f"Removing existing temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir, exist_ok=True)

    from dataset.data_loader import get_transforms

    transform_pipeline = get_transforms(opt, is_train=is_train)

    patches_per_image = (
        opt.patches_per_image if hasattr(opt, "patches_per_image") else 8
    )

    print(f"Creating new patch dataset with parameters:")
    print(f"  - Patch size: {patch_size_str}")
    print(f"  - Patches per image: {patches_per_image}")
    print(f"  - Min pixel content: {min_pixel}")
    print(f"  - Output directory: {temp_dir}")

    preprocess_patches(
        data_path=data_path,
        output_dir=temp_dir,
        which_direction="AtoB",
        transform_pipeline=transform_pipeline,
        is_train=is_train,
        patches_per_image=patches_per_image,
        batch_size=opt.batch_size if hasattr(opt, "batch_size") else 4,
        num_workers=opt.num_workers if hasattr(opt, "num_workers") else 4,
        max_cache_size_mb=512,
    )

    return temp_dir, manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess patches for training")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for patches"
    )
    parser.add_argument(
        "--patch_size",
        nargs=3,
        type=int,
        default=[64, 64, 32],
        help="Patch size (x, y, z)",
    )
    parser.add_argument(
        "--patches_per_image", type=int, default=8, help="Number of patches per image"
    )
    parser.add_argument(
        "--is_train", action="store_true", help="Whether this is training data"
    )
    parser.add_argument(
        "--force_recreate",
        action="store_true",
        help="Force recreation of patches even if they exist",
    )

    args = parser.parse_args()

    class Options:
        def __init__(self):
            self.patch_size = args.patch_size
            self.data_path = args.data_path
            self.patches_per_image = args.patches_per_image
            self.drop_ratio = 0.1
            self.min_pixel = 0.9
            self.batch_size = 4

    opt = Options()

    from dataset.data_loader import get_transforms

    transform_pipeline = get_transforms(opt, is_train=args.is_train)

    os.makedirs(args.output_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "patch_manifest.json")
    if os.path.exists(manifest_path) and not args.force_recreate:
        patch_count = len(
            [f for f in os.listdir(args.output_dir) if f.startswith("img_")]
        )
        print(
            f"Found {patch_count} existing patches. Use --force_recreate to regenerate."
        )
    else:
        if args.force_recreate and os.path.exists(args.output_dir):
            print(f"Removing existing directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)

        preprocess_patches(
            data_path=args.data_path,
            output_dir=args.output_dir,
            transform_pipeline=transform_pipeline,
            is_train=args.is_train,
            patches_per_image=args.patches_per_image,
        )
