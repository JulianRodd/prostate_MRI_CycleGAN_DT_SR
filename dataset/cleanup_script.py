import argparse
import json
import os
import shutil

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def validate_patch_content(image_path, label_path):
    try:
        # Read image and label
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        # Get arrays
        image_array = sitk.GetArrayFromImage(image)
        label_array = sitk.GetArrayFromImage(label)

        # Check for NaN or inf values
        if np.isnan(image_array).any() or np.isinf(image_array).any():
            return False, 0.0

        # Calculate content metrics
        image_content = np.sum(image_array > -0.95)
        label_content = np.sum(label_array > 0)
        total_pixels = np.prod(image_array.shape)
        content_ratio = max(image_content, label_content) / total_pixels * 100

        # Validation criteria
        min_content_threshold = total_pixels * 0.01
        image_variance = np.var(image_array)
        low_variance = image_variance < 1e-5

        is_valid = (
            image_content > min_content_threshold
            or label_content > min_content_threshold
        ) and not low_variance

        return is_valid, content_ratio

    except Exception as e:
        print(f"Error validating patch: {e}")
        return False, 0.0


def backup_manifest(manifest_path):
    backup_path = os.path.join(
        os.path.dirname(manifest_path), "patch_manifest.backup.json"
    )
    shutil.copy2(manifest_path, backup_path)
    print(f"Backed up manifest to {backup_path}")


def load_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Convert keys to integers
    return {int(k): v for k, v in manifest.items()}


def save_manifest(manifest_path, manifest):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"Updated manifest")


def move_empty_patches(empty_patches, manifest, patch_dir):
    empty_dir = os.path.join(patch_dir, "empty_patches")
    os.makedirs(empty_dir, exist_ok=True)

    print(f"Moving empty patches to {empty_dir}")

    for idx, _, _ in empty_patches:
        patch_info = manifest[idx]
        image_path = patch_info["patch_image"]
        label_path = patch_info["patch_label"]

        try:
            if os.path.exists(image_path):
                shutil.move(
                    image_path, os.path.join(empty_dir, os.path.basename(image_path))
                )

            if os.path.exists(label_path):
                shutil.move(
                    label_path, os.path.join(empty_dir, os.path.basename(label_path))
                )

            # Mark as invalid in manifest
            manifest[idx]["is_valid"] = False

        except Exception as e:
            print(f"Error moving patch {idx}: {e}")

    return manifest


def cleanup_empty_patches(patch_dir, content_threshold=1.0, dry_run=True):
    manifest_path = os.path.join(patch_dir, "patch_manifest.json")

    if not os.path.exists(manifest_path):
        print(f"No manifest found at {manifest_path}")
        return {"error": "No manifest found"}

    # Load and backup manifest
    manifest = load_manifest(manifest_path)
    backup_manifest(manifest_path)

    # Scan for empty patches
    empty_patches = []
    total_patches = len(manifest)

    print(f"Scanning {total_patches} patches for content...")
    for idx, patch_info in tqdm(manifest.items()):
        image_path = patch_info["patch_image"]
        label_path = patch_info["patch_label"]

        # Check if files exist
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            empty_patches.append((idx, 0.0, "Missing files"))
            continue

        # Validate content
        is_valid, content_ratio = validate_patch_content(image_path, label_path)

        # Add content information to manifest
        manifest[idx]["content_ratio"] = float(content_ratio)
        manifest[idx]["is_valid"] = bool(is_valid)

        if not is_valid or content_ratio < content_threshold:
            empty_patches.append((idx, content_ratio, "Empty content"))

    # Update manifest with content information
    save_manifest(manifest_path, manifest)

    # Process empty patches
    if empty_patches:
        print(
            f"Found {len(empty_patches)} empty patches out of {total_patches} ({len(empty_patches)/total_patches*100:.2f}%)"
        )

        for idx, ratio, reason in empty_patches[:10]:  # Show first 10
            print(f"  Patch {idx}: {ratio:.2f}% content - {reason}")

        if len(empty_patches) > 10:
            print(f"  ... and {len(empty_patches) - 10} more")

        if not dry_run:
            manifest = move_empty_patches(empty_patches, manifest, patch_dir)
            save_manifest(manifest_path, manifest)
    else:
        print("No empty patches found")

    # Return statistics
    return {
        "total_patches": total_patches,
        "empty_patches": len(empty_patches),
        "percentage_empty": len(empty_patches) / total_patches * 100,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up empty patches from a dataset"
    )
    parser.add_argument(
        "--patch_dir", type=str, required=True, help="Directory containing patches"
    )
    parser.add_argument(
        "--content_threshold",
        type=float,
        default=1.0,
        help="Minimum percentage of content required",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report empty patches, don't remove them",
    )

    args = parser.parse_args()

    stats = cleanup_empty_patches(
        args.patch_dir, content_threshold=args.content_threshold, dry_run=args.dry_run
    )

    print("\nCleanup Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
