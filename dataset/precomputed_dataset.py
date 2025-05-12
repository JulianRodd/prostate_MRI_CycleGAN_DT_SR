import gc
import json
import os
import random
import warnings

import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data


class PrecomputedPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path,
        is_train=True,
        max_cache_size_mb=512,
        content_threshold=0.01,
        skip_empty_patches=True,
    ):
        self.manifest_path = manifest_path
        self.is_train = is_train
        self.bit = sitk.sitkFloat32
        self.content_threshold = content_threshold
        self.skip_empty_patches = skip_empty_patches

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.manifest = {int(k): v for k, v in self.manifest.items()}
        self.indices = sorted(list(self.manifest.keys()))

        if skip_empty_patches:
            valid_indices = []
            for idx in self.indices:
                patch_info = self.manifest[idx]

                if "content_ratio" in patch_info and "is_valid" in patch_info:
                    if (
                        patch_info["is_valid"]
                        and patch_info["content_ratio"] >= content_threshold * 100
                    ):
                        valid_indices.append(idx)
                else:
                    valid_indices.append(idx)

            filtered_count = len(self.indices) - len(valid_indices)
            if filtered_count > 0:
                print(
                    f"Filtered out {filtered_count} empty patches based on manifest information"
                )

            self.indices = valid_indices

        self.using_cpu = not torch.cuda.is_available()
        if self.using_cpu:
            max_cache_size_mb = min(256, max_cache_size_mb)
            print(f"CPU mode detected: Limiting cache to {max_cache_size_mb}MB")

        self.max_cache_size_mb = max_cache_size_mb
        self.current_cache_size_mb = 0
        self.cached_items = {}
        self.cached_items_size = {}
        self.access_count = {}

        print(f"Loaded manifest with {len(self.indices)} patches")

        self.bad_patch_indices = set()
        self.total_empty_patches = 0

        self.index_map = {i: idx for i, idx in enumerate(self.indices)}

    def _ensure_contiguous_tensor(self, tensor):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensor = tensor.clone().detach()
        return tensor

    def __len__(self):
        return len(self.indices)

    def read_image(self, path):
        # Handle path remapping for different environments
        if "/data/temporary/julian/" in path and os.environ.get(
            "KAGGLE_KERNEL_RUN_TYPE"
        ):
            # We're on Kaggle but have server paths
            path = path.replace(
                "/data/temporary/julian/organized_data",
                "/kaggle/input/prostate-data/organized_data",
            )

        if path in self.cached_items:
            self.access_count[path] = self.access_count.get(path, 0) + 1
            return self.cached_items[path]

        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.SetLoadPrivateTags(False)
            image = reader.Execute()

            if image.GetPixelID() != sitk.sitkFloat32:
                cast_filter = sitk.CastImageFilter()
                cast_filter.SetOutputPixelType(sitk.sitkFloat32)
                image = cast_filter.Execute(image)

            size_mb = (np.prod(image.GetSize()) * 4) / (1024 * 1024)

            if size_mb > self.max_cache_size_mb * 0.4:
                return image

            if self.current_cache_size_mb + size_mb > self.max_cache_size_mb:
                items_to_remove = sorted(
                    [
                        (k, v)
                        for k, v in self.access_count.items()
                        if k in self.cached_items
                    ],
                    key=lambda x: x[1],
                )

                while (
                    self.current_cache_size_mb + size_mb > self.max_cache_size_mb
                    and items_to_remove
                ):
                    key, _ = items_to_remove.pop(0)
                    if key in self.cached_items_size:
                        self.current_cache_size_mb -= self.cached_items_size[key]

                    if key in self.cached_items:
                        del self.cached_items[key]
                    if key in self.cached_items_size:
                        del self.cached_items_size[key]
                    if key in self.access_count:
                        del self.access_count[key]

            self.cached_items[path] = image
            self.cached_items_size[path] = size_mb
            self.current_cache_size_mb += size_mb
            self.access_count[path] = 1

            return image
        except Exception as e:
            print(f"Error reading image {path}: {e}")
            gc.collect()
            raise

    def validate_patch_content(self, image_tensor, label_tensor):
        if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
            return False, 0.0

        image_content = torch.sum(image_tensor > -0.95).item()
        label_content = torch.sum(label_tensor > 0).item()
        total_pixels = torch.numel(image_tensor)
        content_ratio = max(image_content, label_content) / total_pixels * 100
        is_valid = content_ratio >= self.content_threshold * 100

        return is_valid, content_ratio

    def normalize_tensor(self, tensor):
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

        eps = 1e-8
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        if abs(tensor_max - tensor_min) > eps:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        else:
            tensor = torch.zeros_like(tensor)

        tensor = tensor * 2 - 1
        return tensor

    def get_fallback_index(self, problematic_index):
        original_idx = self.index_map[problematic_index]
        good_indices = [
            i for i in range(len(self.indices)) if i not in self.bad_patch_indices
        ]

        if not good_indices:
            self.bad_patch_indices = {problematic_index}
            good_indices = [
                i for i in range(len(self.indices)) if i != problematic_index
            ]

        fallback_idx = random.choice(good_indices)
        return fallback_idx

    def __getitem__(self, index):
        try:
            actual_index = self.indices[index]
            patch_info = self.manifest[actual_index]
            image_path = patch_info["patch_image"]
            label_path = patch_info["patch_label"]

            image = self.read_image(image_path)
            gc.collect()
            label = self.read_image(label_path)

            if image is None or label is None:
                raise RuntimeError(
                    f"Failed to load patches: {image_path} or {label_path}"
                )

            image_array = sitk.GetArrayFromImage(image).astype(np.float32)
            image_tensor = torch.from_numpy(image_array)

            image = None
            image_array = None
            gc.collect()

            image_tensor = self.normalize_tensor(image_tensor)
            image_tensor = image_tensor.permute(1, 2, 0)
            image_tensor = image_tensor.unsqueeze(0)

            label_array = sitk.GetArrayFromImage(label).astype(np.float32)
            label_tensor = torch.from_numpy(label_array)

            label = None
            label_array = None
            gc.collect()

            label_tensor = self.normalize_tensor(label_tensor)
            label_tensor = label_tensor.permute(1, 2, 0)
            label_tensor = label_tensor.unsqueeze(0)

            is_valid, content_ratio = self.validate_patch_content(
                image_tensor, label_tensor
            )

            if self.skip_empty_patches and not is_valid:
                self.bad_patch_indices.add(index)
                self.total_empty_patches += 1

                if self.total_empty_patches <= 5:
                    warnings.warn(
                        f"Found empty patch (index {index}, ratio: {content_ratio:.2f}%). Using a substitute."
                    )
                elif self.total_empty_patches == 6:
                    warnings.warn(
                        f"Further warnings about empty patches will be suppressed."
                    )

                fallback_idx = self.get_fallback_index(index)
                return self.__getitem__(fallback_idx)

            if torch.isnan(image_tensor).any() or torch.isnan(label_tensor).any():
                raise ValueError("NaN values detected after processing")

            gc.collect()

            if self.using_cpu:
                image_tensor = image_tensor.to(torch.float32)
                label_tensor = label_tensor.to(torch.float32)

            image_tensor = self._ensure_contiguous_tensor(image_tensor)
            label_tensor = self._ensure_contiguous_tensor(label_tensor)

            return image_tensor, label_tensor

        except Exception as e:
            print(f"Error processing patch {index}: {str(e)}")

            if self.skip_empty_patches:
                self.bad_patch_indices.add(index)
                fallback_idx = self.get_fallback_index(index)
                return self.__getitem__(fallback_idx)

            gc.collect()
            raise

    def get_original_image_info(self, index):
        actual_index = self.indices[index]
        patch_info = self.manifest[actual_index]
        original_image_path = patch_info["original_image"]
        original_image = self.read_image(original_image_path)

        return {
            "spacing": original_image.GetSpacing(),
            "origin": original_image.GetOrigin(),
            "direction": original_image.GetDirection(),
            "size": original_image.GetSize(),
        }

    def get_bad_patches_stats(self):
        return {
            "total_indices": len(self.indices),
            "bad_patches_count": len(self.bad_patch_indices),
            "bad_patches_percentage": len(self.bad_patch_indices)
            / max(1, len(self.indices))
            * 100,
        }
