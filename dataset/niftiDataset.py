import gc
import os
import random
import time as time_module

import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data

from dataset.processing.augmentation.augmentation import Augmentation
from dataset.processing.random_crop import RandomCrop
from utils.utils import lstFiles


class NifitDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        which_direction="AtoB",
        transforms=None,
        shuffle_labels=False,
        train=False,
        test=False,
        is_inference=False,
        max_cache_size_mb=512,
    ):
        self.data_path = data_path
        self.which_direction = which_direction
        self.transforms = transforms
        self.shuffle_labels = shuffle_labels
        self.train = train
        self.test = test
        self.is_inference = is_inference
        self.bit = sitk.sitkFloat32

        self.using_cpu = not torch.cuda.is_available()
        if self.using_cpu:
            max_cache_size_mb = min(256, max_cache_size_mb)
            print(f"CPU mode detected: Limiting cache to {max_cache_size_mb}MB")

        if is_inference:
            self.images_list = [data_path] if isinstance(data_path, str) else data_path
            self.labels_list = [None] * len(self.images_list)
        else:
            img_path = os.path.join(data_path, "invivo")
            label_path = os.path.join(data_path, "exvivo")

            self.images_list = lstFiles(img_path)
            self.labels_list = lstFiles(label_path)
            print(
                f"Found {len(self.images_list)} images and {len(self.labels_list)} labels"
            )

        self.images_size = len(self.images_list)
        self.labels_size = len(self.labels_list)

        self.max_cache_size_mb = max_cache_size_mb
        self.current_cache_size_mb = 0
        self.cached_items = {}
        self.cached_items_size = {}
        self.access_count = {}
        self.last_accessed = {}

        self.patches_per_image = 1
        if transforms:
            for transform in transforms:
                if isinstance(transform, RandomCrop):
                    self.patches_per_image = transform.patches_per_image
                    break

    def read_image(self, path):
        if path in self.cached_items:
            self.access_count[path] = self.access_count.get(path, 0) + 1
            self.last_accessed[path] = time_module.time()
            return self.cached_items[path]

        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.SetLoadPrivateTags(False)
            image = reader.Execute()
            result = self.preprocess_image(image)

            size_mb = (np.prod(image.GetSize()) * 4) / (1024 * 1024)

            if size_mb > self.max_cache_size_mb * 0.4 or (
                self.using_cpu and size_mb > 50
            ):
                return result

            current_time = time_module.time()
            if self.cached_items:
                cache_items = [
                    (k, self.last_accessed.get(k, 0)) for k in self.cached_items.keys()
                ]
                cache_items.sort(key=lambda x: x[1])

                while (
                    self.current_cache_size_mb + size_mb > self.max_cache_size_mb
                    and cache_items
                ):
                    oldest_key = cache_items.pop(0)[0]
                    if oldest_key in self.cached_items_size:
                        self.current_cache_size_mb -= self.cached_items_size[oldest_key]
                    self._remove_from_cache(oldest_key)

            self.cached_items[path] = result
            self.cached_items_size[path] = size_mb
            self.current_cache_size_mb += size_mb
            self.access_count[path] = 1
            self.last_accessed[path] = current_time

            return result
        except Exception as e:
            print(f"Error reading image {path}: {e}")
            gc.collect()
            raise

    def preprocess_image(self, image):
        if image.GetPixelID() != sitk.sitkFloat32:
            cast_filter = sitk.CastImageFilter()
            cast_filter.SetOutputPixelType(sitk.sitkFloat32)
            image = cast_filter.Execute(image)
        return image

    def __len__(self):
        return self.images_size * self.patches_per_image

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

    def _remove_from_cache(self, key):
        if key in self.cached_items:
            del self.cached_items[key]
        if key in self.cached_items_size:
            del self.cached_items_size[key]
        if key in self.access_count:
            del self.access_count[key]
        if key in self.last_accessed:
            del self.last_accessed[key]

    def __getitem__(self, index):
        try:
            image_index = index // self.patches_per_image
            patch_index = index % self.patches_per_image

            image_path = self.images_list[image_index]
            if self.shuffle_labels:
                label_index = random.randint(0, self.labels_size - 1)
                label_path = self.labels_list[label_index]
            else:
                label_path = self.labels_list[image_index]

            image = self.read_image(image_path)
            gc.collect()
            label = self.read_image(label_path)

            if image is None or label is None:
                raise RuntimeError(
                    f"Failed to load images: {image_path} or {label_path}"
                )

            sample = {"image": image, "label": label}

            if self.transforms:
                for t in self.transforms:
                    try:
                        if isinstance(t, RandomCrop):
                            sample = t.extract_center_biased_patch(sample, patch_index)
                        else:
                            sample = t(sample)

                        if isinstance(t, (RandomCrop, Augmentation)):
                            gc.collect()

                    except Exception as e:
                        print(f"Transform error in {t.__class__.__name__}: {str(e)}")
                        raise

            image_array = sitk.GetArrayFromImage(sample["image"]).astype(np.float32)
            image_tensor = torch.from_numpy(image_array)

            sample["image"] = None
            image_array = None
            gc.collect()

            image_tensor = self.normalize_tensor(image_tensor)
            image_tensor = image_tensor.permute(1, 2, 0)
            image_tensor = image_tensor.unsqueeze(0)

            label_array = sitk.GetArrayFromImage(sample["label"]).astype(np.float32)
            label_tensor = torch.from_numpy(label_array)

            sample["label"] = None
            label_array = None
            gc.collect()

            label_tensor = self.normalize_tensor(label_tensor)
            label_tensor = label_tensor.permute(1, 2, 0)
            label_tensor = label_tensor.unsqueeze(0)

            if torch.isnan(image_tensor).any() or torch.isnan(label_tensor).any():
                raise ValueError("NaN values detected after processing")

            gc.collect()

            if self.using_cpu:
                image_tensor = image_tensor.to(torch.float32)
                label_tensor = label_tensor.to(torch.float32)

            return image_tensor, label_tensor

        except Exception as e:
            print(f"Error processing item {index}: {str(e)}")
            gc.collect()
            raise

    def get_original_image_info(self, index):
        image_index = index // self.patches_per_image
        image = self.read_image(self.images_list[image_index])
        return {
            "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(),
            "direction": image.GetDirection(),
            "size": image.GetSize(),
        }
