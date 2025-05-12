import datetime
import math
import os

import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm


class ModelInferencer:
    def __init__(self, model):
        self.model = model

    def _generate_patch_indices(
        self, image_shape, patch_size, stride_inplane, stride_layer, batch_size
    ):
        """Generate patch indices maintaining spatial relationships"""
        inum = (
            int(math.ceil((image_shape[0] - patch_size[0]) / float(stride_inplane))) + 1
        )
        jnum = (
            int(math.ceil((image_shape[1] - patch_size[1]) / float(stride_inplane))) + 1
        )
        knum = (
            int(math.ceil((image_shape[2] - patch_size[2]) / float(stride_layer))) + 1
        )

        patch_indices = []
        temp_indices = []
        patch_total = 0

        for k in range(knum):
            for i in range(inum):
                for j in range(jnum):
                    if patch_total % batch_size == 0:
                        if temp_indices:
                            patch_indices.append(temp_indices)
                        temp_indices = []

                    istart = min(i * stride_inplane, image_shape[0] - patch_size[0])
                    iend = istart + patch_size[0]

                    jstart = min(j * stride_inplane, image_shape[1] - patch_size[1])
                    jend = jstart + patch_size[1]

                    kstart = min(k * stride_layer, image_shape[2] - patch_size[2])
                    kend = kstart + patch_size[2]

                    temp_indices.append([istart, iend, jstart, jend, kstart, kend])
                    patch_total += 1

        if temp_indices:
            patch_indices.append(temp_indices)

        return patch_indices

    def run_inference(
        self,
        image_path,
        result_path,
        patch_size,
        stride_inplane,
        stride_layer,
        transforms=None,
        batch_size=1,
    ):
        """Run inference using patch-based approach with adaptive min-max normalization"""
        print(f"{datetime.datetime.now()}: Starting inference...")
        print(f"Using patch size: {patch_size}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(self.model, "netG"):
            self.model.netG = self.model.netG.to(device)
        elif hasattr(self.model, "cuda"):
            self.model = self.model.cuda()
        elif hasattr(self.model, "to"):
            self.model = self.model.to(device)

        if hasattr(self.model, "eval"):
            self.model.eval()

        print(f"Inference running on device: {device}")

        reader = sitk.ImageFileReader()
        reader.SetFileName(image_path)
        image = reader.Execute()

        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
        image = castImageFilter.Execute(image)

        label = sitk.Image(image.GetSize(), sitk.sitkFloat32)
        label.SetOrigin(image.GetOrigin())
        label.SetDirection(image.GetDirection())
        label.SetSpacing(image.GetSpacing())

        sample = {"image": image, "label": label}
        if transforms:
            for transform in transforms:
                sample = transform(sample)

        image_np = sitk.GetArrayFromImage(sample["image"])
        image_np = np.transpose(image_np, (2, 1, 0))

        label_np = np.zeros_like(image_np)
        weight_np = np.zeros_like(image_np)

        patch_indices = self._generate_patch_indices(
            image_np.shape, patch_size, stride_inplane, stride_layer, batch_size
        )

        patch_ranges = {}

        print(
            f"Processing {len(patch_indices)} batches with total {sum(len(batch) for batch in patch_indices)} patches"
        )

        for batch_idx, patches_batch in enumerate(tqdm(patch_indices)):
            for patch_idx in patches_batch:
                istart, iend, jstart, jend, kstart, kend = patch_idx
                patch = image_np[istart:iend, jstart:jend, kstart:kend]

                patch_min = float(np.min(patch))
                patch_max = float(np.max(patch))

                patch_key = f"{istart}_{iend}_{jstart}_{jend}_{kstart}_{kend}"
                patch_ranges[patch_key] = (patch_min, patch_max)

                eps = 1e-8
                if abs(patch_max - patch_min) < eps:
                    if batch_idx % 10 == 0:
                        print(f"Warning: Skipping constant patch at {patch_key}")
                    continue

                normalized_patch = (patch - patch_min) / (patch_max - patch_min)
                normalized_patch = normalized_patch * 2 - 1

                patch_tensor = torch.from_numpy(
                    normalized_patch[np.newaxis, np.newaxis, :, :, :]
                )
                patch_tensor = patch_tensor.to(device, dtype=torch.float32)

                self.model.set_input(patch_tensor)
                self.model.test()

                visuals = self.model.get_current_visuals()
                pred = visuals[
                    "fake_A" if self.model.opt.which_direction == "BtoA" else "fake_B"
                ]
                pred = pred.squeeze().data.cpu().numpy()

                output_shape = label_np[istart:iend, jstart:jend, kstart:kend].shape
                pred_slice = pred
                if pred.shape[2] != output_shape[2]:
                    pred_slice = pred[:, :, : output_shape[2]]

                pred_slice = (pred_slice + 1) / 2
                pred_slice = pred_slice * (patch_max - patch_min) + patch_min

                label_np[istart:iend, jstart:jend, kstart:kend] += pred_slice
                weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        epsilon = 1e-8
        weight_np = np.where(weight_np > 0, weight_np, epsilon)
        label_np = label_np / weight_np

        label = self.from_numpy_to_itk(label_np, image)

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(result_path)
        writer.Execute(label)
        print(f"{datetime.datetime.now()}: Saved result to {result_path}")

        patch_ranges = None
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def from_numpy_to_itk(self, image_np, reference_image):
        """Convert numpy array to ITK image."""
        image_np = np.transpose(image_np, (2, 1, 0))
        image = sitk.GetImageFromArray(image_np)
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())
        image.SetSpacing(reference_image.GetSpacing())
        return image
