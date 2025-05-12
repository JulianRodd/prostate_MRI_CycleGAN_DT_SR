import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

from metrics.prostateMRIFeatureMetrics import ProstateMRIFeatureMetrics


class NCC:
    """
    Normalized Cross-Correlation (NCC) calculator for measuring similarity between image volumes.

    Computes the NCC between two image volumes, with optional resizing for memory efficiency.
    NCC is a measure of structural similarity that ranges from -1 (completely anti-correlated)
    to 1 (perfectly correlated).

    Args:
        device (str): Device for computation ('cpu' or 'cuda'). Default: 'cpu'
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.max_size = 16 if device == "cpu" else 256

    def _resize_if_needed(self, volume):
        """
        Resize volume if dimensions exceed maximum size limit for memory efficiency.

        Args:
            volume (torch.Tensor): Input volume tensor

        Returns:
            torch.Tensor: Resized volume if needed, otherwise original volume
        """
        if volume.dim() == 5:
            _, _, d, h, w = volume.shape

            if d > self.max_size or h > self.max_size or w > self.max_size:
                factor = min(self.max_size / max(d, h, w), 1.0)
                new_d = max(4, int(d * factor))
                new_h = max(16, int(h * factor))
                new_w = max(16, int(w * factor))

                return F.interpolate(
                    volume,
                    size=(new_d, new_h, new_w),
                    mode="trilinear",
                    align_corners=False,
                )

        return volume

    def calculate(self, img1, img2):
        """
        Calculate normalized cross-correlation between two image volumes.

        Args:
            img1 (torch.Tensor): First image volume
            img2 (torch.Tensor): Second image volume

        Returns:
            float: NCC value between -1 and 1
        """
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        img1 = self._resize_if_needed(img1)
        img2 = self._resize_if_needed(img2)

        if img1.dim() == 5:
            b = img1.size(0)
            img1_flat = img1.reshape(b, -1)
            img2_flat = img2.reshape(b, -1)
        else:
            b = img1.size(0)
            img1_flat = img1.reshape(b, -1)
            img2_flat = img2.reshape(b, -1)

        img1_mean = img1_flat.mean(dim=1, keepdim=True)
        img2_mean = img2_flat.mean(dim=1, keepdim=True)

        img1_centered = img1_flat - img1_mean
        img2_centered = img2_flat - img2_mean

        numerator = (img1_centered * img2_centered).sum(dim=1)

        img1_std = torch.sqrt((img1_centered**2).sum(dim=1))
        img2_std = torch.sqrt((img2_centered**2).sum(dim=1))

        denominator = img1_std * img2_std
        denominator = torch.clamp(denominator, min=1e-8)

        ncc_per_batch = numerator / denominator

        return ncc_per_batch.mean()


class FID:
    """
    Fréchet Inception Distance (FID) calculator optimized for medical imaging volumes.

    This implementation is specifically adapted for prostate MRI data, with options
    for slice-based processing and automatic filtering of irrelevant slices.

    Args:
        device (str): Device for computation ('cpu' or 'cuda'). Default: 'cpu'
        slice_sampling (bool): Whether to sample slices for large volumes. Default: True
        max_size (int): Maximum dimension size for resizing. Default: 1024
        max_slices (int): Maximum number of slices to process. Default: 128
        batch_size (int): Batch size for feature extraction. Default: 4
        max_feature_dim (int): Maximum feature dimension. Default: 1024
    """

    def __init__(
        self,
        device="cpu",
        slice_sampling=True,
        max_size=1024,
        max_slices=128,
        batch_size=4,
        max_feature_dim=1024,
    ):
        self.device = device

        # Optimize parameters for GPU
        if device != "cpu":
            self.max_size = 512
            self.slice_sampling = False
            self.max_slices = 256
            self.batch_size = 16  # Increased batch size for GPU
            self.max_feature_dim = 8192
        else:
            self.max_size = max_size
            self.slice_sampling = slice_sampling
            self.max_slices = max_slices
            self.batch_size = batch_size
            self.max_feature_dim = max_feature_dim

        self.feature_extractor = None

        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.total_slices_processed = 0
        self.accepted_slices = 0
        self._cached_indices = {}

    def _ensure_feature_extractor(self):
        """Initialize feature extractor if not already created"""
        if self.feature_extractor is None:
            self.feature_extractor = ProstateMRIFeatureMetrics(
                device=self.device,
                use_layers=[
                    "model.submodule.encoder5",
                ],
                layer_weights={
                    "model.submodule.encoder5": 1.0,
                },
            )

    def _resize_if_needed(self, volume):
        """
        Resize volume if dimensions exceed maximum size limit for memory efficiency.

        Args:
            volume (torch.Tensor): Input volume tensor

        Returns:
            torch.Tensor: Resized volume with sampled slices
        """
        _, _, d, h, w = volume.shape

        volume = self._sample_slices(volume)
        _, _, d, h, w = volume.shape

        if d > self.max_size or h > self.max_size or w > self.max_size:
            d_factor = min(1.0, self.max_size / d)
            h_factor = min(1.0, self.max_size / h)
            w_factor = min(1.0, self.max_size / w)

            factor = min(d_factor, h_factor, w_factor)

            new_d = max(4, int(d * factor))
            new_h = max(16, int(h * factor))
            new_w = max(16, int(w * factor))

            mode = "trilinear"
            align_corners = True

            return F.interpolate(
                volume,
                size=(new_d, new_h, new_w),
                mode=mode,
                align_corners=align_corners,
            )

        return volume

    def _sample_slices(self, volume):
        """
        Sample a subset of slices from volume to reduce computation.

        Args:
            volume (torch.Tensor): Input volume tensor [B,C,D,H,W]

        Returns:
            torch.Tensor: Volume with sampled slices
        """
        _, _, d, h, w = volume.shape

        if not self.slice_sampling or d <= self.max_slices or self.device != "cpu":
            return volume

        indices = np.linspace(0, d - 1, self.max_slices, dtype=int)
        indices = [int(i) for i in indices]

        slices = [volume[:, :, i : i + 1, :, :] for i in indices]
        return torch.cat(slices, dim=2)

    def process_2d_images(self, images):
        """
        Process 2D image slices for FID calculation with resizing if needed.

        Args:
            images (torch.Tensor): Input 2D images [B,C,H,W]

        Returns:
            torch.Tensor: Processed images
        """
        if images.dim() == 4:
            batch_size, channels, height, width = images.shape

            if height > self.max_size or width > self.max_size:
                h_factor = min(1.0, self.max_size / height)
                w_factor = min(1.0, self.max_size / width)
                factor = min(h_factor, w_factor)

                new_h = max(16, int(height * factor))
                new_w = max(16, int(width * factor))

                return F.interpolate(
                    images,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=True,
                )

        return images

    def _combine_layer_features(self, features_dict):
        """
        Combine features from different layers with dimensionality reduction.

        Args:
            features_dict (dict): Dictionary of features from different layers

        Returns:
            torch.Tensor: Combined feature tensor
        """
        all_features = []

        for layer_name, weight in self.feature_extractor.layer_weights.items():
            if layer_name in features_dict:
                feature_tensor = features_dict[layer_name]

                flat_features = feature_tensor.view(feature_tensor.size(0), -1)
                orig_dim = flat_features.size(1)

                if layer_name == list(self.feature_extractor.layer_weights.keys())[0]:
                    print(
                        f"Original feature dimension: {orig_dim}, max allowed: {self.max_feature_dim}"
                    )

                if orig_dim > self.max_feature_dim:
                    # Use a consistent random seed for reproducibility
                    if (
                        hasattr(self, "_cached_indices")
                        and self._cached_indices.get(orig_dim) is not None
                    ):
                        indices = self._cached_indices[orig_dim]
                    else:
                        # Use a more structured approach - sample evenly across the feature dimension
                        # This preserves more information than pure random sampling
                        step = orig_dim / self.max_feature_dim
                        indices = torch.floor(
                            torch.arange(0, orig_dim, step, device=flat_features.device)
                        ).long()
                        indices = indices[: self.max_feature_dim]

                        # Make sure indices are unique and sorted
                        indices = torch.unique(indices)
                        indices, _ = torch.sort(indices)

                        # If we still don't have enough indices, add some randomly
                        if len(indices) < self.max_feature_dim:
                            remaining = self.max_feature_dim - len(indices)
                            mask = torch.ones(
                                orig_dim, device=flat_features.device
                            ).bool()
                            mask[indices] = False
                            remaining_indices = torch.nonzero(mask).squeeze()

                            if len(remaining_indices) > 0:
                                # Use a fixed seed for reproducibility
                                g = torch.Generator(device=flat_features.device)
                                g.manual_seed(42)
                                perm = torch.randperm(
                                    len(remaining_indices),
                                    generator=g,
                                    device=flat_features.device,
                                )
                                random_indices = remaining_indices[perm[:remaining]]
                                indices = torch.cat([indices, random_indices])
                                indices, _ = torch.sort(indices)

                        if not hasattr(self, "_cached_indices"):
                            self._cached_indices = {}
                        self._cached_indices[orig_dim] = indices

                    flat_features = flat_features[:, indices]

                # Normalize features
                mean = flat_features.mean(dim=1, keepdim=True)
                std = flat_features.std(dim=1, keepdim=True) + 1e-8
                normalized = (flat_features - mean) / std * weight

                all_features.append(normalized)

        if not all_features:
            feature_dim = 10 if self.device == "cpu" else 100
            return torch.zeros(1, feature_dim, device=self.device)

        result = torch.cat(all_features, dim=1)
        return result

    def extract_features_slice_based(self, images):
        """
        Extract features from all 2D image slices with optimized handling for prostate MRI.

        Args:
            images (torch.Tensor): Input image volume [B,C,D,H,W] or [B,C,H,W]

        Returns:
            torch.Tensor: Extracted features
        """
        self._ensure_feature_extractor()

        total_slices = 0
        accepted_slices = 0
        # Reduced threshold to keep more slices with prostate tissue
        min_content_percentage = 0.001  # 0.1% rather than 1%

        if images.dim() == 5:
            batch_size, channels, depth, height, width = images.shape
            print(
                f"Original volume shape: [B={batch_size}, C={channels}, D={depth}, H={height}, W={width}]"
            )

            # Determine which dimension has the most appropriate slices for prostate MRI
            # For prostate MRI, typically the axial view is most informative
            # This is usually the smallest dimension or the one with spacing closest to 3mm
            slice_dimensions = [(depth, 2, "D"), (height, 3, "H"), (width, 4, "W")]
            # Sort by dimension size (smallest first, as it's often the through-plane direction)
            slice_dimensions.sort(key=lambda x: x[0])

            # By default, use the smallest dimension for slicing (usually axial for prostate)
            slice_dim_size, slice_dim_idx, slice_dim_name = slice_dimensions[0]
            print(
                f"Auto-selected slicing along {slice_dim_name} dimension ({slice_dim_size} slices)"
            )

            all_slices = []

            # Create a more flexible slicing approach that can slice along any dimension
            for b in range(batch_size):
                for slice_idx in range(slice_dim_size):
                    total_slices += 1

                    # Dynamic slicing based on the determined dimension
                    if slice_dim_idx == 2:  # D dimension
                        slice_img = images[
                            b : b + 1, :, slice_idx : slice_idx + 1, :, :
                        ].squeeze(2)
                    elif slice_dim_idx == 3:  # H dimension
                        slice_img = images[
                            b : b + 1, :, :, slice_idx : slice_idx + 1, :
                        ].squeeze(3)
                    elif slice_dim_idx == 4:  # W dimension
                        slice_img = images[
                            b : b + 1, :, :, :, slice_idx : slice_idx + 1
                        ].squeeze(4)

                    # Check for invalid values
                    has_nan = torch.isnan(slice_img).any()
                    has_inf = torch.isinf(slice_img).any()

                    # Check for content (less strict than before)
                    non_background_percentage = (
                        (slice_img > -0.95).float().mean()
                    ).item()

                    # Much more lenient filtering to preserve more slices
                    if has_nan or has_inf:
                        # Only skip slices with actual invalid values
                        if has_nan:
                            print(f"  Skipping slice {slice_idx} due to NaN values")
                        if has_inf:
                            print(f"  Skipping slice {slice_idx} due to Inf values")
                        continue

                    # Almost no content filtering - just skip completely empty slices
                    if non_background_percentage < min_content_percentage:
                        print(
                            f"  Skipping slice {slice_idx} with only {non_background_percentage * 100:.4f}% non-background"
                        )
                        continue

                    all_slices.append(slice_img)
                    accepted_slices += 1

                    # For very large volumes, sample systematically to avoid memory issues
                    # while still ensuring enough slices for reliable FID
                    if (
                        slice_dim_size > 200
                        and accepted_slices >= 100
                        and slice_idx % 2 == 0
                    ):
                        continue

            self.total_slices_processed += total_slices
            self.accepted_slices += accepted_slices

            print(f"Slice statistics:")
            print(f"  - Total slices: {total_slices}")
            print(
                f"  - Accepted: {accepted_slices} ({accepted_slices / max(total_slices, 1) * 100:.1f}%)"
            )

            if all_slices:
                images = torch.cat(all_slices, dim=0)
                if len(all_slices) < 50:
                    print(
                        f"Warning: Only {len(all_slices)} slices accepted, recommended minimum is 50 for reliable FID"
                    )
                    print(f"Will proceed with available slices anyway")
            else:
                print(
                    f"Warning: No valid slices extracted. Falling back to a single representative slice."
                )
                # Create a fallback slice from the middle of the volume
                mid_d = depth // 2
                mid_h = height // 2
                mid_w = width // 2

                if slice_dim_idx == 2:  # D dimension
                    fallback_slice = images[0:1, :, mid_d : mid_d + 1, :, :].squeeze(2)
                elif slice_dim_idx == 3:  # H dimension
                    fallback_slice = images[0:1, :, :, mid_h : mid_h + 1, :].squeeze(3)
                else:  # W dimension
                    fallback_slice = images[0:1, :, :, :, mid_w : mid_w + 1].squeeze(4)

                return fallback_slice.to(self.device)
        else:
            # Handle 2D images (already slices)
            filtered_slices = []
            for i in range(images.size(0)):
                total_slices += 1
                slice_img = images[i : i + 1]

                has_nan = torch.isnan(slice_img).any()
                has_inf = torch.isinf(slice_img).any()

                if has_nan or has_inf:
                    continue

                filtered_slices.append(slice_img)
                accepted_slices += 1

            if filtered_slices:
                images = torch.cat(filtered_slices, dim=0)
            else:
                print(f"Warning: No valid slices after filtering {total_slices} slices")
                return torch.zeros(1, 10, device=self.device)

        # Process 2D images and extract features in batches
        images = self.process_2d_images(images)
        batch_size = images.size(0)
        batch_per_pass = self.batch_size
        features_list = []

        # Track how many features we successfully extract
        features_extracted = 0
        print(
            f"Extracting features from {batch_size} slices in batches of {batch_per_pass}"
        )

        with torch.no_grad():
            for i in range(0, batch_size, batch_per_pass):
                end_idx = min(i + batch_per_pass, batch_size)
                batch = images[i:end_idx]

                if i % 50 == 0:
                    print(f"Processing batch {i}/{batch_size}")

                if batch.dim() == 4:
                    batch = batch.unsqueeze(2)

                try:
                    batch_features = self.feature_extractor.extract_features(batch)
                    combined_features = self._combine_layer_features(batch_features)
                    # Keep features on current device instead of moving to CPU
                    features_list.append(combined_features)
                    features_extracted += end_idx - i
                except Exception as e:
                    print(f"Error extracting features for batch {i}:{end_idx}: {e}")
                    # Continue with next batch

                if self.device != "cpu":
                    torch.cuda.empty_cache()

        print(
            f"Successfully extracted features from {features_extracted}/{batch_size} slices"
        )

        if not features_list:
            print("Warning: No features were successfully extracted")
            return torch.zeros(1, 10, device=self.device)

        all_features = torch.cat(features_list, dim=0)
        print(f"Final features shape: {all_features.shape}")

        return all_features

    def calculate_slice_based(self, real_images, generated_images):
        """
        GPU-optimized FID calculation between real and generated image volumes using slice-based approach.

        Args:
            real_images (torch.Tensor): Real image volume
            generated_images (torch.Tensor): Generated image volume

        Returns:
            float: FID score (lower is better)
        """
        real_images = real_images.to(self.device)
        generated_images = generated_images.to(self.device)

        self.total_slices_processed = 0
        self.accepted_slices = 0

        print("Extracting features for real image slices...")
        real_features = self.extract_features_slice_based(real_images)
        real_slices_accepted = self.accepted_slices
        print(f"Real features extracted, shape: {real_features.shape}")
        print(f"Real slices accepted for FID: {real_slices_accepted}")

        self.total_slices_processed = 0
        self.accepted_slices = 0

        print("Extracting features for generated image slices...")
        generated_features = self.extract_features_slice_based(generated_images)
        generated_slices_accepted = self.accepted_slices
        print(f"Generated features extracted, shape: {generated_features.shape}")
        print(f"Generated slices accepted for FID: {generated_slices_accepted}")

        real_slices_for_fid = real_features.shape[0]
        generated_slices_for_fid = generated_features.shape[0]

        print(f"\nFinal slice count summary:")
        print(f"  - Real slices accepted: {real_slices_for_fid}")
        print(f"  - Generated slices accepted: {generated_slices_for_fid}")
        print(
            f"  - Total slices for FID calculation: {real_slices_for_fid + generated_slices_for_fid}"
        )
        print(f"  - Feature dimension: {real_features.shape[1]}")

        min_required_slices = 50
        if (
            real_slices_for_fid < min_required_slices
            or generated_slices_for_fid < min_required_slices
        ):
            print(f"Warning: Not enough slices for reliable FID calculation!")
            print(f"  - Minimum recommended: {min_required_slices} slices per domain")
            print(f"  - Will proceed with available slices anyway")

            if real_slices_for_fid < 2 or generated_slices_for_fid < 2:
                print(
                    "Error: At least 2 slices per domain are required for covariance calculation"
                )
                return float("inf")

        if real_features.shape[0] == 0 or generated_features.shape[0] == 0:
            print("Error: Zero features extracted for at least one domain")
            return float("inf")

        # KEY OPTIMIZATION: Keep calculations on GPU if available
        if self.device != "cpu":
            print(f"Calculating FID directly on {self.device}...")
            return self._calculate_frechet_distance_torch(
                real_features, generated_features
            )
        else:
            # For CPU, use the original numpy implementation for backwards compatibility
            real_features_np = real_features.cpu().numpy()
            generated_features_np = generated_features.cpu().numpy()

            # Calculate mean and covariance matrices
            mu1 = np.mean(real_features_np, axis=0)
            mu2 = np.mean(generated_features_np, axis=0)
            sigma1 = np.cov(real_features_np, rowvar=False)
            sigma2 = np.cov(generated_features_np, rowvar=False)

            # Calculate FID using the original function
            return self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def _calculate_frechet_distance_torch(
        self,
        real_features: torch.Tensor,
        generated_features: torch.Tensor,
        eps: float = 1e-6,
    ) -> float:
        """
        Memory-efficient Frechet Distance calculation using PyTorch.

        Uses eigenvalues for better memory efficiency.

        Args:
            real_features (torch.Tensor): Features from real images
            generated_features (torch.Tensor): Features from generated images
            eps (float): Small epsilon value for numerical stability

        Returns:
            float: FID score (lower is better)
        """
        with torch.no_grad():
            # 1) Means
            mu1 = real_features.mean(dim=0)
            mu2 = generated_features.mean(dim=0)
            diff = mu1 - mu2
            mean_term = diff.dot(diff)  # ||mu1 - mu2||^2

            # 2) Covariances (unbiased)
            def cov(x: torch.Tensor, mu: torch.Tensor):
                x_centered = x - mu.unsqueeze(0)
                n = x.size(0)
                # no grads, so this won't keep buffers
                return (x_centered.t() @ x_centered) / max(n - 1, 1)

            sigma1 = cov(real_features, mu1)
            sigma2 = cov(generated_features, mu2)

            # 3) Regularize & cast to double if needed
            d = sigma1.size(0)
            I = torch.eye(d, device=self.device, dtype=torch.double)
            sigma1 = (sigma1 + eps * I).to(torch.double)
            sigma2 = (sigma2 + eps * I).to(torch.double)

            # 4) Trace terms
            trace1 = sigma1.trace()
            trace2 = sigma2.trace()

            # 5) Compute Tr[(σ1 σ2)^(1/2)] - FIXED: avoid in-place operation
            # Create product matrix and symmetrize it without in-place ops
            prod = sigma1.matmul(sigma2)
            prod = 0.5 * (
                prod + prod.transpose(0, 1)
            )  # Changed to avoid in-place operation

            try:
                # Only eigenvalues → lower memory
                eigvals = torch.linalg.eigvalsh(prod, UPLO="U")
                eigvals = eigvals.clamp(min=0.0)  # Changed to non-in-place clamp
                trace_sqrt = torch.sqrt(eigvals).sum()
            except RuntimeError:
                # Only singular values → no full SVD factors
                svals = torch.linalg.svdvals(prod)
                trace_sqrt = torch.sqrt(svals.clamp(min=0.0)).sum()

            # 6) Final FID (clamped to ≥0)
            fid = mean_term.to(torch.double) + trace1 + trace2 - 2.0 * trace_sqrt
            return float(fid.clamp(min=0.0).item())

    def extract_features(self, images):
        """
        Extract features from image volumes without background filtering.

        Args:
            images (torch.Tensor): Input image volume

        Returns:
            torch.Tensor: Extracted features
        """
        self._ensure_feature_extractor()

        if images.dim() == 4:
            images = images.unsqueeze(2)

        filtered_volumes = []
        total_volumes = images.size(0)
        accepted_volumes = 0
        rejected_volumes = 0

        for i in range(total_volumes):
            volume = images[i : i + 1]

            has_nan = torch.isnan(volume).any()
            has_inf = torch.isinf(volume).any()

            if has_nan or has_inf:
                rejected_volumes += 1
            else:
                filtered_volumes.append(volume)
                accepted_volumes += 1

        if filtered_volumes:
            images = torch.cat(filtered_volumes, dim=0)
        else:
            if rejected_volumes > 0:
                print(
                    f"Warning: All {total_volumes} volumes were rejected due to invalid values"
                )
                print("Using original volumes without filtering to avoid failure")

        images = self._sample_slices(images)
        images = self._resize_if_needed(images)

        batch_size = images.size(0)
        batch_per_pass = self.batch_size if self.device != "cpu" else 1
        features_list = []

        with torch.no_grad():
            for i in range(0, batch_size, batch_per_pass):
                end_idx = min(i + batch_per_pass, batch_size)
                batch = images[i:end_idx]

                batch_features = self.feature_extractor.extract_features(batch)
                combined_features = self._combine_layer_features(batch_features)
                # Keep features on current device
                features_list.append(combined_features)

                if self.device != "cpu":
                    torch.cuda.empty_cache()

        all_features = torch.cat(features_list, dim=0) if features_list else None

        if all_features is None:
            print("Warning: No valid features were extracted")
            return torch.zeros(batch_size, 10, device=self.device)

        return all_features

    def calculate(self, real_images, generated_images):
        """
        Calculate FID between real and generated images without background filtering.

        Args:
            real_images (torch.Tensor): Real image volume
            generated_images (torch.Tensor): Generated image volume

        Returns:
            float: FID score (lower is better)
        """
        real_images = real_images.to(self.device)
        generated_images = generated_images.to(self.device)

        real_features = self.extract_features(real_images)
        generated_features = self.extract_features(generated_images)

        # Use the GPU optimized calculation if on GPU
        if self.device != "cpu":
            return self._calculate_frechet_distance_torch(
                real_features, generated_features
            )
        else:
            # For CPU, use original numpy implementation for backwards compatibility
            real_features_np = real_features.cpu().numpy()
            generated_features_np = generated_features.cpu().numpy()

            mu1 = np.mean(real_features_np, axis=0)
            mu2 = np.mean(generated_features_np, axis=0)

            if real_features_np.shape[0] == 1:
                rng = np.random.RandomState(42)
                real_features_np = np.repeat(real_features_np, 2, axis=0)
                real_features_np[1] += rng.normal(0, 1e-5, real_features_np[1].shape)

            if generated_features_np.shape[0] == 1:
                rng = np.random.RandomState(42)
                generated_features_np = np.repeat(generated_features_np, 2, axis=0)
                generated_features_np[1] += rng.normal(
                    0, 1e-5, generated_features_np[1].shape
                )

            if real_features_np.shape[1] > 512:
                rng = np.random.RandomState(42)
                indices = rng.choice(real_features_np.shape[1], 512, replace=False)

                indices = np.sort(indices)

                real_features_np = real_features_np[:, indices]
                generated_features_np = generated_features_np[:, indices]
                mu1 = mu1[indices]
                mu2 = mu2[indices]

            sigma1 = np.cov(real_features_np, rowvar=False)
            sigma2 = np.cov(generated_features_np, rowvar=False)

            return self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Calculate Frechet distance between two multivariate Gaussians with improved numerical stability.

        Args:
            mu1 (numpy.ndarray): Mean of first distribution
            sigma1 (numpy.ndarray): Covariance of first distribution
            mu2 (numpy.ndarray): Mean of second distribution
            sigma2 (numpy.ndarray): Covariance of second distribution
            eps (float): Small epsilon value for numerical stability

        Returns:
            float: FID score (lower is better)
        """
        diff = mu1 - mu2
        mean_diff_squared = np.sum(diff**2)

        # Add regularization to ensure positive definiteness
        feature_dim = sigma1.shape[0]
        adaptive_eps = eps * (1.0 + np.log1p(feature_dim) * 0.1)
        reg_sigma1 = sigma1 + np.eye(feature_dim) * adaptive_eps
        reg_sigma2 = sigma2 + np.eye(feature_dim) * adaptive_eps

        try:
            # Direct matrix square root calculation (standard approach)
            covmean, _ = scipy.linalg.sqrtm(reg_sigma1 @ reg_sigma2, disp=False)

            if np.iscomplexobj(covmean):
                imag_magnitude = np.max(np.abs(covmean.imag))
                if imag_magnitude < 1e-3:
                    # Negligible imaginary component, safe to take real part
                    covmean = covmean.real
                else:
                    print(
                        f"Warning: Large imaginary component in matrix sqrt: {imag_magnitude}"
                    )
                    # If imaginary component is large, there might be a numerical issue
                    # Add stronger regularization and retry
                    reg_sigma1 = sigma1 + np.eye(feature_dim) * (adaptive_eps * 10)
                    reg_sigma2 = sigma2 + np.eye(feature_dim) * (adaptive_eps * 10)
                    covmean, _ = scipy.linalg.sqrtm(reg_sigma1 @ reg_sigma2, disp=False)
                    covmean = covmean.real

            tr_covmean = np.trace(covmean)
            fid = (
                mean_diff_squared + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
            )

        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Error in FID calculation: {e}")
            print("Using approximation without sqrt term")
            # Fallback to a rough approximation
            fid = mean_diff_squared + np.trace(sigma1) + np.trace(sigma2)

        return float(fid)


class DomainMetricsCalculator:
    """
    Unified calculator for domain translation metrics.

    Includes FID and NCC metrics optimized for comparing image domains in
    medical imaging applications.

    Args:
        device (str): Device for computation ('cpu' or 'cuda'). Default: 'cpu'
        fid_max_size (int): Maximum dimension size for FID calculation. Default: 1024
        fid_max_slices (int): Maximum slices for FID calculation. Default: 128
        batch_size (int): Batch size for feature extraction. Default: 4
    """

    def __init__(
        self,
        device="cpu",
        fid_max_size=1024,
        fid_max_slices=128,
        batch_size=4,
    ):
        self.device = device
        self.fid_calculator = None
        self.ncc_calculator = None

        # Improve device configuration with better defaults for GPU
        if device != "cpu":
            self.fid_max_size = 512
            self.fid_max_slices = 256
            self.batch_size = 16  # Increased batch size for GPU
            self.max_feature_dim = 8192
        else:
            self.fid_max_size = fid_max_size
            self.fid_max_slices = fid_max_slices
            self.batch_size = batch_size
            self.max_feature_dim = 512

        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def _ensure_fid_calculator(self):
        """Initialize FID calculator if not already created"""
        if self.fid_calculator is None:
            self.fid_calculator = FID(
                device=self.device,
                slice_sampling=True,
                max_size=self.fid_max_size,
                max_slices=self.fid_max_slices,
                batch_size=self.batch_size,
                max_feature_dim=self.max_feature_dim,
            )

    def _ensure_ncc_calculator(self):
        """Initialize NCC calculator if not already created"""
        if self.ncc_calculator is None:
            self.ncc_calculator = NCC(device=self.device)

    def calculate_slice_based_fid(self, real_images, gen_images):
        """
        Calculate FID using a slice-based approach for 3D volumes.

        Args:
            real_images (torch.Tensor): Real image volume
            gen_images (torch.Tensor): Generated image volume

        Returns:
            float: FID score (lower is better)
        """
        self._ensure_fid_calculator()
        try:
            fid_value = self.fid_calculator.calculate_slice_based(
                real_images, gen_images
            )

            if fid_value == float("inf"):
                print(
                    "Warning: Attempting manual FID calculation based on feature shapes"
                )
                with torch.no_grad():
                    real_features = self.fid_calculator.extract_features_slice_based(
                        real_images
                    )
                    gen_features = self.fid_calculator.extract_features_slice_based(
                        gen_images
                    )

                    real_count = real_features.shape[0]
                    gen_count = gen_features.shape[0]

                    if real_count >= 2 and gen_count >= 2:
                        # Use optimized PyTorch calculation if on GPU
                        if self.device != "cpu":
                            manual_fid = (
                                self.fid_calculator._calculate_frechet_distance_torch(
                                    real_features, gen_features
                                )
                            )
                        else:
                            real_np = real_features.numpy()
                            gen_np = gen_features.numpy()
                            mu1 = np.mean(real_np, axis=0)
                            mu2 = np.mean(gen_np, axis=0)
                            sigma1 = np.cov(real_np, rowvar=False)
                            sigma2 = np.cov(gen_np, rowvar=False)
                            manual_fid = (
                                self.fid_calculator._calculate_frechet_distance(
                                    mu1, sigma1, mu2, sigma2
                                )
                            )
                        return manual_fid

            return fid_value
        except Exception as e:
            print(f"Error calculating slice-based FID: {e}")
            import traceback

            traceback.print_exc()
            return float("inf")

    def calculate_fid(self, real_images, gen_images):
        """
        Calculate FID between real and generated image volumes.

        Args:
            real_images (torch.Tensor): Real image volume
            gen_images (torch.Tensor): Generated image volume

        Returns:
            float: FID score (lower is better)
        """
        self._ensure_fid_calculator()
        try:
            fid_value = self.fid_calculator.calculate(real_images, gen_images)
            return fid_value
        except Exception as e:
            print(f"Error calculating FID: {e}")
            import traceback

            traceback.print_exc()
            return float("nan")

    def calculate_ncc(self, real_images, gen_images):
        """
        Calculate NCC between real and generated image volumes.

        Args:
            real_images (torch.Tensor): Real image volume
            gen_images (torch.Tensor): Generated image volume

        Returns:
            float: NCC value between -1 and 1 (higher is better)
        """
        self._ensure_ncc_calculator()
        return self.ncc_calculator.calculate(real_images, gen_images)

    def calculate_all_metrics(self, real_images, gen_images):
        """
        Calculate all available domain metrics.

        Args:
            real_images (torch.Tensor): Real image volume
            gen_images (torch.Tensor): Generated image volume

        Returns:
            dict: Dictionary of metric names and values
        """
        metrics = {}

        try:
            metrics["fid"] = self.calculate_fid(real_images, gen_images)
        except Exception as e:
            print(f"Error calculating FID: {e}")
            metrics["fid"] = float("nan")
            if self.device != "cpu":
                torch.cuda.empty_cache()

        try:
            metrics["ncc"] = self.calculate_ncc(real_images, gen_images)
        except Exception as e:
            print(f"Error calculating NCC: {e}")
            metrics["ncc"] = float("nan")

        return metrics
