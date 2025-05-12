class FeatureVisualizer:
    """
    Utility class for visualizing features in discriminator and generator networks.
    Used for debugging and analyzing network behavior during training or testing.
    """

    def __init__(self, save_path="./feature_maps", max_channels=16, normalize=True):
        """
        Initialize the feature visualizer.

        Args:
            save_path (str): Directory to save visualizations
            max_channels (int): Maximum number of channels to visualize per layer
            normalize (bool): Whether to normalize feature maps for better visualization
        """
        self.save_path = Path(save_path)
        self.max_channels = max_channels
        self.normalize = normalize
        os.makedirs(self.save_path, exist_ok=True)

    def _prepare_feature_grid(self, feature_maps, layer_name, slice_idx=None):
        """
        Prepare a grid of feature maps for visualization with robust handling of various tensor shapes.

        Args:
            feature_maps (torch.Tensor): Feature maps of various shapes
            layer_name (str): Name of the layer
            slice_idx (int, optional): Z-slice to visualize for 3D volumes

        Returns:
            np.ndarray: Grid of feature maps
        """
        try:
            feature_maps = feature_maps[0]

            feature_maps = feature_maps.detach().cpu().numpy()

            if feature_maps.size == 0:
                print(f"Warning: Empty feature maps for {layer_name}")
                return np.zeros((10, 10))

            num_channels = feature_maps.shape[0]

            feature_maps_2d = []

            for c in range(min(num_channels, self.max_channels)):
                channel_data = feature_maps[c]

                if len(channel_data.shape) == 0:
                    feature_maps_2d.append(np.ones((5, 5)) * channel_data)

                elif len(channel_data.shape) == 1:
                    size = int(np.ceil(np.sqrt(channel_data.size)))
                    padded = np.zeros((size * size))
                    padded[: channel_data.size] = channel_data
                    feature_maps_2d.append(padded.reshape(size, size))

                elif len(channel_data.shape) == 2:
                    feature_maps_2d.append(channel_data)

                elif len(channel_data.shape) == 3:
                    if slice_idx is None:
                        middle_idx = channel_data.shape[2] // 2
                    else:
                        middle_idx = min(slice_idx, channel_data.shape[2] - 1)
                    feature_maps_2d.append(channel_data[:, :, middle_idx])

                elif len(channel_data.shape) >= 4:
                    if slice_idx is None:
                        depth_dim = 2
                        if channel_data.shape[2] < channel_data.shape[-1]:
                            depth_dim = -1
                        middle_idx = channel_data.shape[depth_dim] // 2
                    else:
                        depth_dim = 2
                        middle_idx = min(slice_idx, channel_data.shape[depth_dim] - 1)

                    idx = [slice(None), slice(None)]
                    for d in range(2, len(channel_data.shape)):
                        if d == depth_dim:
                            idx.append(middle_idx)
                        else:
                            idx.append(0)

                    slice_2d = channel_data[tuple(idx)]
                    feature_maps_2d.append(slice_2d)

            if not feature_maps_2d:
                return np.zeros((10, 10))

            grid_size = int(np.ceil(np.sqrt(len(feature_maps_2d))))

            max_h = max(fm.shape[0] for fm in feature_maps_2d)
            max_w = max(fm.shape[1] for fm in feature_maps_2d)

            grid = np.zeros((grid_size * max_h, grid_size * max_w))

            for i, feat in enumerate(feature_maps_2d):
                if i >= grid_size * grid_size:
                    break

                if self.normalize:
                    min_val, max_val = feat.min(), feat.max()
                    if max_val > min_val:
                        feat = (feat - min_val) / (max_val - min_val)

                if feat.shape[0] != max_h or feat.shape[1] != max_w:
                    feat_tensor = (
                        torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
                    )
                    feat_tensor = F.interpolate(
                        feat_tensor,
                        size=(max_h, max_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    feat = feat_tensor.squeeze().numpy()

                row = i // grid_size
                col = i % grid_size

                h, w = feat.shape
                grid[row * max_h : row * max_h + h, col * max_w : col * max_w + w] = (
                    feat
                )

            return grid

        except Exception as e:
            print(f"Error in _prepare_feature_grid for {layer_name}: {str(e)}")
            return np.zeros((10, 10))

    def visualize_discriminator(
        self, model, input_tensor, output_path=None, depth_slice=None
    ):
        """
        Visualize feature maps for each layer of the discriminator.

        Args:
            model (nn.Module): Discriminator model
            input_tensor (torch.Tensor): Input tensor
            output_path (str, optional): Path to save visualization
            depth_slice (int, optional): Z-slice to visualize for 3D volumes

        Returns:
            dict: Dictionary of feature maps for each layer
        """
        features = {}

        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    features[name] = output[0]
                else:
                    features[name] = output

            return hook

        def safe_register_hook(component, component_name):
            try:
                if component is not None:
                    hooks.append(
                        component.register_forward_hook(get_activation(component_name))
                    )
                    return True
                return False
            except Exception as e:
                print(f"Warning: Could not register hook for {component_name}: {e}")
                return False

        if isinstance(model, torch.nn.DataParallel):
            module_to_hook = model.module
        else:
            module_to_hook = model

        if hasattr(module_to_hook, "initial"):
            safe_register_hook(module_to_hook.initial, "initial")

        if hasattr(module_to_hook, "layers"):
            for i, layer in enumerate(module_to_hook.layers):
                safe_register_hook(layer, f"layer_{i}")

        if hasattr(module_to_hook, "final"):
            safe_register_hook(module_to_hook.final, "final")

        with torch.no_grad():
            try:
                _ = model(input_tensor)
            except Exception as e:
                print(f"Error during discriminator forward pass: {e}")
                import traceback

                traceback.print_exc()

        for hook in hooks:
            hook.remove()

        if not features:
            print("No features were captured during discriminator visualization!")
            return {}

        if output_path is None:
            output_path = self.save_path / f"discriminator_features.png"
        else:
            output_path = Path(output_path)
            os.makedirs(output_path.parent, exist_ok=True)

        num_features = len(features)
        if num_features == 0:
            return features

        plt.figure(figsize=(24, 20))
        plt.suptitle("Discriminator Feature Maps", fontsize=16)

        ordered_features = []
        if "initial" in features:
            ordered_features.append(("initial", features["initial"]))

        for i in range(num_features):
            layer_name = f"layer_{i}"
            if layer_name in features:
                ordered_features.append((layer_name, features[layer_name]))

        if "final" in features:
            ordered_features.append(("final", features["final"]))

        for name, feat in features.items():
            if name not in [item[0] for item in ordered_features]:
                ordered_features.append((name, feat))

        rows = len(ordered_features)
        for i, (name, feat) in enumerate(ordered_features):
            plt.subplot(rows, 2, 2 * i + 1)
            try:
                grid = self._prepare_feature_grid(feat, name, depth_slice)
                plt.imshow(grid, cmap="viridis")
                plt.title(f"{name} - {feat.shape}", fontsize=12)
            except Exception as e:
                print(f"Error visualizing discriminator feature {name}: {e}")
                plt.text(
                    0.5, 0.5, f"Error visualizing {name}", ha="center", va="center"
                )
            plt.axis("off")

            plt.subplot(rows, 2, 2 * i + 2)
            try:
                if feat[0].dim() >= 3:
                    if feat[0].dim() == 3:
                        channel_activations = feat[0].mean(dim=(1, 2))
                    else:
                        channel_activations = feat[0].mean(dim=(1, 2, 3))

                    channel_activations = channel_activations.detach().cpu().numpy()
                    plt.bar(range(len(channel_activations)), channel_activations)
                    plt.title(f"Channel Activations", fontsize=10)
                else:
                    plt.text(0.5, 0.5, "No channel dimension", ha="center", va="center")
            except Exception as e:
                print(f"Error plotting discriminator activations for {name}: {e}")
                plt.text(
                    0.5, 0.5, "Error plotting activations", ha="center", va="center"
                )

        plt.tight_layout()
        try:
            plt.savefig(output_path)
        except Exception as e:
            print(f"Error saving discriminator visualization: {e}")

        plt.close("all")
        return features

    def visualize_generator(
        self, model, input_tensor, output_path=None, depth_slice=None
    ):
        """
        Visualize feature maps for each layer of the generator.

        Args:
            model (nn.Module): Generator model (UNet or UNetWithSTN)
            input_tensor (torch.Tensor): Input tensor
            output_path (str, optional): Path to save visualization
            depth_slice (int, optional): Z-slice to visualize for 3D volumes

        Returns:
            dict: Dictionary of feature maps for each layer
        """
        features = {}

        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                features[name] = output

            return hook

        def safe_register_hook(component, component_name):
            try:
                if component is not None:
                    hooks.append(
                        component.register_forward_hook(get_activation(component_name))
                    )
                    return True
                return False
            except Exception as e:
                print(f"Warning: Could not register hook for {component_name}: {e}")
                return False

        is_unet_stn = hasattr(model, "unet")

        if is_unet_stn:
            safe_register_hook(model.loc_conv, "stn_loc_features")
            model_to_hook = model.unet
        else:
            model_to_hook = model

        if isinstance(model_to_hook, torch.nn.DataParallel):
            model_to_hook = model_to_hook.module

        for i in range(1, 5):
            if hasattr(model_to_hook, f"enc{i}"):
                safe_register_hook(getattr(model_to_hook, f"enc{i}"), f"encoder_{i}")

        if hasattr(model_to_hook, "bridge"):
            safe_register_hook(model_to_hook.bridge, "bridge")

        for i in range(1, 5):
            if hasattr(model_to_hook, f"dec{i}"):
                safe_register_hook(getattr(model_to_hook, f"dec{i}"), f"decoder_{i}")
            if hasattr(model_to_hook, f"fusion{i}"):
                safe_register_hook(getattr(model_to_hook, f"fusion{i}"), f"fusion_{i}")

        if hasattr(model_to_hook, "output"):
            safe_register_hook(model_to_hook.output, "output")

        with torch.no_grad():
            try:
                _ = model(input_tensor)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                import traceback

                traceback.print_exc()

        for hook in hooks:
            hook.remove()

        if not features:
            print("No features were captured during visualization!")
            return {}

        if output_path is None:
            output_path = self.save_path / f"generator_features.png"
        else:
            output_path = Path(output_path)
            os.makedirs(output_path.parent, exist_ok=True)
        encoder_features = [
            f for f in features.keys() if "encoder" in f or "bridge" in f or "stn" in f
        ]
        decoder_features = [
            f
            for f in features.keys()
            if "decoder" in f or "fusion" in f or "output" in f
        ]

        if encoder_features:
            plt.figure(figsize=(24, 20))
            plt.suptitle("Generator Feature Maps - Encoder Path", fontsize=16)

            rows = len(encoder_features)
            for i, name in enumerate(sorted(encoder_features)):
                feat = features[name]

                plt.subplot(rows, 2, 2 * i + 1)
                try:
                    grid = self._prepare_feature_grid(feat, name, depth_slice)
                    plt.imshow(grid, cmap="viridis")
                    plt.title(f"{name} - {feat.shape}", fontsize=12)
                except Exception as e:
                    print(f"Error visualizing grid for {name}: {e}")
                    plt.text(
                        0.5, 0.5, f"Error visualizing {name}", ha="center", va="center"
                    )
                plt.axis("off")

                plt.subplot(rows, 2, 2 * i + 2)
                try:
                    if feat[0].dim() >= 3:
                        if feat[0].dim() == 3:
                            channel_activations = feat[0].mean(dim=(1, 2))
                        else:
                            channel_activations = feat[0].mean(dim=(1, 2, 3))

                        channel_activations = channel_activations.detach().cpu().numpy()
                        plt.bar(range(len(channel_activations)), channel_activations)
                        plt.title(f"Channel Activations", fontsize=10)
                    else:
                        plt.text(
                            0.5, 0.5, "No channel dimension", ha="center", va="center"
                        )
                except Exception as e:
                    print(f"Error plotting channel activations for {name}: {e}")
                    plt.text(
                        0.5, 0.5, "Error plotting activations", ha="center", va="center"
                    )

            plt.tight_layout()
            try:
                plt.savefig(str(output_path).replace(".png", "_encoder.png"))
            except Exception as e:
                print(f"Error saving encoder visualization: {e}")
            plt.close()

        if decoder_features:
            plt.figure(figsize=(24, 20))
            plt.suptitle("Generator Feature Maps - Decoder Path", fontsize=16)

            rows = len(decoder_features)
            for i, name in enumerate(sorted(decoder_features)):
                feat = features[name]

                plt.subplot(rows, 2, 2 * i + 1)
                try:
                    grid = self._prepare_feature_grid(feat, name, depth_slice)
                    plt.imshow(grid, cmap="viridis")
                    plt.title(f"{name} - {feat.shape}", fontsize=12)
                except Exception as e:
                    print(f"Error visualizing grid for {name}: {e}")
                    plt.text(
                        0.5, 0.5, f"Error visualizing {name}", ha="center", va="center"
                    )
                plt.axis("off")

                plt.subplot(rows, 2, 2 * i + 2)
                try:
                    if feat[0].dim() >= 3:
                        if feat[0].dim() == 3:
                            channel_activations = feat[0].mean(dim=(1, 2))
                        else:
                            channel_activations = feat[0].mean(dim=(1, 2, 3))

                        channel_activations = channel_activations.detach().cpu().numpy()
                        plt.bar(range(len(channel_activations)), channel_activations)
                        plt.title(f"Channel Activations", fontsize=10)
                    else:
                        plt.text(
                            0.5, 0.5, "No channel dimension", ha="center", va="center"
                        )
                except Exception as e:
                    print(f"Error plotting channel activations for {name}: {e}")
                    plt.text(
                        0.5, 0.5, "Error plotting activations", ha="center", va="center"
                    )

            plt.tight_layout()
            try:
                plt.savefig(str(output_path).replace(".png", "_decoder.png"))
            except Exception as e:
                print(f"Error saving decoder visualization: {e}")
            plt.close()

        return features


import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class SliceFeatureVisualizer:
    """
    Enhanced utility class for visualizing features in discriminator and generator networks
    across full 2D slices by stitching patch-level feature maps.
    """

    def __init__(
        self,
        save_path="./slice_feature_maps",
        max_channels=16,
        normalize=True,
        patch_size=(200, 200, 20),
        stride_inplane=100,
        stride_layer=10,
    ):
        """
        Initialize the slice feature visualizer.

        Args:
            save_path (str): Directory to save visualizations
            max_channels (int): Maximum number of channels to visualize per layer
            normalize (bool): Whether to normalize feature maps for better visualization
            patch_size (tuple): Size of patches for sliding window (D, H, W)
            stride_inplane (int): Stride for in-plane dimensions (H, W)
            stride_layer (int): Stride for depth dimension (D)
        """
        self.save_path = Path(save_path)
        self.max_channels = max_channels
        self.normalize = normalize
        self.patch_size = patch_size
        self.stride_inplane = stride_inplane
        self.stride_layer = stride_layer
        os.makedirs(self.save_path, exist_ok=True)

    def get_patch_grid_positions(self, volume_size):
        """
        Calculate grid positions for patches across the volume.

        Args:
            volume_size (tuple): Size of the input volume (D, H, W)

        Returns:
            list: List of (z_start, y_start, x_start) coordinates for patches
        """
        D, H, W = volume_size
        d_patch, h_patch, w_patch = self.patch_size

        z_positions = list(range(0, D - d_patch + 1, self.stride_layer))
        if D - d_patch > 0 and z_positions[-1] + d_patch < D:
            z_positions.append(D - d_patch)

        y_positions = list(range(0, H - h_patch + 1, self.stride_inplane))
        if H - h_patch > 0 and y_positions[-1] + h_patch < H:
            y_positions.append(H - h_patch)

        x_positions = list(range(0, W - w_patch + 1, self.stride_inplane))
        if W - w_patch > 0 and x_positions[-1] + w_patch < W:
            x_positions.append(W - w_patch)

        patch_positions = []
        for z in z_positions:
            for y in y_positions:
                for x in x_positions:
                    patch_positions.append((z, y, x))

        return patch_positions

    def extract_patch(self, volume, start_pos):
        """
        Extract a patch from the volume at the specified position.

        Args:
            volume (torch.Tensor): Input volume [1, C, D, H, W]
            start_pos (tuple): Starting position (z, y, x)

        Returns:
            torch.Tensor: Extracted patch
        """
        z, y, x = start_pos
        d_patch, h_patch, w_patch = self.patch_size

        return volume[:, :, z : z + d_patch, y : y + h_patch, x : x + w_patch]

    def _extract_features(self, model, patch, model_type="generator"):
        """
        Extract features from a model for a given patch.

        Args:
            model (nn.Module): The model to extract features from
            patch (torch.Tensor): Input patch
            model_type (str): Either "generator" or "discriminator"

        Returns:
            dict: Dictionary of feature maps from each layer
        """
        features = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    features[name] = output[0].detach()
                else:
                    features[name] = output.detach()

            return hook

        if isinstance(model, torch.nn.DataParallel):
            module_to_hook = model.module
        else:
            module_to_hook = model

        if model_type == "generator":
            if hasattr(module_to_hook, "unet"):
                module_to_hook = module_to_hook.unet

            for i in range(1, 5):
                if hasattr(module_to_hook, f"enc{i}"):
                    hooks.append(
                        getattr(module_to_hook, f"enc{i}").register_forward_hook(
                            get_activation(f"encoder_{i}")
                        )
                    )

            if hasattr(module_to_hook, "bridge"):
                hooks.append(
                    module_to_hook.bridge.register_forward_hook(
                        get_activation("bridge")
                    )
                )

            for i in range(1, 5):
                if hasattr(module_to_hook, f"dec{i}"):
                    hooks.append(
                        getattr(module_to_hook, f"dec{i}").register_forward_hook(
                            get_activation(f"decoder_{i}")
                        )
                    )

        elif model_type == "discriminator":
            if hasattr(module_to_hook, "initial"):
                hooks.append(
                    module_to_hook.initial.register_forward_hook(
                        get_activation("initial")
                    )
                )

            if hasattr(module_to_hook, "layers"):
                for i, layer in enumerate(module_to_hook.layers):
                    hooks.append(
                        layer.register_forward_hook(get_activation(f"layer_{i}"))
                    )

            if hasattr(module_to_hook, "final"):
                hooks.append(
                    module_to_hook.final.register_forward_hook(get_activation("final"))
                )

        with torch.no_grad():
            _ = model(patch)

        for hook in hooks:
            hook.remove()

        return features

    def _prepare_feature_grid(self, feature_maps, layer_name, slice_idx=None):
        """
        Prepare a grid of feature maps for visualization.

        Args:
            feature_maps (torch.Tensor): Feature maps tensor
            layer_name (str): Name of the layer
            slice_idx (int): Z-slice to visualize for 3D volumes

        Returns:
            np.ndarray: Grid of feature maps
        """
        try:
            feature_maps = feature_maps[0]

            feature_maps = feature_maps.detach().cpu().numpy()

            if feature_maps.size == 0:
                print(f"Warning: Empty feature maps for {layer_name}")
                return np.zeros((10, 10))

            num_channels = feature_maps.shape[0]

            feature_maps_2d = []

            for c in range(min(num_channels, self.max_channels)):
                channel_data = feature_maps[c]

                if len(channel_data.shape) == 3:
                    if slice_idx is None:
                        middle_idx = channel_data.shape[0] // 2
                    else:
                        middle_idx = min(slice_idx, channel_data.shape[0] - 1)
                    feature_maps_2d.append(channel_data[middle_idx])

                elif len(channel_data.shape) == 2:
                    feature_maps_2d.append(channel_data)

                else:
                    print(
                        f"Warning: Unsupported shape {channel_data.shape} for {layer_name}"
                    )
                    continue

            if not feature_maps_2d:
                return np.zeros((10, 10))

            grid_size = int(np.ceil(np.sqrt(len(feature_maps_2d))))

            max_h = max(fm.shape[0] for fm in feature_maps_2d)
            max_w = max(fm.shape[1] for fm in feature_maps_2d)

            grid = np.zeros((grid_size * max_h, grid_size * max_w))

            for i, feat in enumerate(feature_maps_2d):
                if i >= grid_size * grid_size:
                    break

                if self.normalize:
                    min_val, max_val = feat.min(), feat.max()
                    if max_val > min_val:
                        feat = (feat - min_val) / (max_val - min_val)

                row = i // grid_size
                col = i % grid_size

                h, w = feat.shape
                grid[row * max_h : row * max_h + h, col * max_w : col * max_w + w] = (
                    feat
                )

            return grid

        except Exception as e:
            print(f"Error in _prepare_feature_grid for {layer_name}: {str(e)}")
            return np.zeros((10, 10))

    def visualize_slice_features(
        self, model, input_volume, slice_idx, model_type="generator", output_path=None
    ):
        """
        Visualize features for a specific slice by extracting features from patches and stitching them together.

        Args:
            model (nn.Module): Model to visualize
            input_volume (torch.Tensor): Input volume [1, C, D, H, W]
            slice_idx (int): Z-slice to visualize
            model_type (str): Model type ("generator" or "discriminator")
            output_path (str): Path to save visualizations

        Returns:
            dict: Dictionary of stitched feature maps for each layer
        """
        if output_path is None:
            output_path = (
                self.save_path / f"{model_type}_slice_{slice_idx}_features.png"
            )
        else:
            output_path = Path(output_path)
            os.makedirs(output_path.parent, exist_ok=True)

        _, _, D, H, W = input_volume.shape

        patch_positions = self.get_patch_grid_positions((D, H, W))

        all_layer_features = {}

        print(f"Processing {len(patch_positions)} patches for slice {slice_idx}...")
        for pos in tqdm(patch_positions):
            patch = self.extract_patch(input_volume, pos)

            patch_features = self._extract_features(model, patch, model_type=model_type)

            for layer_name, feat in patch_features.items():
                if layer_name not in all_layer_features:
                    all_layer_features[layer_name] = []

                all_layer_features[layer_name].append((pos, feat))

        layer_grids = {}

        ordered_layers = sorted(all_layer_features.keys())

        for layer_name in ordered_layers:
            layer_patches = all_layer_features[layer_name]

            if not layer_patches:
                continue

            _, first_feat = layer_patches[0]

            feat_channels = first_feat.shape[1]

            for channel_idx in range(min(feat_channels, self.max_channels)):

                _, sample_feat = layer_patches[0]

                if model_type == "generator":
                    if "encoder" in layer_name:
                        level = int(layer_name.split("_")[-1])
                        scale_factor = 2 ** (level - 1)
                    elif "bridge" in layer_name:
                        scale_factor = 2**4
                    elif "decoder" in layer_name:
                        level = int(layer_name.split("_")[-1])
                        scale_factor = 2 ** (4 - level)
                    else:
                        scale_factor = 1
                else:
                    if "initial" in layer_name:
                        scale_factor = 1
                    elif "layer" in layer_name:
                        level = int(layer_name.split("_")[-1])
                        scale_factor = 2 ** min(level + 1, 3)
                    else:
                        scale_factor = 2**3

                feat_H = max(1, H // scale_factor)
                feat_W = max(1, W // scale_factor)

                stitched_feat = np.zeros((feat_H, feat_W))

                weight_map = np.zeros((feat_H, feat_W))

                for (z, y, x), feat in layer_patches:
                    if slice_idx < z or slice_idx >= z + self.patch_size[0]:
                        continue

                    patch_slice_idx = slice_idx - z

                    if feat.dim() >= 5:
                        patch_feat = feat[0, channel_idx, patch_slice_idx].cpu().numpy()
                    elif feat.dim() == 4:
                        patch_feat = feat[0, channel_idx].cpu().numpy()
                    else:
                        continue

                    feat_y = max(0, y // scale_factor)
                    feat_x = max(0, x // scale_factor)

                    feat_h_patch = patch_feat.shape[0]
                    feat_w_patch = patch_feat.shape[1]

                    feat_h_patch = min(feat_h_patch, feat_H - feat_y)
                    feat_w_patch = min(feat_w_patch, feat_W - feat_x)

                    if feat_h_patch <= 0 or feat_w_patch <= 0:
                        continue

                    y_coords = np.arange(feat_h_patch) / feat_h_patch
                    x_coords = np.arange(feat_w_patch) / feat_w_patch
                    y_weights = 0.5 - 0.5 * np.cos(2 * np.pi * y_coords)
                    x_weights = 0.5 - 0.5 * np.cos(2 * np.pi * x_coords)
                    weight_mask = np.outer(y_weights, x_weights)

                    stitched_feat[
                        feat_y : feat_y + feat_h_patch, feat_x : feat_x + feat_w_patch
                    ] += (patch_feat[:feat_h_patch, :feat_w_patch] * weight_mask)

                    weight_map[
                        feat_y : feat_y + feat_h_patch, feat_x : feat_x + feat_w_patch
                    ] += weight_mask

                mask = weight_map > 0
                if mask.any():
                    stitched_feat[mask] /= weight_map[mask]

                if layer_name not in layer_grids:
                    layer_grids[layer_name] = []

                layer_grids[layer_name].append(stitched_feat)

        num_layers = len(layer_grids)
        if num_layers == 0:
            print("No features to visualize!")
            return {}

        cols = min(3, num_layers)
        rows = (num_layers + cols - 1) // cols

        plt.figure(figsize=(5 * cols, 4 * rows))
        plt.suptitle(
            f"{model_type.capitalize()} Feature Maps - Slice {slice_idx}", fontsize=16
        )

        for i, layer_name in enumerate(sorted(layer_grids.keys())):
            layer_feat = layer_grids[layer_name]

            if not layer_feat:
                continue

            plt.subplot(rows, cols, i + 1)

            channels_grid = self._create_channel_grid(layer_feat)

            plt.imshow(channels_grid, cmap="viridis")
            plt.title(f"{layer_name}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Slice visualization saved to {output_path}")

        return layer_grids

    def _create_channel_grid(self, feature_maps_list):
        """
        Create a grid of channel feature maps.

        Args:
            feature_maps_list (list): List of feature maps

        Returns:
            np.ndarray: Grid of feature maps
        """
        grid_size = int(np.ceil(np.sqrt(len(feature_maps_list))))

        max_h = max(fm.shape[0] for fm in feature_maps_list)
        max_w = max(fm.shape[1] for fm in feature_maps_list)

        grid = np.zeros((grid_size * max_h, grid_size * max_w))

        for i, feat in enumerate(feature_maps_list):
            if i >= grid_size * grid_size:
                break

            if self.normalize:
                min_val, max_val = feat.min(), feat.max()
                if max_val > min_val:
                    feat = (feat - min_val) / (max_val - min_val)

            row = i // grid_size
            col = i % grid_size

            h, w = feat.shape
            grid[row * max_h : row * max_h + h, col * max_w : col * max_w + w] = feat

        return grid

    def visualize_slice(
        self, model, input_volume, slice_idx, model_type="generator", output_path=None
    ):
        """
        Visualize a specific slice of the feature maps.

        Args:
            model (nn.Module): Model to visualize
            input_volume (torch.Tensor): Input volume [1, C, D, H, W]
            slice_idx (int): Z-slice to visualize
            model_type (str): Model type ("generator" or "discriminator")
            output_path (str): Path to save visualizations

        Returns:
            dict: Dictionary of stitched feature maps for each layer
        """
        return self.visualize_slice_features(
            model, input_volume, slice_idx, model_type, output_path
        )

    def visualize_generator_slice(
        self, model, input_volume, slice_idx, output_path=None
    ):
        """
        Visualize generator features for a specific slice.

        Args:
            model (nn.Module): Generator model
            input_volume (torch.Tensor): Input volume [1, C, D, H, W]
            slice_idx (int): Z-slice to visualize
            output_path (str): Path to save visualization

        Returns:
            dict: Dictionary of stitched feature maps for each layer
        """
        return self.visualize_slice(
            model, input_volume, slice_idx, "generator", output_path
        )

    def visualize_discriminator_slice(
        self, model, input_volume, slice_idx, output_path=None
    ):
        """
        Visualize discriminator features for a specific slice.

        Args:
            model (nn.Module): Discriminator model
            input_volume (torch.Tensor): Input volume [1, C, D, H, W]
            slice_idx (int): Z-slice to visualize
            output_path (str): Path to save visualization

        Returns:
            dict: Dictionary of stitched feature maps for each layer
        """
        return self.visualize_slice(
            model, input_volume, slice_idx, "discriminator", output_path
        )


def visualize_gradient_flow(model, save_path="./gradient_flow.png"):
    """
    Visualize gradient flow through the network.

    Args:
        model (nn.Module): Model to visualize
        save_path (str): Path to save the visualization
    """
    ave_grads = []
    max_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    plt.figure(figsize=(15, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            "Max gradients",
            "Mean gradients",
        ]
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_attention_maps(
    model, input_tensor, save_path="./attention_maps.png", depth_slice=None
):
    """
    Visualize attention maps in the model with improved normalization and 3D handling.

    Args:
        model (nn.Module): Model containing attention mechanisms
        input_tensor (torch.Tensor): Input tensor
        save_path (str): Path to save visualization
        depth_slice (int, optional): Z-slice to visualize for 3D volumes
    """
    attention_maps = {}

    hooks = []

    def get_attention(name):
        def hook(module, input, output):
            if hasattr(module, "gamma"):
                gamma = module.gamma.item()
                attention_maps[name] = {"gamma": gamma, "output": output}

        return hook

    def register_hooks(model, prefix=""):
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if "attention" in name.lower() or "Attention" in module.__class__.__name__:
                hooks.append(module.register_forward_hook(get_attention(full_name)))
            register_hooks(module, full_name)

    register_hooks(model)

    with torch.no_grad():
        try:
            _ = model(input_tensor)
        except Exception as e:
            print(f"Error during forward pass: {e}")

    for hook in hooks:
        hook.remove()

    if not attention_maps:
        print("No attention maps found in the model")
        return

    plt.figure(figsize=(15, 10))
    plt.suptitle("Attention Maps", fontsize=16)

    num_maps = len(attention_maps)
    rows = int(np.ceil(np.sqrt(num_maps)))
    cols = int(np.ceil(num_maps / rows))

    for i, (name, data) in enumerate(attention_maps.items()):
        plt.subplot(rows, cols, i + 1)
        output = data["output"]
        gamma = data.get("gamma", 1.0)

        if isinstance(output, torch.Tensor):
            if len(output.shape) == 5:
                if depth_slice is None:
                    depth_slice = output.shape[2] // 2
                output = output[0, :, depth_slice]
                if output.shape[0] > 1:
                    output = output.mean(dim=0)
                else:
                    output = output[0]

            elif len(output.shape) == 4:
                output = output[0]
                if output.shape[0] > 1:
                    output = output.mean(dim=0)
                else:
                    output = output[0]

            output = output.detach().cpu().numpy()
            plt.imshow(output, cmap="inferno")
            plt.title(f"{name} (γ={gamma:.3f})")
            plt.axis("off")

            plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_slice_attention_maps(model, input_tensor, slice_idx, save_path):
    """
    Visualize attention maps for a specific slice.

    Args:
        model (nn.Module): Model containing attention mechanisms
        input_tensor (torch.Tensor): Input tensor
        slice_idx (int): Z-slice to visualize
        save_path (str): Path to save visualization
    """
    attention_maps = {}
    hooks = []

    def get_attention(name):
        def hook(module, input, output):
            if hasattr(module, "gamma"):
                gamma = module.gamma.item()
                attention_maps[name] = {"gamma": gamma, "output": output}

        return hook

    def register_hooks(model, prefix=""):
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if "attention" in name.lower() or "Attention" in module.__class__.__name__:
                hooks.append(module.register_forward_hook(get_attention(full_name)))
            register_hooks(module, full_name)

    register_hooks(model)

    with torch.no_grad():
        _ = model(input_tensor)

    for hook in hooks:
        hook.remove()

    if not attention_maps:
        print("No attention maps found in the model")
        return

    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Attention Maps - Slice {slice_idx}", fontsize=16)

    num_maps = len(attention_maps)
    rows = int(np.ceil(np.sqrt(num_maps)))
    cols = int(np.ceil(num_maps / rows))

    for i, (name, data) in enumerate(attention_maps.items()):
        plt.subplot(rows, cols, i + 1)
        output = data["output"]
        gamma = data.get("gamma", 1.0)

        if isinstance(output, torch.Tensor):
            if len(output.shape) == 5:
                slice_idx_adj = min(slice_idx, output.shape[2] - 1)
                output = output[0, :, slice_idx_adj]
                if output.shape[0] > 1:
                    output = output.mean(dim=0)
                else:
                    output = output[0]

            elif len(output.shape) == 4:
                output = output[0]
                if output.shape[0] > 1:
                    output = output.mean(dim=0)
                else:
                    output = output[0]

            output = output.detach().cpu().numpy()
            plt.imshow(output, cmap="inferno")
            plt.title(f"{name} (γ={gamma:.3f})")
            plt.axis("off")

            plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
