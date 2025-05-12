import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention3D(nn.Module):
    """
    Self-attention mechanism for 3D volumes with stronger initial attention.

    Implements a self-attention module specifically optimized for 3D medical volumes
    with enhanced attention strength through stronger gamma initialization.

    Args:
        channels (int): Number of input channels
        gamma_init (float): Initial value for the attention scaling factor
    """

    def __init__(self, channels: int, gamma_init=0.3):
        super().__init__()
        reduced_channels = max(channels // 8, 1)

        self.query = nn.Conv3d(channels, channels, kernel_size=1)
        self.key = nn.Conv3d(channels, channels, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]
        use_downsampling = torch.prod(torch.tensor(orig_size)) > 32768
        use_downsampling = False
        if use_downsampling:
            x_small = F.interpolate(
                x, scale_factor=0.5, mode="trilinear", align_corners=True
            )
            q = self.query(x_small)
            k = self.key(x_small)
            v = self.value(x_small)
        else:
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)

        batch_size = x.size(0)
        q_flat = q.view(batch_size, -1, q.size(2) * q.size(3) * q.size(4)).permute(
            0, 2, 1
        )
        k_flat = k.view(batch_size, -1, k.size(2) * k.size(3) * k.size(4))
        v_flat = v.view(batch_size, -1, v.size(2) * v.size(3) * v.size(4))

        attn = torch.bmm(q_flat, k_flat) * 1.3
        attn = F.softmax(attn, dim=2)

        out = torch.bmm(v_flat, attn.permute(0, 2, 1))
        out = out.view(batch_size, -1, *v.shape[2:])

        if use_downsampling:
            out = F.interpolate(
                out, size=orig_size, mode="trilinear", align_corners=True
            )

        return x + self.gamma * out


class FusionBlock(nn.Module):
    """
    Lightweight fusion block for feature aggregation with progressive weighting.

    This block combines features from different sources with progressive weighting
    based on layer depth to give more influence to deeper features.

    Args:
        channels_list: List of channel dimensions for each input feature map
    """

    def __init__(self, channels_list):
        super().__init__()
        self.projections = nn.ModuleList()

        self.out_channels = max(channels_list)

        for channels in channels_list:
            self.projections.append(
                nn.Sequential(
                    nn.Conv3d(channels, self.out_channels, kernel_size=1),
                    nn.InstanceNorm3d(self.out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        self.blend = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1)

    def forward(self, features):
        target_size = features[-1].shape[2:]
        processed_features = []
        num_features = len(features)

        for i, feat in enumerate(features):
            if feat is None or feat.numel() == 0:
                continue

            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="trilinear", align_corners=True
                )

            proj = self.projections[i](feat)

            layer_weight = 0.5 + (i / max(1, num_features - 1)) * 1.0
            proj = proj * layer_weight

            processed_features.append(proj)

        if processed_features:
            fused = processed_features[0]
            for feat in processed_features[1:]:
                fused = fused + feat
            total_weights = sum(
                [
                    0.5 + (i / max(1, num_features - 1)) * 1.0
                    for i in range(len(processed_features))
                ]
            )
            fused = fused / total_weights
        else:
            batch_size = features[0].shape[0]
            fused = torch.zeros(
                batch_size, self.out_channels, *target_size, device=features[0].device
            )

        fused = self.blend(fused)
        return fused


class PatchDiscriminator(nn.Module):
    """
    Optimized PatchGAN discriminator for 3D volumes.

    A streamlined implementation of PatchGAN discriminator for 3D volumes,
    removing unnecessary components while maintaining essential discriminative capabilities.
    Includes spectral normalization for improved stability.

    Args:
        input_channels (int): Number of input channels
        base_channels (int): Number of base channels in the first layer
        n_layers (int): Number of downsampling layers
        norm_layer: Normalization layer to use
        use_sigmoid (bool): Whether to use sigmoid activation in the output
        dropout_rate (float): Dropout rate for regularization
    """

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 64,
        n_layers: int = 4,
        norm_layer=nn.InstanceNorm3d,
        use_sigmoid: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        use_bias = norm_layer == nn.InstanceNorm3d

        self.initial = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(
                    input_channels,
                    base_channels,
                    kernel_size=3,
                    stride=(1, 2, 2),
                    padding=1,
                    bias=use_bias,
                )
            ),
            nn.LeakyReLU(0.2, True),
            nn.Dropout3d(self.dropout_rate),
        )

        self.layers = nn.ModuleList()
        self.feature_channels = [base_channels]

        nf_mult = 1
        for n in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            out_channels = base_channels * nf_mult
            self.feature_channels.append(out_channels)

            layer = self._create_layer(
                base_channels * nf_mult_prev,
                out_channels,
                stride=2 if n < n_layers else 1,
                norm_layer=norm_layer,
                use_bias=use_bias,
            )
            self.layers.append(layer)

        self.final = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(
                    self.feature_channels[-1],
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                )
            )
        )

        if use_sigmoid:
            self.final.append(nn.Sigmoid())

    def _create_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_layer,
        use_bias: bool,
    ):
        """
        Create a discriminator layer without attention.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the convolutional layer
            norm_layer: Normalization layer to use
            use_bias (bool): Whether to use bias in convolutional layers

        Returns:
            nn.Sequential: Layer block
        """
        layer = [
            nn.utils.spectral_norm(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=use_bias,
                )
            ),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Dropout3d(self.dropout_rate),
        ]

        return nn.Sequential(*layer)

    def forward(self, x, real_samples=None, get_features=False, use_wasserstein=False):
        """
        Forward pass with support for relativistic discrimination and Wasserstein GAN.

        Args:
            x: Input tensor
            real_samples: Optional real samples for relativistic GAN
            get_features: Whether to return intermediate features
            use_wasserstein: Whether to use Wasserstein mode

        Returns:
            Output tensor and optionally features
        """
        # Convert to float32 for consistent computation with mixed precision
        x = x.to(dtype=torch.float32)
        if real_samples is not None:
            real_samples = real_samples.to(dtype=torch.float32)

        if self.training:
            noise_level = 0.05
            x = x + torch.randn_like(x) * noise_level
            if real_samples is not None:
                real_samples = (
                    real_samples + torch.randn_like(real_samples) * noise_level
                )

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

        features = []
        x = self.initial(x)
        features.append(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        output = self.final(x)

        # For Wasserstein GAN, don't apply clamping or relativistic adjustments
        if not use_wasserstein:
            output = torch.clamp(output, -100.0, 100.0)

            if real_samples is not None:
                r = self.initial(real_samples)
                for layer in self.layers:
                    r = layer(r)
                real_output = self.final(r)
                real_output = torch.clamp(real_output, -100.0, 100.0)

                mean_real = torch.mean(real_output)
                relativistic_diff = output - mean_real
                output = relativistic_diff * 1.2

        if get_features:
            return output, features
        return output

    def forward_multiscale(self, x):
        """
        Forward pass with multi-scale discrimination for better domain signals.

        Performs discrimination at multiple scales to capture features
        at different resolutions.

        Args:
            x: Input tensor

        Returns:
            Combined multi-scale discrimination output
        """
        result_original = self.forward(x)

        x_small = F.interpolate(
            x, scale_factor=0.5, mode="trilinear", align_corners=True
        )
        result_small = self.forward(x_small)

        result_small_upscaled = F.interpolate(
            result_small,
            size=result_original.shape[2:],
            mode="trilinear",
            align_corners=True,
        )

        return result_original + 0.5 * result_small_upscaled
