import torch
import torch.nn.functional as F
from torch import nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    This module implements channel-wise attention by using global average pooling
    to capture channel-wise dependencies and applying a gating mechanism.

    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for the bottleneck dimension
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(reduced_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        nn.init.zeros_(self.fc2.bias)
        if hasattr(self.fc2, "bias") and self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class ResidualBlock(nn.Module):
    """
    Residual block with optional dropout.

    Implements a basic residual block with two convolutional layers,
    instance normalization, and a learnable beta parameter for scaling
    the residual connection.

    Args:
        channels (int): Number of input and output channels
        use_dropout (bool): Whether to use dropout after the first activation
    """

    def __init__(self, channels, use_dropout=False):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout3d(0.5)
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        if self.use_dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        beta = torch.sigmoid(self.beta)
        return residual + beta * out


class AttentionBlock(nn.Module):
    """
    Self-attention block for 3D feature maps.

    Implements self-attention mechanism to capture long-range dependencies
    across the feature map. Uses query, key, value projections to compute
    attention weights.

    Args:
        channels (int): Number of input channels
        gamma_init (float): Initial value for the gamma parameter
    """

    def __init__(self, channels, gamma_init=0.5):
        super().__init__()
        reduced_channels = max(channels // 8, 1)

        self.query = nn.Conv3d(channels, channels, kernel_size=1)
        self.key = nn.Conv3d(channels, channels, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, x):
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

        batch_size = q.size(0)
        q_flat = q.view(batch_size, -1, q.size(2) * q.size(3) * q.size(4)).permute(
            0, 2, 1
        )
        k_flat = k.view(batch_size, -1, k.size(2) * k.size(3) * k.size(4))
        v_flat = v.view(batch_size, -1, v.size(2) * v.size(3) * v.size(4))

        attn = torch.bmm(q_flat, k_flat)
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
    Feature fusion block for combining features from encoder and decoder.

    This block fuses features from the encoder skip connection and the decoder path
    using 1x1 convolution, normalization, and channel attention.

    Args:
        in_channels (int): Number of input channels from combined sources
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = SEBlock(out_channels, reduction=4)

    def forward(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=True
            )

        combined = torch.cat([x, skip], dim=1)
        out = self.conv(combined)
        out = self.norm(out)
        out = self.relu(out)
        out = self.attention(out)

        return out


class UNet3D(nn.Module):
    """
    3D UNet model for volumetric medical image processing.

    This model implements a 3D UNet architecture with skip connections,
    residual blocks, and attention mechanisms. Optimized for MRI domain translation.

    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        base_channels (int): Number of base channels (multiplied in deeper layers)
        norm_layer: Normalization layer to use (default: nn.InstanceNorm3d)
        use_dropout (bool): Whether to use dropout in decoder blocks
        use_residual (bool): Whether to use residual connections for final output
        use_full_attention (bool): Whether to use attention in all decoder blocks
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        base_channels=64,
        norm_layer=nn.InstanceNorm3d,
        use_dropout=False,
        use_residual=False,
        use_full_attention=False,  # New parameter to enable attention in all decoder blocks
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_full_attention = use_full_attention
        use_bias = norm_layer == nn.InstanceNorm3d
        self.bridge_features = None
        self.enc1 = self._make_encoder_block(
            input_channels, base_channels, norm_layer, use_bias
        )
        self.enc2 = self._make_encoder_block(
            base_channels, base_channels * 2, norm_layer, use_bias
        )
        self.enc3 = self._make_encoder_block(
            base_channels * 2, base_channels * 4, norm_layer, use_bias
        )
        self.enc4 = self._make_encoder_block(
            base_channels * 4, base_channels * 8, norm_layer, use_bias
        )

        self.bridge = nn.Sequential(
            nn.Conv3d(
                base_channels * 8,
                base_channels * 16,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            ),
            norm_layer(base_channels * 16),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 16, use_dropout),
            AttentionBlock(base_channels * 16, gamma_init=0.5),
            ResidualBlock(base_channels * 16, use_dropout),
            SEBlock(base_channels * 16, reduction=4),
        )

        self.fusion4 = FusionBlock(
            base_channels * 16 + base_channels * 8, base_channels * 8
        )
        self.fusion3 = FusionBlock(
            base_channels * 8 + base_channels * 4, base_channels * 4
        )
        self.fusion2 = FusionBlock(
            base_channels * 4 + base_channels * 2, base_channels * 2
        )
        self.fusion1 = FusionBlock(base_channels * 2 + base_channels, base_channels)

        self.dec4 = self._make_decoder_block(
            base_channels * 8,
            base_channels * 8,
            norm_layer,
            use_bias,
            use_dropout,
            self.use_full_attention,
        )
        self.dec3 = self._make_decoder_block(
            base_channels * 4,
            base_channels * 4,
            norm_layer,
            use_bias,
            use_dropout,
            self.use_full_attention,
        )
        self.dec2 = self._make_decoder_block(
            base_channels * 2,
            base_channels * 2,
            norm_layer,
            use_bias,
            use_dropout,
            self.use_full_attention,
        )
        self.dec1 = self._make_decoder_block(
            base_channels,
            base_channels,
            norm_layer,
            use_bias,
            use_dropout,
            self.use_full_attention,
        )

        self.output = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, output_channels, kernel_size=1),
            nn.Tanh(),
        )

    def _make_encoder_block(self, in_channels, out_channels, norm_layer, use_bias):
        """
        Create encoder block with consistent normalization.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            norm_layer: Normalization layer to use
            use_bias (bool): Whether to use bias in convolutional layers

        Returns:
            nn.Sequential: Encoder block
        """
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias
            ),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias
            ),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_decoder_block(
        self,
        in_channels,
        out_channels,
        norm_layer,
        use_bias,
        use_dropout,
        add_attention,
    ):
        """
        Create decoder block with optional attention.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            norm_layer: Normalization layer to use
            use_bias (bool): Whether to use bias in convolutional layers
            use_dropout (bool): Whether to use dropout
            add_attention (bool): Whether to add attention mechanism

        Returns:
            nn.Sequential: Decoder block
        """
        layers = [
            ResidualBlock(in_channels, use_dropout),
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias
            ),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        ]

        if add_attention:
            layers.append(AttentionBlock(out_channels, gamma_init=0.3))
        else:
            layers.append(SEBlock(out_channels))

        if use_dropout:
            layers.append(nn.Dropout(0.3))

        return nn.Sequential(*layers)

    def _adaptive_pool(self, x):
        """
        Custom pooling that handles any dimension being too small.

        Args:
            x: Input tensor

        Returns:
            Pooled tensor
        """
        _, _, d, h, w = x.shape
        kernel_size = (1 if d < 2 else 2, 1 if h < 2 else 2, 1 if w < 2 else 2)
        if kernel_size == (1, 1, 1):
            return x
        return F.avg_pool3d(x, kernel_size=kernel_size)

    def _adaptive_upsample(self, x, target_size=None):
        """
        Custom upsampling that handles any small dimensions.

        Args:
            x: Input tensor
            target_size: Target size for upsampling (optional)

        Returns:
            Upsampled tensor
        """
        if target_size is not None:
            return F.interpolate(
                x, size=target_size, mode="trilinear", align_corners=True
            )
        else:
            _, _, d, h, w = x.shape
            scale_d = 2 if d > 1 else 1
            scale_h = 2 if h > 1 else 1
            scale_w = 2 if w > 1 else 1

            if scale_d == scale_h == scale_w == 2:
                return F.interpolate(
                    x, scale_factor=2, mode="trilinear", align_corners=True
                )
            else:
                target_d = d * scale_d
                target_h = h * scale_h
                target_w = w * scale_w
                return F.interpolate(
                    x,
                    size=(target_d, target_h, target_w),
                    mode="trilinear",
                    align_corners=True,
                )

    def forward(self, x):
        """
        Forward pass through the UNet3D.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        original_input = x

        original_size = x.shape[2:]

        if x.dtype != torch.float32:
            x = x.float()

        x1 = self.enc1(x)
        x1_down = self._adaptive_pool(x1)

        x2 = self.enc2(x1_down)
        x2_down = self._adaptive_pool(x2)

        x3 = self.enc3(x2_down)
        x3_down = self._adaptive_pool(x3)

        x4 = self.enc4(x3_down)
        x4_down = self._adaptive_pool(x4)

        bridge = self.bridge(x4_down)
        self.bridge_features = [bridge]

        up4 = self._adaptive_upsample(bridge, target_size=x4.shape[2:])
        fused4 = self.fusion4(up4, x4)
        dec4 = self.dec4(fused4)

        up3 = self._adaptive_upsample(dec4, target_size=x3.shape[2:])
        fused3 = self.fusion3(up3, x3)
        dec3 = self.dec3(fused3)

        up2 = self._adaptive_upsample(dec3, target_size=x2.shape[2:])
        fused2 = self.fusion2(up2, x2)
        dec2 = self.dec2(fused2)

        up1 = self._adaptive_upsample(dec2, target_size=x1.shape[2:])
        fused1 = self.fusion1(up1, x1)
        dec1 = self.dec1(fused1)

        out = self.output(dec1)

        if out.shape[2:] != original_size:
            out = self._adaptive_upsample(out, target_size=original_size)

        if self.use_residual:
            return 2.0 * out + original_input
        else:
            return out


class UNet3DWithSTN(nn.Module):
    """
    UNet3D with a Spatial Transformer Network (STN) for tissue-specific deformations.

    This network enhances UNet3D with a spatial transformer network that can learn
    biologically plausible deformations mimicking prostate tissue behavior
    in MRI domain translation.

    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        base_channels (int): Number of base channels (multiplied in deeper layers)
        norm_layer: Normalization layer to use (default: nn.InstanceNorm3d)
        use_dropout (bool): Whether to use dropout in decoder blocks
        use_residual (bool): Whether to use residual connections for final output
        use_full_attention (bool): Whether to use attention in all decoder blocks
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        base_channels=64,
        norm_layer=nn.InstanceNorm3d,
        use_dropout=False,
        use_residual=False,
        use_full_attention=False,  # Added parameter for full attention
    ):
        super().__init__()
        self.use_residual = use_residual

        self.loc_conv = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(True),
        )

        control_point_size = 5
        self.control_points_dim = control_point_size**3 * 3

        self.displacement_generator = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, self.control_points_dim),
        )

        self.displacement_generator[-1].weight.data.zero_()
        self.displacement_generator[-1].bias.data.zero_()

        self.reg_factor = 0.5

        self.unet = UNet3D(
            input_channels,
            output_channels,
            base_channels=base_channels,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            use_residual=use_residual,  # Pass use_residual to the UNet
            use_full_attention=use_full_attention,  # Pass use_full_attention to the UNet
        )

    def _create_displacement_field(self, x, displacement_params):
        """
        Create an elastic displacement field for tissue deformation.

        Creates a displacement field that maintains the overall shape and border integrity,
        using a control point grid approach to ensure smooth, localized deformations.

        Args:
            x: Input tensor [B, C, D, H, W]
            displacement_params: Tensor of parameters for the displacement field

        Returns:
            Sampling grid for elastic deformation
        """
        batch_size = x.size(0)
        device = x.device
        size_d, size_h, size_w = x.size(2), x.size(3), x.size(4)

        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, size_d, device=device),
            torch.linspace(-1, 1, size_h, device=device),
            torch.linspace(-1, 1, size_w, device=device),
        )

        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=3).unsqueeze(0)
        base_grid = base_grid.repeat(batch_size, 1, 1, 1, 1)

        control_size = 5
        control_points = displacement_params.reshape(
            batch_size, control_size, control_size, control_size, 3
        )

        max_displacement = 0.05
        control_points = torch.tanh(control_points) * max_displacement

        control_grid_d, control_grid_h, control_grid_w = torch.meshgrid(
            torch.linspace(-1, 1, control_size, device=device),
            torch.linspace(-1, 1, control_size, device=device),
            torch.linspace(-1, 1, control_size, device=device),
        )
        control_grid = torch.stack(
            [control_grid_w, control_grid_h, control_grid_d], dim=3
        ).unsqueeze(0)
        control_grid = control_grid.repeat(batch_size, 1, 1, 1, 1)

        control_grid = control_grid + control_points

        d_dist = torch.min(torch.abs(grid_d + 1), torch.abs(grid_d - 1))
        h_dist = torch.min(torch.abs(grid_h + 1), torch.abs(grid_h - 1))
        w_dist = torch.min(torch.abs(grid_w + 1), torch.abs(grid_w - 1))

        border_threshold = 0.2
        edge_mask = torch.min(
            torch.min(d_dist / border_threshold, h_dist / border_threshold),
            w_dist / border_threshold,
        )
        edge_mask = torch.clamp(edge_mask, 0, 1)
        edge_mask = edge_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        query_points = (base_grid + 1) / 2

        query_points = query_points * (control_size - 1)

        displacement_field = F.grid_sample(
            control_points.permute(0, 4, 1, 2, 3),
            query_points.reshape(batch_size, -1, 1, 1, 3),
            mode="bilinear",
            align_corners=True,
        )
        displacement_field = displacement_field.reshape(
            batch_size, 3, size_d, size_h, size_w
        )
        displacement_field = displacement_field.permute(0, 2, 3, 4, 1)

        displacement_field = displacement_field * edge_mask.unsqueeze(0)

        intensity_threshold = 0.05
        avg_intensity = torch.mean(x, dim=1, keepdim=True)
        tissue_mask = (avg_intensity > intensity_threshold).float()

        kernel_size = 3
        padding = kernel_size // 2
        dilated_mask = F.max_pool3d(
            tissue_mask, kernel_size=kernel_size, stride=1, padding=padding
        )

        eroded_mask = (
            -F.max_pool3d(
                -tissue_mask + 1.0, kernel_size=kernel_size, stride=1, padding=padding
            )
            + 1.0
        )

        gradient_mask = (dilated_mask - eroded_mask).clamp(0, 1)
        gradient_mask = gradient_mask.permute(0, 2, 3, 4, 1).repeat(1, 1, 1, 1, 3)

        combined_mask = gradient_mask * edge_mask.unsqueeze(0)

        displacement_field = displacement_field * combined_mask

        sampling_grid = base_grid + displacement_field

        return sampling_grid

    def _create_similarity_matrix(self, similarity_params):
        """
        Create identity transformation matrix.

        This method is kept for backward compatibility but now returns an identity
        transformation matrix since we're no longer using affine transformations.

        Args:
            similarity_params: Tensor of shape [batch_size, N] with parameters

        Returns:
            Identity transformation matrix
        """
        batch_size = similarity_params.shape[0]
        device = similarity_params.device

        identity = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        zeros = torch.zeros(batch_size, 3, 1, device=device)

        return torch.cat([identity, zeros], dim=2)

    def forward(self, x, return_transformation=False):
        """
        Forward pass with elastic deformation that preserves boundaries.

        Args:
            x: Input tensor [B, C, D, H, W]
            return_transformation: If True, return displacement parameters

        Returns:
            Transformed output and optionally displacement parameters
        """
        original_input = x

        if min(x.shape[2:]) >= 8:
            loc_features = self.loc_conv(x)
            displacement_params = self.displacement_generator(loc_features)

            sampling_grid = self._create_displacement_field(x, displacement_params)

            x_transformed = F.grid_sample(
                x,
                sampling_grid,
                align_corners=True,
                mode="bilinear",
                padding_mode="border",
            )

            out = self.unet(x_transformed)

            if self.use_residual:
                result = 2.0 * out + x_transformed
            else:
                result = out

            if return_transformation:
                return result, displacement_params
            return result

        else:
            out = self.unet(x)

            if self.use_residual:
                return 2.0 * out + x
            else:
                return out

    def compute_deformation_regularization(self, displacement_params):
        """
        Compute regularization terms for smooth, boundary-preserving deformations.

        Enforces smoothness and boundary preservation in the displacement field.

        Args:
            displacement_params: Tensor of displacement field parameters

        Returns:
            Regularization loss term
        """
        batch_size = displacement_params.shape[0]
        control_size = 5

        displacements = displacement_params.reshape(
            batch_size, control_size, control_size, control_size, 3
        )

        magnitude_loss = torch.norm(displacements, dim=4).mean()

        diff_x = torch.abs(
            displacements[:, 1:, :, :, :] - displacements[:, :-1, :, :, :]
        ).mean()
        diff_y = torch.abs(
            displacements[:, :, 1:, :, :] - displacements[:, :, :-1, :, :]
        ).mean()
        diff_z = torch.abs(
            displacements[:, :, :, 1:, :] - displacements[:, :, :, :-1, :]
        ).mean()
        smoothness_loss = diff_x + diff_y + diff_z

        edges = []
        edges.append(displacements[:, 0, 0, :, :])
        edges.append(displacements[:, 0, -1, :, :])
        edges.append(displacements[:, -1, 0, :, :])
        edges.append(displacements[:, -1, -1, :, :])
        edges.append(displacements[:, 0, 1:-1, 0, :])
        edges.append(displacements[:, 0, 1:-1, -1, :])
        edges.append(displacements[:, -1, 1:-1, 0, :])
        edges.append(displacements[:, -1, 1:-1, -1, :])
        edges.append(displacements[:, 1:-1, 0, 0, :])
        edges.append(displacements[:, 1:-1, 0, -1, :])
        edges.append(displacements[:, 1:-1, -1, 0, :])
        edges.append(displacements[:, 1:-1, -1, -1, :])

        edge_displacement = torch.cat(
            [e.reshape(batch_size, -1, 3) for e in edges], dim=1
        )
        edge_loss = torch.norm(edge_displacement, dim=2).mean()

        reg_loss = 0.5 * magnitude_loss + 1.0 * smoothness_loss + 2.0 * edge_loss

        return reg_loss
