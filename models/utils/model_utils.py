import logging

import torch
from torch import nn

from models.discriminator.patchgan import PatchDiscriminator
from models.generator.unet import UNet3D, UNet3DWithSTN
from utils.model_utils import get_norm_layer, init_net

logger = logging.getLogger(__name__)


def define_discriminator(
    input_nc,
    ndf,
    n_layers_D=4,
    norm="batch",
    use_sigmoid=False,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
):
    """
    Create an enhanced PatchGAN discriminator network for 3D volumes.

    This function creates a PatchDiscriminator with spectral normalization
    and advanced features designed for 3D medical image discrimination.

    Args:
        input_nc (int): Number of input channels
        ndf (int): Number of filters in the first convolutional layer
        n_layers_D (int): Number of convolutional layers in the discriminator
        norm (str): Type of normalization ('batch', 'instance', etc.)
        use_sigmoid (bool): Whether to use sigmoid activation in the output layer
        init_type (str): Weight initialization method
        init_gain (float): Scaling factor for weight initialization
        gpu_ids (list): List of GPU IDs to use

    Returns:
        PatchDiscriminator: Initialized discriminator network
    """
    try:
        norm_layer = get_norm_layer(norm_type=norm)
    except ValueError as e:
        logger.error(f"Invalid normalization type: {norm}")
        raise ValueError(f"Invalid normalization type: {norm}. Error: {str(e)}")

    logger.info(f"Creating EnhancedPatchGANDiscriminator3D with {n_layers_D} layers")
    net = PatchDiscriminator(
        input_nc,
        ndf,
        n_layers_D,
        norm_layer=norm_layer,
        use_sigmoid=use_sigmoid,
        dropout_rate=0.1,
    )

    logger.info(
        f"Initializing discriminator with method: {init_type}, gain: {init_gain}"
    )
    try:
        return init_net(net, init_type, init_gain, gpu_ids)
    except Exception as e:
        logger.error(f"Failed to initialize discriminator: {str(e)}")
        raise RuntimeError(
            f"Failed to initialize discriminator with {init_type} initialization: {str(e)}"
        )


def define_generator(
    input_nc,
    output_nc,
    ngf,
    norm="instance",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    use_stn=False,
    use_residual=False,
    use_full_attention=False,
):
    """
    Create an optimized 3D UNet generator for MRI domain translation.

    Creates either a standard UNet3D or UNet3DWithSTN (Spatial Transformer Network)
    based on the use_stn parameter. The STN helps handle tissue deformations.

    Args:
        input_nc (int): Number of input channels
        output_nc (int): Number of output channels
        ngf (int): Number of filters in the first convolutional layer (base_channels)
        norm (str): Type of normalization ('instance', 'batch')
        use_dropout (bool): Whether to use dropout layers
        init_type (str): Weight initialization method
        init_gain (float): Scaling factor for weight initialization
        gpu_ids (list): List of GPU IDs to use
        use_stn (bool): Whether to include spatial transformer network
        pretrained_path (str): Path to pretrained model weights (optional)
        patch_size (tuple): Input patch size (optional)
        use_residual (bool): Whether to use residual connections
        use_full_attention (bool): Whether to use attention in all decoder blocks

    Returns:
        nn.Module: Initialized generator network (UNet3D or UNet3DWithSTN)
    """
    if norm == "instance":
        norm_layer = nn.InstanceNorm3d
    elif norm == "batch":
        norm_layer = nn.BatchNorm3d
    else:
        raise ValueError(f"Invalid normalization type: {norm}")

    if use_stn:
        net = UNet3DWithSTN(
            input_nc,
            output_nc,
            base_channels=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            use_residual=use_residual,
            use_full_attention=use_full_attention,
        )
    else:
        net = UNet3D(
            input_nc,
            output_nc,
            base_channels=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            use_residual=use_residual,
            use_full_attention=use_full_attention,
        )

    def init_weights(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain * 0.8)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.01)

    net.apply(init_weights)

    if len(gpu_ids) > 0 and torch.cuda.is_available():
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = nn.DataParallel(net, gpu_ids)

    return net
