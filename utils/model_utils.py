import functools

import torch
from torch import nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


def get_norm_layer(norm_type="instance"):
    """
    Returns the normalization layer for the network.

    Args:
        norm_type (str): Type of normalization layer ('batch', 'instance', or 'none')

    Returns:
        function: A partial function that creates the normalization layer

    Raises:
        NotImplementedError: If the normalization type is not implemented
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm3d, affine=False, track_running_stats=True
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """
    Returns a learning rate scheduler for the optimizer.

    Args:
        optimizer: The optimizer for which to create a scheduler
        opt: Object containing scheduling parameters (lr_policy, niter, etc.)

    Returns:
        torch.optim.lr_scheduler: The configured learning rate scheduler

    Raises:
        NotImplementedError: If the learning rate policy is not implemented
    """
    if opt.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(
                opt.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.niter, eta_min=0
        )
    else:
        raise NotImplementedError(
            "learning rate policy [%s] is not implemented" % opt.lr_policy
        )
    return scheduler


def init_weights(net, init_type="normal", gain=0.02):
    """
    Initialize network weights using various initialization methods.

    Args:
        net (torch.nn.Module): Network to initialize
        init_type (str): Initialization method ('normal', 'xavier', 'kaiming', 'orthogonal')
        gain (float): Scaling factor for initialization methods
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm3d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """
    Initialize a network and move it to GPU if available.

    Args:
        net (torch.nn.Module): Network to initialize
        init_type (str): Initialization method
        init_gain (float): Scaling factor for initialization
        gpu_ids (list): List of GPU IDs to use

    Returns:
        torch.nn.Module: Initialized network
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net
