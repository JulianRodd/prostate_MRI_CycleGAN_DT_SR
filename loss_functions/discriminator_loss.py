import torch

from models.utils.cycle_gan_utils import compute_gradient_penalty


def discriminator_loss(netD, real, fake, opt, criterionGAN):
    """
    Enhanced discriminator loss with relativistic formulation and safety checks.

    Args:
        netD: Discriminator network
        real: Real input tensor
        fake: Fake input tensor (generated)
        opt: Options object containing configuration parameters
        criterionGAN: GAN loss criterion

    Returns:
        torch.Tensor: Clamped discriminator loss
    """
    real = real.float()
    fake = fake.detach().float()

    if real.requires_grad == False:
        real.requires_grad_(True)

    noise_scale = 0.01
    real_with_noise = real + torch.randn_like(real) * noise_scale
    fake_with_noise = fake + torch.randn_like(fake) * noise_scale

    real_with_noise = torch.nan_to_num(
        real_with_noise, nan=0.0, posinf=10.0, neginf=-10.0
    )
    fake_with_noise = torch.nan_to_num(
        fake_with_noise, nan=0.0, posinf=10.0, neginf=-10.0
    )

    if hasattr(netD, "forward") and "get_features" in netD.forward.__code__.co_varnames:
        pred_real = netD(real_with_noise, get_features=False)
        pred_fake = netD(fake_with_noise, get_features=False)
    else:
        pred_real = netD(real_with_noise)
        pred_fake = netD(fake_with_noise)

    if isinstance(pred_real, tuple):
        pred_real = pred_real[0]
    if isinstance(pred_fake, tuple):
        pred_fake = pred_fake[0]

    loss_D_real = criterionGAN(pred_real, True, pred_fake)
    loss_D_fake = criterionGAN(pred_fake, False, pred_real)

    loss = (loss_D_real + loss_D_fake) * 0.5
    if opt.use_wasserstein:
        gradient_penalty = compute_gradient_penalty(
            netD,
            real,
            fake,
            real.device,
            use_checkpoint=getattr(opt, "use_gradient_checkpointing", False),
        )
        lambda_gp = 0.1
        loss = loss + lambda_gp * gradient_penalty

    clamped_loss = torch.min(
        torch.max(loss, torch.tensor(0.0, device=loss.device, requires_grad=True)),
        torch.tensor(100.0, device=loss.device, requires_grad=True),
    )

    return clamped_loss
