from typing import Optional

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Enhanced Generative Adversarial Network (GAN) loss computation.
    Supports LSGAN, vanilla GAN, hinge loss, Wasserstein GAN, and relativistic variants.

    Args:
        use_lsgan (bool): If True, uses least squares GAN loss.
        use_hinge (bool): If True, uses hinge loss.
        use_wasserstein (bool): If True, uses Wasserstein distance as loss.
        relativistic (bool): If True, uses relativistic discriminator formulation.
        device (torch.device): Device to run the computations on.
    """

    def __init__(
        self,
        use_lsgan,
        use_hinge,
        relativistic,
        use_wasserstein=False,
        device: Optional[torch.device] = None,
    ):
        super(GANLoss, self).__init__()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))

        self.use_lsgan = use_lsgan
        self.use_hinge = use_hinge
        self.use_wasserstein = use_wasserstein
        self.relativistic = relativistic

        if sum([use_lsgan, use_hinge, use_wasserstein]) > 1:
            raise ValueError("Can only use one of LSGAN, hinge, or Wasserstein loss")

        if use_wasserstein:
            self.loss = self._wasserstein_loss
        elif use_hinge:
            self.loss = self._hinge_loss
        elif use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        """
        Create target tensor with label smoothing.

        Args:
            prediction: Discriminator predictions
            target_is_real: Whether the target should be real (1) or fake (0)

        Returns:
            torch.Tensor: Target tensor with label smoothing
        """
        if target_is_real:
            target_tensor = torch.ones_like(prediction) * 0.9
        else:
            target_tensor = torch.ones_like(prediction) * 0.1
        return target_tensor.to(prediction.device)

    def _hinge_loss(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        """
        Compute hinge loss without relying on smoothed label values.

        Args:
            prediction: Discriminator predictions
            target_is_real: Whether the target should be real (1) or fake (0)

        Returns:
            torch.Tensor: Computed hinge loss
        """
        if target_is_real:
            return torch.mean(torch.relu(1.0 - prediction))
        else:
            return torch.mean(torch.relu(1.0 + prediction))

    def _wasserstein_loss(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        """
        Compute Wasserstein loss.

        In Wasserstein GAN:
        - For real samples: We want to maximize D(x), so loss = -mean(D(x))
        - For fake samples: We want to minimize D(G(z)), so loss = mean(D(G(z)))

        We return a loss value that should be minimized, following standard optimization practices.

        Args:
            prediction: Discriminator predictions
            target_is_real: Whether the target should be real (1) or fake (0)

        Returns:
            torch.Tensor: Computed Wasserstein loss
        """
        if target_is_real:
            return -torch.mean(prediction)
        else:
            return torch.mean(prediction)

    def __call__(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        opposite_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate loss given prediction and whether target is real.

        Args:
            prediction (torch.Tensor): Current predictions (D(x) or D(G(z)))
            target_is_real (bool): Whether the target is real (True) or fake (False)
            opposite_pred (torch.Tensor, optional): Opposite type predictions for relativistic GAN

        Returns:
            torch.Tensor: Computed GAN loss
        """
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Prediction must be a torch.Tensor")

        if self.use_wasserstein:
            return self._wasserstein_loss(prediction, target_is_real)

        if self.relativistic and opposite_pred is not None:
            if self.use_hinge:
                if target_is_real:
                    return torch.mean(
                        torch.relu(1.0 - (prediction - torch.mean(opposite_pred)))
                    )
                else:
                    return torch.mean(
                        torch.relu(1.0 + (prediction - torch.mean(opposite_pred)))
                    )
            else:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                return self.loss(prediction - torch.mean(opposite_pred), target_tensor)

        if self.use_hinge:
            return self._hinge_loss(prediction, target_is_real)
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
