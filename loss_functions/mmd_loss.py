import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedMMDLoss(nn.Module):
    """
    Simplified Maximum Mean Discrepancy Loss for domain adaptation.
    Based on the approach described in the article on Maximum Mean Discrepancy
    by Onur Tunali.

    This implementation is much simpler than the previous PatchFriendlyMMDLoss.

    Args:
        kernel_type: Type of kernel to use ('multiscale' or 'rbf')
        device: Computing device (CPU/GPU)
    """

    def __init__(self, kernel_type="multiscale", device=None):
        super(SimplifiedMMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(self, source, target):
        """
        Compute the MMD loss between source and target features.

        Args:
            source: Source domain features
            target: Target domain features

        Returns:
            torch.Tensor: MMD loss value
        """
        if source.dim() > 2:
            source = source.view(source.size(0), -1)
        if target.dim() > 2:
            target = target.view(target.size(0), -1)

        source = torch.nan_to_num(source, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)

        xx = torch.mm(source, source.t())
        yy = torch.mm(target, target.t())
        xy = torch.mm(source, target.t())

        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.t() + rx - 2 * xx
        dyy = ry.t() + ry - 2 * yy
        dxy = rx.t() + ry - 2 * xy

        XX = torch.zeros(xx.shape).to(self.device)
        YY = torch.zeros(xx.shape).to(self.device)
        XY = torch.zeros(xx.shape).to(self.device)

        if self.kernel_type == "multiscale":
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx) ** -1
                YY += a**2 * (a**2 + dyy) ** -1
                XY += a**2 * (a**2 + dxy) ** -1

        elif self.kernel_type == "rbf":
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        mmd = torch.mean(XX + YY - 2 * XY)

        return mmd * 0.1


def compute_patch_mmd_loss(source, target, device=None, weight=1.0):
    """
    Compute MMD loss between source and target features.

    Args:
        source: Features from source domain (in-vivo patches)
        target: Features from target domain (ex-vivo patches)
        device: Computing device (CPU/GPU)
        weight: Loss weight

    Returns:
        torch.Tensor: MMD loss
    """

    if device is None:
        device = source.device if isinstance(source, torch.Tensor) else "cuda"

    if isinstance(source, list):
        source = torch.cat([s.flatten(start_dim=1) for s in source], dim=1)
    if isinstance(target, list):
        target = torch.cat([t.flatten(start_dim=1) for t in target], dim=1)

    source = source.view(source.size(0), -1) if source.dim() > 2 else source
    target = target.view(target.size(0), -1) if target.dim() > 2 else target

    source = torch.nan_to_num(source, nan=0.0)
    target = torch.nan_to_num(target, nan=0.0)
    mmd_loss_fn = SimplifiedMMDLoss(kernel_type="multiscale", device=device)

    if isinstance(source, list) and isinstance(target, list):
        total_loss = 0
        valid_layer_count = 0

        for i, (src_feat, tgt_feat) in enumerate(zip(source, target)):
            if src_feat.shape[2:] != tgt_feat.shape[2:]:
                src_feat = F.interpolate(
                    src_feat, size=tgt_feat.shape[2:], mode="trilinear"
                )

            src_feat = src_feat.view(src_feat.size(0), -1)
            tgt_feat = tgt_feat.view(tgt_feat.size(0), -1)

            layer_loss = mmd_loss_fn(src_feat, tgt_feat)
            total_loss += layer_loss
            valid_layer_count += 1

        if valid_layer_count > 0:
            return (total_loss / valid_layer_count) * weight
        else:
            return torch.tensor(0.01, device=device) * weight
    else:
        return mmd_loss_fn(source, target) * weight
