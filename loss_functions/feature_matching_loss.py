from torch.nn import functional as F


def feature_matching_loss(real_features, fake_features, weights=None):
    """
    Compute feature matching loss between real and fake feature maps.

    Matches the statistics of features extracted by the discriminator for real and fake images.
    Performs automatic resizing if feature shapes don't match.

    Args:
        real_features: List of feature maps from real images
        fake_features: List of feature maps from fake/generated images
        weights: List of weights for each feature layer, defaults to progressive weighting

    Returns:
        torch.Tensor: Weighted feature matching loss
    """
    if weights is None:
        weights = [
            0.5 + i / (len(real_features) - 1) for i in range(len(real_features))
        ]

    total_loss = 0.0

    for i, (real_feat, fake_feat) in enumerate(zip(real_features, fake_features)):
        if real_feat.shape != fake_feat.shape:
            fake_feat_resized = F.interpolate(
                fake_feat,
                size=real_feat.shape[2:],
                mode="trilinear" if len(real_feat.shape) == 5 else "bilinear",
                align_corners=True,
            )
            layer_loss = F.l1_loss(fake_feat_resized, real_feat.detach())
        else:
            layer_loss = F.l1_loss(fake_feat, real_feat.detach())

        total_loss = total_loss + (layer_loss * weights[i])

    return total_loss
