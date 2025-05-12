import os

import numpy as np
import torch
import torch.nn.functional as F
from monai.bundle import ConfigParser, download
from scipy import linalg


class ProstateMRIFeatureMetrics:
    """
    Feature extractor class for prostate MRI images using pre-trained models.

    This class leverages anatomy-aware models to extract meaningful features
    for perceptual loss calculation and similarity metrics specific to
    prostate MRI data.

    Args:
        model_name (str): Name of the pre-trained model. Default: 'prostate_mri_anatomy'
        model_version (str): Version of the model. Default: '0.3.1'
        use_layers (list): List of layer names to extract features from. Default: None
        layer_weights (dict): Dictionary of layer weights for feature extraction. Default: None
        device (str): Device for computation. Default: None (auto-detect)
    """

    def __init__(
        self,
        model_name="prostate_mri_anatomy",
        model_version="0.3.1",
        use_layers=None,
        layer_weights=None,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.zoo_dir = os.path.abspath("./models")
        download(name=model_name, version=model_version, bundle_dir=self.zoo_dir)

        model_config_file = os.path.join(
            self.zoo_dir, model_name, "configs", "inference.json"
        )
        self.model_config = ConfigParser()
        self.model_config.read_config(model_config_file)

        full_model = self.model_config.get_parsed_content("network").to(self.device)
        checkpoint = os.path.join(self.zoo_dir, model_name, "models", "model.pt")
        full_model.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )

        self.encoders = self._extract_encoder_modules(full_model)

        for module in self.encoders.values():
            module.eval()

        self.use_layers = use_layers or [
            "model.submodule.encoder1",
            "model.submodule.encoder3",
            "model.submodule.encoder5",
        ]

        self.layer_weights = layer_weights or {
            "model.submodule.encoder1": 0.2,
            "model.submodule.encoder3": 0.3,
            "model.submodule.encoder5": 0.5,
        }

    def _extract_encoder_modules(self, full_model):
        """
        Extract encoder modules from the full model.

        Args:
            full_model (torch.nn.Module): Full pre-trained model

        Returns:
            dict: Dictionary of encoder modules
        """
        encoders = {}

        if hasattr(full_model, "model") and hasattr(full_model.model, "0"):
            encoders["initial_conv"] = full_model.model[0]

        module_mapping = {
            "model.submodule.encoder1": full_model.model[0],
            "model.submodule.encoder3": full_model.model[1].submodule[0],
            "model.submodule.encoder5": full_model.model[1].submodule[1].submodule[0],
        }

        if hasattr(full_model.model[1], "down1"):
            encoders["down1"] = full_model.model[1].down1

        if hasattr(full_model.model[1].submodule[1], "down2"):
            encoders["down2"] = full_model.model[1].submodule[1].down2

        for name, module in module_mapping.items():
            encoders[name] = module

        return encoders

    def extract_features(self, image_batch):
        """
        Extract features from image batch using pre-trained encoder modules.

        Args:
            image_batch (torch.Tensor): Batch of images [B,C,D,H,W] or [B,C,H,W]

        Returns:
            dict: Dictionary of features from different layers
        """
        if len(image_batch.shape) == 4:
            image_batch = image_batch.unsqueeze(2)

        if image_batch.size(1) > 1:
            image_batch = image_batch.mean(dim=1, keepdim=True)

        features = {}
        with torch.no_grad():
            x = self.encoders["initial_conv"](image_batch)
            features["model.submodule.encoder1"] = x

            if "down1" in self.encoders:
                x = self.encoders["down1"](x)

            if "model.submodule.encoder3" in self.encoders:
                x = self.encoders["model.submodule.encoder3"](x)
                features["model.submodule.encoder3"] = x

            if "down2" in self.encoders:
                x = self.encoders["down2"](x)

            if "model.submodule.encoder5" in self.encoders:
                x = self.encoders["model.submodule.encoder5"](x)
                features["model.submodule.encoder5"] = x

        return features

    def preprocess(self, x):
        """
        Preprocess input tensor for feature extraction.

        Normalizes the tensor and pads dimensions to multiples of 16.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Preprocessed tensor
        """
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)

        _, _, d, h, w = x_norm.shape if len(x_norm.shape) == 5 else (*x_norm.shape, 1)

        pad_d = (16 - d % 16) % 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_norm = F.pad(x_norm, (0, pad_w, 0, pad_h, 0, pad_d), mode="reflect")

        return x_norm

    def perceptual_loss(self, real_images, generated_images):
        """
        Calculate perceptual loss between real and generated images.

        Uses weighted L1 loss on deep features from different network layers.

        Args:
            real_images (torch.Tensor): Real images
            generated_images (torch.Tensor): Generated images

        Returns:
            torch.Tensor: Perceptual loss value
        """
        real_images = self.preprocess(real_images)
        generated_images = self.preprocess(generated_images)

        real_features = self.extract_features(real_images)
        gen_features = self.extract_features(generated_images)

        loss = 0.0
        for layer_name in self.use_layers:
            if layer_name in real_features and layer_name in gen_features:
                layer_loss = F.l1_loss(
                    real_features[layer_name], gen_features[layer_name]
                )
                loss += self.layer_weights[layer_name] * layer_loss

        return loss

    def _channel_normalize(self, x):
        """
        Normalize tensor along channel dimension.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Channel-normalized tensor
        """
        norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + 1e-8)
        return x / (norm_factor + 1e-8)

    def calculate_lpips(self, reference_images, generated_images):
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity) between images.

        Args:
            reference_images (torch.Tensor): Reference images
            generated_images (torch.Tensor): Generated images

        Returns:
            torch.Tensor: LPIPS score (lower is better)
        """
        ref_prep = self.preprocess(reference_images)
        gen_prep = self.preprocess(generated_images)

        ref_features = self.extract_features(ref_prep)
        gen_features = self.extract_features(gen_prep)

        lpips_score = 0.0
        for layer_name in self.use_layers:
            if layer_name in ref_features and layer_name in gen_features:
                feat_ref = ref_features[layer_name]
                feat_gen = gen_features[layer_name]

                feat_ref_norm = self._channel_normalize(feat_ref)
                feat_gen_norm = self._channel_normalize(feat_gen)

                dist = torch.mean((feat_ref_norm - feat_gen_norm) ** 2)
                lpips_score += self.layer_weights[layer_name] * dist

        return lpips_score

    def calculate_fid(self, real_images, generated_images, feature_layer=None):
        """
        Calculate FID (Fréchet Inception Distance) between image distributions.

        Args:
            real_images (torch.Tensor): Real images
            generated_images (torch.Tensor): Generated images
            feature_layer (str): Layer to extract features from. Default: encoder5

        Returns:
            float: FID score (lower is better)
        """
        feature_layer = feature_layer or "model.submodule.encoder5"

        real_prep = self.preprocess(real_images)
        gen_prep = self.preprocess(generated_images)

        real_features = self.extract_features(real_prep)
        gen_features = self.extract_features(gen_prep)

        if feature_layer not in real_features or feature_layer not in gen_features:
            raise ValueError(f"Feature layer {feature_layer} not found")

        real_feats = real_features[feature_layer]
        gen_feats = gen_features[feature_layer]

        real_feats = real_feats.view(real_feats.size(0), -1).cpu().numpy()
        gen_feats = gen_feats.view(gen_feats.size(0), -1).cpu().numpy()

        mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
        mu2, sigma2 = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = (
            diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        )

        return float(fid)
