# setup/config_loader.py

import yaml
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    def __init__(self, config_dir: str = "../configurations"):
        self.config_dir = config_dir
        self.base_config = self._load_yaml("base_config.yaml")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        filepath = os.path.join(self.config_dir, filename)
        with open(filepath, "r") as f:
            return yaml.safe_load(f)

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load and merge configuration with base config."""
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

        config = self._load_yaml(config_name)
        merged_config = {**self.base_config, **config}
        return merged_config

    def get_model_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters needed for model name construction."""
        return {
            "ngf": config["network"]["ngf"],
            "ndf": config["network"]["ndf"],
            "patch_x": config["patch"]["x"],
            "patch_y": config["patch"]["y"],
            "patch_z": config["patch"]["z"],
            "patches_per_image": config["training"]["patches_per_image"],
        }

    def build_args_string(
        self, config: Dict[str, Any], is_training: bool = True
    ) -> str:
        """Convert configuration to command line arguments string."""
        args = []

        # Network args (always included)
        args.extend(
            [
                f"--ngf={config['network']['ngf']}",
                f"--ndf={config['network']['ndf']}",
            ]
        )

        # Patch args (always included)
        args.extend(
            [
                f"--patch_size={config['patch']['x']},{config['patch']['y']},{config['patch']['z']}"
            ]
        )

        # Optimization args (some only for training)
        args.append(f"--min_pixel={config['optimization']['min_pixel']}")
        args.append(f"--use_spectral_norm_G")
        args.append(f"--group_name={config['group_name']}")

        if is_training:
            args.extend(
                [
                    f"--batch_size={config['optimization']['batch_size']}",
                    f"--lr={config['optimization']['learning_rate']}",
                    f"--lr_D_Constant={config['optimization']['learning_rate_D_constant']}",
                    # Add scheduler parameters if they exist
                    f"--lr_restart_epochs={config['optimization'].get('lr_restart_epochs', 10)}",
                    f"--lr_restart_mult={config['optimization'].get('lr_restart_mult', 2)}",
                    f"--lr_min_G={config['optimization'].get('lr_min_G', 0.000001)}",
                    f"--lr_min_D={config['optimization'].get('lr_min_D', 0.0000005)}",
                ]
            )

        if is_training:
            # Training specific args
            args.extend(
                [
                    f"--save_latest_freq={config['training']['save_latest_freq']}",
                    f"--print_freq={config['training']['print_freq']}",
                    f"--patches_per_image={config['training']['patches_per_image']}",
                    f"--save_epoch_freq={config['training']['save_epoch_freq']}",
                    f"--niter={config['training']['niter']}",
                    f"--niter_decay={config['training']['niter_decay']}",
                    f"--accumulation_steps={config['training']['accumulation_steps']}",
                    # Lambda scheduler setting
                    # Loss weights
                    f"--lambda_identity={config['losses']['lambda_identity']}",
                    f"--lambda_domain_adaptation={config['losses']['lambda_domain_adaptation']}",
                    f"--lambda_mmd={config['losses']['lambda_mmd']}",
                    f"--lambda_mmd_bridge={config['losses'].get('lambda_mmd_bridge', 0.0)}",
                    f"--lambda_da_contrast={config['losses']['lambda_da_contrast']}",
                    f"--lambda_da_structure={config['losses']['lambda_da_structure']}",
                    f"--lambda_da_texture={config['losses']['lambda_da_texture']}",
                    f"--lambda_da_histogram={config['losses']['lambda_da_histogram']}",
                    f"--lambda_da_ncc={config['losses']['lambda_da_ncc']}",
                    f"--lambda_da_gradient={config['losses']['lambda_da_gradient']}",
                    f"--lambda_vgg={config['losses']['lambda_vgg']}",
                    f"--lambda_feature_matching={config['losses']['lambda_feature_matching']}",
                    f"--lambda_cycle_A={config['losses']['lambda_cycle_a']}",
                    f"--lambda_cycle_B={config['losses']['lambda_cycle_b']}",
                    f"--lambda_ganloss_A={config['losses']['lambda_ganloss_a']}",
                    f"--lambda_ganloss_B={config['losses']['lambda_ganloss_b']}",
                    f"--cycle_loss_type_1={config['losses']['cycle_loss_type_1']}",
                    f"--cycle_loss_type_2={config['losses']['cycle_loss_type_2']}",
                    f"--identity_loss_type_1={config['losses']['identity_loss_type_1']}",
                    f"--identity_loss_type_2={config['losses']['identity_loss_type_2']}",
                    # model
                    # Discriminator args
                    # Registration
                    f"--constant_for_A={config['optimization']['constant_for_a']}",
                    # Lambda weight scheduler parameters
                    f"--identity_phase_out_start={config['losses'].get('identity_phase_out_start', 0.1)}",
                    f"--identity_phase_out_end={config['losses'].get('identity_phase_out_end', 0.4)}",
                    # Note the mismatched name in YAML
                    f"--gan_phase_in_start={config['losses'].get('gan_phase_in_start', 0.05)}",
                    f"--gan_phase_in_end={config['losses'].get('gan_phase_in_end', 0.3)}",
                    # Note the mismatched name in YAML
                    f"--domain_adaptation_phase_in_start={config['losses'].get('domain_adaptation_phase_in_start', 0.2)}",
                    f"--domain_adaptation_phase_in_end={config['losses'].get('domain_adaptation_phase_in_end', 0.6)}",
                    f"--domain_adaptation_scale_max={config['losses'].get('domain_adaptation_scale_max', 1.5)}",
                    f"--cycle_adjust_start={config['losses'].get('cycle_adjust_start', 0.3)}",
                    f"--cycle_adjust_end={config['losses'].get('cycle_adjust_end', 0.7)}",
                    f"--cycle_scale_min={config['losses'].get('cycle_scale_min', 0.7)}",
                    f"--min_identity_weight={config['losses'].get('min_identity_weight', 0.05)}",
                    f"--run_validation_interval={config['training'].get('run_validation_interval', 1)}",
                    f"--mixed_precision",
                    f"--use_gradient_checkpointing",
                    f"--use_full_validation",
                    f"--vis_freq={config['training'].get('vis_freq', 10)}",
                    f"--vis_max_channels={config['training'].get('vis_max_channels', 16)}",
                    f"--disc_update_freq={config['optimization'].get('disc_update_freq', 1)}",
                ]
            )
            if config["training"]["visualize_features"]:
                args.append(f"--visualize_features")
            if config["losses"]["use_lambda_scheduler"]:
                args.append("--use_lambda_scheduler")
            if config["optimization"]["use_lsgan"]:
                args.append("--use_lsgan")
            if config["optimization"]["use_hinge"]:
                args.append("--use_hinge")
            if config["optimization"]["use_wasserstein"]:
                args.append("--use_wasserstein")
            if config["optimization"]["use_relativistic"]:
                args.append("--use_relativistic")

            if config["model"]["use_stn"]:
                args.append("--use_stn")
            if config["model"]["use_residual"]:
                args.append("--use_residual")
            if config["model"]["use_full_attention"]:
                args.append("--use_full_attention")
        # Common args from base config (always included)
        args.extend([f"--{key}={value}" for key, value in config["common"].items()])

        return " ".join(args)
