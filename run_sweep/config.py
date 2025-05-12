from pathlib import Path

import yaml


def load_base_config(config_name="sweep_baseline"):
    config_path = Path("../configurations") / f"{config_name}.yaml"
    if not config_path.exists():
        config_path = Path("../configurations") / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find configuration file: {config_name}.yaml"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_sweep_config():
    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "weighted_metric",
            "goal": "minimize",
        },
        "parameters": {
            "gan_loss_combo": {
                "values": [
                    "lsgan_only",
                    "hinge_only",
                    "wasserstein_only",
                    "hinge_relativistic",
                ]
            },
            "cycle_loss_combo": {
                "values": [
                    "l1",
                    "l2",
                    "ssim",
                    "perceptual",
                    "l1+l2",
                    "l1+ssim",
                    "l1+perceptual",
                    "l2+ssim",
                    "l2+perceptual",
                    "ssim+perceptual",
                ]
            },
            "identity_loss_combo": {
                "values": [
                    "l1",
                    "l2",
                    "ssim",
                    "perceptual",
                    "l1+l2",
                    "l1+ssim",
                    "l1+perceptual",
                    "l2+ssim",
                    "l2+perceptual",
                    "ssim+perceptual",
                ]
            },
            "optimization.disc_update_freq": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 5,
            },
            "optimization.constant_for_a": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.7,
            },
            "optimization.learning_rate_D_constant": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.6,
            },
            "losses.lambda_domain_adaptation": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 3.0,
            },
            "losses.lambda_da_histogram": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 3.0,
            },
            "losses.lambda_da_contrast": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 3.0,
            },
            "losses.lambda_da_structure": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 1.0,
            },
            "losses.lambda_da_gradient": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 1.0,
            },
            "losses.lambda_da_ncc": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 1.0,
            },
            "losses.lambda_da_texture": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 3.0,
            },
            "model.use_residual": {"values": [True, False]},
            "model.use_stn": {"values": [True, False]},
            "losses.lambda_cycle_a": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 10.0,
            },
            "losses.lambda_cycle_b": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 10.0,
            },
            "losses.lambda_identity": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 1.0,
            },
            "losses.lambda_ganloss_a": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 5.0,
            },
            "losses.lambda_ganloss_b": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 5.0,
            },
            "losses.lambda_feature_matching": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 10.0,
            },
        },
        "early_terminate": {"type": "hyperband", "min_iter": 3, "eta": 2, "s": 2},
        "max_runs": 400,
    }

    return sweep_config


def validate_sweep_config(config):
    lambda_da = config.get("losses.lambda_domain_adaptation", 0.0)

    if lambda_da > 0:
        da_params = [
            "losses.lambda_da_histogram",
            "losses.lambda_da_contrast",
            "losses.lambda_da_structure",
            "losses.lambda_da_gradient",
            "losses.lambda_da_ncc",
            "losses.lambda_da_texture",
        ]

        all_zero = True
        for param in da_params:
            if param in config and config[param] > 0:
                all_zero = False
                break

        if all_zero:
            return False, "Domain adaptation enabled but all sub-lambdas are zero"

    return True, "Valid configuration"


def map_gan_loss_type(config, base_config):
    gan_loss_combo = config.get("gan_loss_combo", "lsgan_only")

    base_config["optimization"]["use_hinge"] = False
    base_config["optimization"]["use_lsgan"] = False
    base_config["optimization"]["use_wasserstein"] = False
    base_config["optimization"]["use_relativistic"] = False

    if gan_loss_combo == "lsgan_only":
        base_config["optimization"]["use_lsgan"] = True
    elif gan_loss_combo == "hinge_only":
        base_config["optimization"]["use_hinge"] = True
    elif gan_loss_combo == "wasserstein_only":
        base_config["optimization"]["use_wasserstein"] = True
    elif gan_loss_combo == "hinge_relativistic":
        base_config["optimization"]["use_hinge"] = True
        base_config["optimization"]["use_relativistic"] = True

    return base_config


def map_categorical_loss_combos(config, base_config):
    cycle_combo = config.get("cycle_loss_combo", "l1")

    if "+" in cycle_combo:
        cycle_type_1, cycle_type_2 = cycle_combo.split("+")
    else:
        cycle_type_1 = cycle_combo
        cycle_type_2 = "None"

    base_config["losses"]["cycle_loss_type_1"] = cycle_type_1
    base_config["losses"]["cycle_loss_type_2"] = cycle_type_2

    identity_combo = config.get("identity_loss_combo", "l1")

    if "+" in identity_combo:
        identity_type_1, identity_type_2 = identity_combo.split("+")
    else:
        identity_type_1 = identity_combo
        identity_type_2 = "None"

    base_config["losses"]["identity_loss_type_1"] = identity_type_1
    base_config["losses"]["identity_loss_type_2"] = identity_type_2

    return base_config


def apply_domain_adaptation_params(config, base_config):
    lambda_domain_adaptation = config.get("losses.lambda_domain_adaptation", 0.0)

    base_config["losses"]["lambda_domain_adaptation"] = lambda_domain_adaptation

    if lambda_domain_adaptation > 0.0:
        da_params = [
            "lambda_da_histogram",
            "lambda_da_contrast",
            "lambda_da_structure",
            "lambda_da_gradient",
            "lambda_da_ncc",
            "lambda_da_texture",
        ]

        for param in da_params:
            config_key = f"losses.{param}"
            if config_key in config:
                base_config["losses"][param] = config[config_key]
    else:
        base_config["losses"]["lambda_da_histogram"] = 0.0
        base_config["losses"]["lambda_da_contrast"] = 0.0
        base_config["losses"]["lambda_da_structure"] = 0.0
        base_config["losses"]["lambda_da_gradient"] = 0.0
        base_config["losses"]["lambda_da_ncc"] = 0.0
        base_config["losses"]["lambda_da_texture"] = 0.0

    return base_config


def reduce_training_epochs(base_config):
    base_config["training"]["niter"] = 2
    base_config["training"]["niter_decay"] = 2
    return base_config
