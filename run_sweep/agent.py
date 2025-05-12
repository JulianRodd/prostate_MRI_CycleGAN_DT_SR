import os
import gc
import re
import json
import subprocess
import yaml
import wandb
import psutil
from config import (
    load_base_config,
    validate_sweep_config,
    map_gan_loss_type,
    map_categorical_loss_combos,
    apply_domain_adaptation_params,
    reduce_training_epochs,
)
from metrics import calculate_weighted_metric
from cleanup import cleanup_sweep_resources


def run_sweep_agent(use_kaggle=False):
    run = None

    try:
        run = wandb.init(reinit=True)

        is_valid, reason = validate_sweep_config(wandb.config)
        if not is_valid:
            print(f"Skipping invalid configuration: {reason}")
            wandb.log(
                {
                    "skipped": True,
                    "val_fid_domain": 999.9,
                    "weighted_metric": 1.0,
                    "skip_reason": reason,
                }
            )
            return 0

        return run_sweep_trial(run, use_kaggle)

    except Exception as e:
        print(f"Error in sweep agent: {str(e)}")
        if run:
            wandb.log({"error": str(e)})
        return 1

    finally:
        gc.collect()

        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared CUDA cache")
        except:
            pass


def run_sweep_trial(run, use_kaggle=False):
    run_id = f"{run.id[-6:]}"

    temp_config_file = f"configurations/sweep_{run_id}.yaml"
    temp_output_dir = f"results/sweep_{run_id}"

    metrics = {}
    process = None

    try:
        base_config = load_base_config("sweep_baseline")

        base_config = map_gan_loss_type(wandb.config, base_config)
        base_config = map_categorical_loss_combos(wandb.config, base_config)
        base_config = apply_domain_adaptation_params(wandb.config, base_config)
        base_config = reduce_training_epochs(base_config)

        if use_kaggle:
            if (
                "data_path" in base_config
                and "/kaggle/input" in base_config["data_path"]
            ):
                print(f"Detected Kaggle input path: {base_config['data_path']}")
                base_config["patch_output_dir"] = "/kaggle/working/patches"
                print(
                    f"Setting patch output directory to: {base_config['patch_output_dir']}"
                )

            base_config["temp_dir"] = "/kaggle/working/temp"
            print(f"Setting temp directory to: {base_config['temp_dir']}")

            os.makedirs("/kaggle/working/patches", exist_ok=True)
            os.makedirs("/kaggle/working/temp", exist_ok=True)
            os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
            os.makedirs("/kaggle/working/results", exist_ok=True)

        for key, value in wandb.config.items():
            if key in ["gan_loss_combo", "cycle_loss_combo", "identity_loss_combo"]:
                continue

            if key.startswith("losses.lambda_da_"):
                continue

            parts = key.split(".")
            config_level = base_config
            for part in parts[:-1]:
                if part not in config_level:
                    config_level[part] = {}
                config_level = config_level[part]

            config_level[parts[-1]] = value

        wandb.log(
            {
                "config_summary": {
                    "gan_loss": json.dumps(
                        {
                            "combo": wandb.config.get("gan_loss_combo", "lsgan_only"),
                            "use_hinge": base_config["optimization"]["use_hinge"],
                            "use_lsgan": base_config["optimization"]["use_lsgan"],
                            "use_wasserstein": base_config["optimization"][
                                "use_wasserstein"
                            ],
                            "use_relativistic": base_config["optimization"][
                                "use_relativistic"
                            ],
                        }
                    ),
                    "disc_update_freq": base_config["optimization"].get(
                        "disc_update_freq", 1
                    ),
                    "cycle_loss": json.dumps(
                        {
                            "type_1": base_config["losses"]["cycle_loss_type_1"],
                            "type_2": base_config["losses"]["cycle_loss_type_2"],
                            "lambda_a": base_config["losses"]["lambda_cycle_a"],
                            "lambda_b": base_config["losses"]["lambda_cycle_b"],
                        }
                    ),
                    "identity_loss": json.dumps(
                        {
                            "type_1": base_config["losses"]["identity_loss_type_1"],
                            "type_2": base_config["losses"]["identity_loss_type_2"],
                            "lambda": base_config["losses"]["lambda_identity"],
                        }
                    ),
                    "feature_matching_lambda": base_config["losses"][
                        "lambda_feature_matching"
                    ],
                    "domain_adaptation": json.dumps(
                        {
                            "enabled": base_config["losses"]["lambda_domain_adaptation"]
                            > 0,
                            "main_lambda": base_config["losses"][
                                "lambda_domain_adaptation"
                            ],
                            "histogram": base_config["losses"]["lambda_da_histogram"],
                            "contrast": base_config["losses"]["lambda_da_contrast"],
                            "structure": base_config["losses"]["lambda_da_structure"],
                            "gradient": base_config["losses"]["lambda_da_gradient"],
                            "ncc": base_config["losses"]["lambda_da_ncc"],
                            "texture": base_config["losses"]["lambda_da_texture"],
                        }
                    ),
                }
            }
        )

        os.makedirs(os.path.dirname(temp_config_file), exist_ok=True)

        with open(temp_config_file, "w") as f:
            yaml.dump(base_config, f)

        train_cmd = "train-kaggle" if use_kaggle else "train-server"
        cmd = ["bash", "commands/run.sh", train_cmd, f"sweep_{run_id}"]

        print(f"Running: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        metrics = {}

        for line in process.stdout:
            print(line, end="")

            fid_match = re.search(r"val_fid_domain:\s*([\d.]+)", line)
            if fid_match:
                metrics["val_fid_domain"] = float(fid_match.group(1))
                wandb.log({"val_fid_domain": metrics["val_fid_domain"]})

            ssim_match = re.search(r"val_ssim_sr:\s*([\d.]+)", line)
            if ssim_match:
                metrics["val_ssim_sr"] = float(ssim_match.group(1))
                wandb.log({"val_ssim_sr": metrics["val_ssim_sr"]})

            ncc_match = re.search(r"val_ncc_domain:\s*([\d.]+)", line)
            if ncc_match:
                metrics["val_ncc_domain"] = float(ncc_match.group(1))
                wandb.log({"val_ncc_domain": metrics["val_ncc_domain"]})

            psnr_match = re.search(r"val_psnr_sr:\s*([\d.]+)", line)
            if psnr_match:
                metrics["val_psnr_sr"] = float(psnr_match.group(1))
                wandb.log({"val_psnr_sr": metrics["val_psnr_sr"]})

            lpips_match = re.search(r"val_lpips_sr:\s*([\d.]+)", line)
            if lpips_match:
                metrics["val_lpips_sr"] = float(lpips_match.group(1))
                wandb.log({"val_lpips_sr": metrics["val_lpips_sr"]})

            da_match = re.search(r"val_da_(\w+):\s*([\d.]+)", line)
            if da_match:
                metric_name = f"val_da_{da_match.group(1)}"
                metrics[metric_name] = float(da_match.group(2))
                wandb.log({metric_name: metrics[metric_name]})

            cycle_loss_match = re.search(
                r"validation/cycle_loss_total:\s*([\d.]+)", line
            )
            if cycle_loss_match:
                metrics["validation/cycle_loss_total"] = float(
                    cycle_loss_match.group(1)
                )
                wandb.log(
                    {
                        "validation/cycle_loss_total": metrics[
                            "validation/cycle_loss_total"
                        ]
                    }
                )

            disc_loss_match = re.search(r"validation/disc_loss:\s*([\d.]+)", line)
            if disc_loss_match:
                metrics["validation/disc_loss"] = float(disc_loss_match.group(1))
                wandb.log({"validation/disc_loss": metrics["validation/disc_loss"]})

            identity_loss_match = re.search(
                r"validation/identity_loss:\s*([\d.]+)", line
            )
            if identity_loss_match:
                metrics["validation/identity_loss"] = float(
                    identity_loss_match.group(1)
                )
                wandb.log(
                    {"validation/identity_loss": metrics["validation/identity_loss"]}
                )

            epoch_match = re.search(r"End of epoch (\d+)", line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                wandb.log({"epoch": epoch})

                if any(
                    k in metrics
                    for k in [
                        "val_fid_domain",
                        "val_ssim_sr",
                        "val_psnr_sr",
                        "val_lpips_sr",
                        "val_ncc_domain",
                    ]
                ):
                    weighted_metric = calculate_weighted_metric(metrics)
                    wandb.log({"weighted_metric": weighted_metric})

                if metrics:
                    wandb.log({f"summary/{k}": v for k, v in metrics.items()})

        if process:
            process.wait()

        if any(
            k in metrics
            for k in [
                "val_fid_domain",
                "val_ssim_sr",
                "val_psnr_sr",
                "val_lpips_sr",
                "val_ncc_domain",
            ]
        ):
            weighted_metric = calculate_weighted_metric(metrics)
            wandb.log(
                {
                    "final/weighted_metric": weighted_metric,
                    "weighted_metric": weighted_metric,
                }
            )

        if metrics:
            wandb.log({f"final/{k}": v for k, v in metrics.items()})

        return process.returncode if process else 1

    finally:
        gc.collect()
        cleanup_sweep_resources(run_id, temp_config_file, temp_output_dir, use_kaggle)

        process_info = psutil.Process()
        memory_info = process_info.memory_info()
        wandb.log({"memory_usage_mb": memory_info.rss / 1024 / 1024})

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared CUDA cache")
        except:
            pass

        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except:
            pass
