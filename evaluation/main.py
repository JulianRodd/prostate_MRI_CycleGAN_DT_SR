import os
import sys
import gc
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options.base_options import BaseOptions
from models.cycle_gan_model import CycleGANModel
from utils.utils import set_seed, init_wandb

# Import modules from our reorganized files
from masking import create_mask_from_invivo, apply_mask_to_exvivo
from metrics import (
    evaluate_slice_based_fid,
    calculate_additional_metrics_for_validation,
    log_fid_to_wandb,
)
from visualization import save_middle_slice_visualizations
from data_processing import load_train_exvivo_data, parse_model_name
from sliding_window import process_sample_with_sliding_window


class FIDEvalOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--models",
            nargs="+",
            type=str,
            default=None,
            help="Specific model names to evaluate (default: all models in checkpoints dir)",
        )

        parser.add_argument(
            "--csv_path",
            type=str,
            default=None,
            help="Path to CSV file containing model configurations to evaluate",
        )

        parser.add_argument(
            "--output_file",
            type=str,
            default=None,
            help="Path to output CSV file for results",
        )

        parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="Which epoch to load for evaluation",
        )
        parser.add_argument(
            "--use_full_validation",
            action="store_true",
            help="Use full images for validation instead of patches",
            default=True,
        )
        parser.add_argument(
            "--patches_per_image",
            type=int,
            required=True,
            help="Number of patches per image",
        )

        # Rest of existing arguments...
        parser.add_argument(
            "--use_spectral_norm_G",
            action="store_true",
            help="use spectral normalization in generator",
        )
        parser.add_argument(
            "--use_stn",
            action="store_true",
            help="use spatial transformer network in generator",
            default=False,
        )
        parser.add_argument(
            "--use_residual",
            action="store_true",
            help="use residual blocks in generator",
            default=False,
        )
        parser.add_argument(
            "--use_full_attention",
            action="store_true",
            help="use full attention in generator",
            default=False,
        )

        parser.add_argument(
            "--use_lsgan",
            action="store_true",
            help="use least squares GAN",
            default=True,
        )
        parser.add_argument(
            "--use_hinge", action="store_true", help="use hinge loss", default=False
        )
        parser.add_argument(
            "--use_relativistic",
            action="store_true",
            help="use relativistic discriminator",
            default=False,
        )
        parser.add_argument(
            "--lambda_identity",
            type=float,
            default=0.5,
            help="weight for identity loss",
        )

        parser.add_argument(
            "--lambda_domain_adaptation",
            type=float,
            default=1.0,
            help="weight for domain adaptation loss",
        )
        parser.add_argument(
            "--lambda_da_contrast",
            type=float,
            default=1.0,
            help="weight for contrast component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_structure",
            type=float,
            default=1.0,
            help="weight for structure component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_texture",
            type=float,
            default=1.0,
            help="weight for texture component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_histogram",
            type=float,
            default=0.0,
            help="weight for histogram component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_gradient",
            type=float,
            default=0.0,
            help="weight for gradient component in domain adaptation",
        )
        parser.add_argument(
            "--lambda_da_ncc",
            type=float,
            default=0.0,
            help="weight for NCC component in domain adaptation",
        )

        # Added memory optimization parameters
        parser.add_argument(
            "--batch_slice_processing",
            action="store_true",
            help="Process slices in batches to reduce memory usage",
            default=True,
        )
        parser.add_argument(
            "--slice_batch_size",
            type=int,
            default=32,
            help="Batch size for slice processing",
        )
        parser.add_argument(
            "--memory_profiling",
            action="store_true",
            help="Enable memory profiling",
            default=False,
        )

        # Check if patch_size already exists in the parser to avoid conflicts
        has_patch_size = False
        for action in parser._actions:
            if action.dest == "patch_size":
                has_patch_size = True
                break

        # Only add patch_size if it doesn't already exist
        if not has_patch_size:
            parser.add_argument(
                "--patch_size",
                nargs="+",
                type=int,
                default=[64, 64, 32],
                help="Size of patches for sliding window approach [D, H, W]",
            )
            parser.add_argument(
                "--min_patch_size",
                nargs="+",
                type=int,
                default=[16, 16, 8],
                help="Minimum size of valid patches [D, H, W]",
            )

        self.isTrain = False
        return parser


def evaluate_model(
    model_name, opt, validation_data, device, train_exvivo_data=None, epoch="latest"
):
    """
    Evaluate a model and return FID scores and additional metrics.
    """
    print(f"\n{'=' * 50}")
    print(f"Evaluating model: {model_name}")
    print(f"Using checkpoint: {epoch}")
    print(f"{'=' * 50}")

    # Reset CUDA cache before each model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    opt.name = model_name
    opt = parse_model_name(model_name, opt)

    # Set which epoch to use
    opt.which_epoch = epoch

    # Set base directory for visualizations
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    opt.use_residual = True
    opt.n_layers_D = 4
    opt.mixed_precision = True
    try:
        # Initialize model with memory optimization settings
        print("Initializing model with memory optimization settings...")

        # Try to enable gradient checkpointing if supported to reduce memory usage
        opt.use_gradient_checkpointing = True

        model = CycleGANModel()
        model.initialize(opt)

        if torch.cuda.is_available():
            opt.gpu_ids = [0]
        else:
            opt.gpu_ids = []

        model.device = device

        # Enable eval mode to reduce memory usage
        model.eval()

        print(f"Loading model from: checkpoints/{model_name}/{epoch}_net_G_A.pth")
        model.load_networks(epoch)

        # Calculate FID scores with visualization
        print("Running FID evaluation with memory optimizations...")
        fid_result = evaluate_slice_based_fid(
            model, validation_data, device, train_exvivo_data, model_name, base_dir
        )

        # Calculate additional metrics
        print("Calculating additional metrics (PSNR, SSIM, LPIPS, NCC)...")
        additional_metrics = calculate_additional_metrics_for_validation(
            model, validation_data, device
        )

        # Combine all metrics
        result = {**fid_result, **additional_metrics}

        # Log results to wandb
        log_fid_to_wandb(model_name, result, opt)

        # Clean up thoroughly after evaluation
        if hasattr(model, "netG_A") and model.netG_A is not None:
            del model.netG_A
        if hasattr(model, "netG_B") and model.netG_B is not None:
            del model.netG_B
        if hasattr(model, "netD_A") and model.netD_A is not None:
            del model.netD_A
        if hasattr(model, "netD_B") and model.netD_B is not None:
            del model.netD_B

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result

    except Exception as e:
        print(f"Error evaluating model {model_name} (epoch {epoch}): {e}")
        import traceback

        traceback.print_exc()

        # Create error result with all metrics
        error_result = {
            "fid_val": float("inf"),
            "fid_train": float("inf"),
            "fid_combined": float("inf"),
            "psnr": 0.0,
            "ssim": 0.0,
            "lpips": 1.0,
            "ncc": 0.0,
            "error": str(e),
        }

        # Still try to log the error to wandb
        log_fid_to_wandb(model_name, error_result, opt)

        return error_result


def save_results_to_csv(results, output_path="fid_evaluation_results.csv"):
    """
    Save evaluation results to a CSV file
    """
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "model_name",
                "fid_val",
                "fid_train",
                "fid_combined",
                "psnr",
                "ssim",
                "lpips",
                "ncc",
                "error",
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for model_name, scores in results.items():
                row = {
                    "model_name": model_name,
                    "fid_val": scores.get("fid_val", float("inf")),
                    "fid_train": scores.get("fid_train", float("inf")),
                    "fid_combined": scores.get("fid_combined", float("inf")),
                    "psnr": scores.get("psnr", 0.0),
                    "ssim": scores.get("ssim", 0.0),
                    "lpips": scores.get("lpips", 1.0),
                    "ncc": scores.get("ncc", 0.0),
                    "error": scores.get("error", ""),
                }
                writer.writerow(row)

        print(f"Results saved to {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        import traceback

        traceback.print_exc()


def process_csv_models(csv_path, opt, validation_data, device, train_exvivo_data=None):
    """
    Process models specified in a CSV file
    """
    results = {}

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} model configurations from {csv_path}")

        # Check required columns
        required_columns = ["model_name", "epoch"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: CSV is missing required columns: {missing_columns}")
            print(f"Required columns: {required_columns}")
            return results

        # Process each model
        for idx, row in df.iterrows():
            model_name = row["model_name"]
            epoch = row["epoch"]
            print(
                f"\nProcessing CSV entry {idx + 1}/{len(df)}: {model_name}, epoch={epoch}"
            )

            # Generate a unique key that includes both model name and epoch
            result_key = f"{model_name}_epoch{epoch}"

            try:
                # Evaluate the model
                model_results = evaluate_model(
                    model_name, opt, validation_data, device, train_exvivo_data, epoch
                )
                # Store results
                results[result_key] = model_results

            except Exception as e:
                # Record the error and continue with next model
                print(f"Error processing model {model_name} (epoch {epoch}): {e}")
                results[result_key] = {
                    "fid_val": float("inf"),
                    "fid_train": float("inf"),
                    "fid_combined": float("inf"),
                    "psnr": 0.0,
                    "ssim": 0.0,
                    "lpips": 1.0,
                    "ncc": 0.0,
                    "error": str(e),
                }

            # Force cleanup after each model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            interim_file = f"interim_results_{timestamp}.csv"
            if hasattr(opt, "output_file") and opt.output_file:
                interim_file = (
                    f"{os.path.splitext(opt.output_file)[0]}_interim_{timestamp}.csv"
                )

            save_results_to_csv(results, interim_file)
            print(f"Saved interim results to {interim_file}")

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        import traceback

        traceback.print_exc()

    return results


def main():
    # Set PyTorch to use expandable segments to reduce memory fragmentation
    import os
    import math
    import wandb

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Parse options
    opt = FIDEvalOptions().parse()
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Initialize wandb if enabled
    if opt.use_wandb:
        try:
            from utils.utils import init_wandb

            init_wandb(opt)
            print("Successfully initialized WandB for FID evaluation")
        except Exception as e:
            print(f"Warning: Failed to initialize WandB: {e}")
            print("Continuing without WandB logging")
            opt.use_wandb = False

    # Configure CUDA for memory efficiency
    if torch.cuda.is_available():
        # Empty cache at start
        torch.cuda.empty_cache()

        # Print initial CUDA memory stats
        print(
            f"Initial CUDA memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB"
        )
        print(
            f"Initial CUDA memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB"
        )

        # Set memory efficient options
        torch.backends.cudnn.benchmark = False  # Disable cudnn benchmarking
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = (
                True  # Allow TF32 for faster computation
            )
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True  # Allow TF32 in cudnn

    # Enable memory profiling if requested
    if opt.memory_profiling:
        try:
            import psutil

            print("Memory profiling enabled")
        except ImportError:
            print("Warning: psutil not available for memory profiling")
            opt.memory_profiling = False

    set_seed(42)

    # Memory optimization settings
    opt.use_full_validation = True
    opt.batch_size = 1  # Process one volume at a time

    # Set additional memory optimization parameters
    opt.slice_batch_size = 8  # Process slices in smaller batches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if we're evaluating from CSV or direct model list
    if opt.csv_path is not None:
        print(f"Using CSV file for model evaluation: {opt.csv_path}")
        if not os.path.exists(opt.csv_path):
            print(f"Error: CSV file not found: {opt.csv_path}")
            sys.exit(1)
    else:
        # Find models to evaluate
        if opt.models is None:
            try:
                model_dirs = [
                    d
                    for d in os.listdir(opt.checkpoints_dir)
                    if os.path.isdir(os.path.join(opt.checkpoints_dir, d))
                    and os.path.exists(
                        os.path.join(
                            opt.checkpoints_dir, d, f"{opt.which_epoch}_net_G_A.pth"
                        )
                    )
                ]
            except Exception as e:
                print(f"Error finding models in checkpoints directory: {e}")
                model_dirs = []
        else:
            model_dirs = opt.models

    try:
        opt.name = "temp_for_dataloader"

        # Setup data loaders with memory-efficient options
        print("Setting up data loaders with memory-efficient options...")

        # Import from dataset module
        from dataset.data_loader import setup_dataloaders

        opt.num_threads = 1  # Reduce number of dataloader workers to save memory
        _, validation_loader = setup_dataloaders(opt)
        print(f"Created validation loader with {len(validation_loader)} samples")

        # Load training exvivo data in a memory-efficient way (with caching)
        print("Loading training exvivo data for combined FID reference...")
        train_exvivo_tensors = load_train_exvivo_data(opt)
        if train_exvivo_tensors:
            print(
                f"Successfully loaded {len(train_exvivo_tensors)} training exvivo volumes"
            )
        else:
            print("Warning: No training exvivo data loaded")

        # Force garbage collection after loading data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error setting up data: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a results directory if it doesn't exist
    results_dir = "../results/fid_evaluation"
    os.makedirs(results_dir, exist_ok=True)

    output_file = (
        opt.output_file
        if opt.output_file
        else f"{results_dir}/fid_evaluation_results_{timestamp}.csv"
    )

    print(f"Results will be saved to: {os.path.abspath(output_file)}")

    # Process either CSV models or direct model list
    if opt.csv_path is not None:
        # Process models from CSV file
        results = process_csv_models(
            opt.csv_path, opt, validation_loader, device, train_exvivo_tensors
        )

        # Save results to specified output file
        save_results_to_csv(results, output_file)

    else:
        # Process individual models
        results = {}
        for model_name in model_dirs:
            model_result = evaluate_model(
                model_name,
                opt,
                validation_loader,
                device,
                train_exvivo_tensors,
                opt.which_epoch,
            )
            results[model_name] = model_result

            # Print results summary including all metrics
            print("\n\nEvaluation Metrics Summary:")
            print("=" * 80)

            # Report all metrics per model
            print("\nAll Metrics:")
            sorted_results = sorted(
                [(name, scores) for name, scores in results.items()],
                key=lambda x: x[1].get("fid_combined", float("inf")),
            )

            # Print header
            print(
                f"{'Model':<30} {'FID':<8} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'NCC':<8}"
            )
            print("-" * 80)

            for model_name, scores in sorted_results:
                print(
                    f"{model_name:<30} "
                    f"{scores.get('fid_combined', float('inf')):<8.3f} "
                    f"{scores.get('psnr', 0.0):<8.3f} "
                    f"{scores.get('ssim', 0.0):<8.3f} "
                    f"{scores.get('lpips', 1.0):<8.3f} "
                    f"{scores.get('ncc', 0.0):<8.3f}"
                )

        # Save results to text file
        text_output = f"{results_dir}/fid_evaluation_results_{timestamp}.txt"
        with open(text_output, "w") as f:
            f.write("FID Score Summary:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Evaluation performed using {opt.which_epoch} checkpoint\n")
            f.write(
                f"Combined FID: Weighted average of validation and training FID scores\n"
            )
            f.write("=" * 50 + "\n\n")

            f.write("Combined FID Scores:\n")
            for model_name, fid in sorted_results:
                f.write(f"{model_name}: {fid:.3f}\n")

        # Save detailed results to CSV
        save_results_to_csv(results, output_file)

    # Close WandB
    if opt.use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Error closing WandB: {e}")

    # Clear training data cache at the end
    import config

    config.CACHED_TRAIN_EXVIVO_DATA = None
    gc.collect()


if __name__ == "__main__":
    main()
