import torch
from tqdm import tqdm

# Import from our reorganized modules
from masking import create_mask_from_invivo, apply_mask_to_exvivo
from data_processing import extract_slices_from_volumes

# For metric calculation
from metrics.do_metrics import DomainMetricsCalculator
from metrics.val_metrics import MetricsCalculator


def calculate_slice_counts(validation_volumes, training_volumes):
    """
    Calculate the actual slice counts for validation and training data
    using the smallest dimension of each volume.
    """
    val_slices = 0

    for vol in validation_volumes:
        if vol.dim() == 5:  # 3D volume [B, C, D, H, W]
            # Find the smallest dimension
            B, C, D, H, W = vol.shape
            dims = [D, H, W]
            smallest_dim = min(dims)
            val_slices += smallest_dim

    train_slices = 0
    for vol in training_volumes:
        if vol.dim() == 5:  # 3D volume [B, C, D, H, W]
            # Find the smallest dimension
            B, C, D, H, W = vol.shape
            dims = [D, H, W]
            smallest_dim = min(dims)
            train_slices += smallest_dim

    return val_slices, train_slices


def calculate_additional_metrics_for_validation(model, validation_data, device):
    """
    Calculate PSNR, SSIM, LPIPS, and NCC metrics for the validation dataset.
    Uses masked versions of the generated outputs.
    """
    metrics_calc = MetricsCalculator(device=device)

    # Initialize accumulators for metrics
    total_metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "ncc": 0.0}
    num_samples = 0

    # Process each validation sample
    for i, data in enumerate(tqdm(validation_data, desc="Calculating metrics")):
        with torch.no_grad():
            # Set input data
            model.set_input(data)

            # Create mask from in vivo input
            invivo_mask = create_mask_from_invivo(model.real_A)

            # Run forward pass
            model.test()

            # Get the outputs
            real_A = model.real_A
            real_B = model.real_B
            fake_B = model.fake_B
            fake_A = model.fake_A if hasattr(model, "fake_A") else None

            # Skip if any outputs are None
            if real_A is None or real_B is None or fake_B is None:
                print(f"Warning: Skipping sample {i} due to None outputs")
                continue

            # Apply mask to fake outputs
            masked_fake_B = apply_mask_to_exvivo(fake_B, invivo_mask)
            masked_fake_A = (
                apply_mask_to_exvivo(fake_A, invivo_mask)
                if fake_A is not None
                else None
            )

            # Create image dictionary for metrics calculation
            images_dict = {
                "real_A": real_A,
                "real_B": real_B,
                "fake_A": (
                    masked_fake_A
                    if masked_fake_A is not None
                    else torch.zeros_like(real_A)
                ),
                "fake_B": masked_fake_B,
            }

            try:
                # Calculate metrics for this sample
                sample_metrics = metrics_calc.calculate_metrics(images_dict)

                # Accumulate metrics
                total_metrics["psnr"] += sample_metrics.get("psnr_sr", 0.0)
                total_metrics["ssim"] += sample_metrics.get("ssim_sr", 0.0)
                total_metrics["lpips"] += sample_metrics.get("lpips_sr", 1.0)
                total_metrics["ncc"] += sample_metrics.get("ncc_domain", 0.0)

                num_samples += 1

                print(
                    f"Sample {i}: PSNR={sample_metrics.get('psnr_sr', 0.0):.4f}, "
                    f"SSIM={sample_metrics.get('ssim_sr', 0.0):.4f}, "
                    f"LPIPS={sample_metrics.get('lpips_sr', 1.0):.4f}, "
                    f"NCC={sample_metrics.get('ncc_domain', 0.0):.4f}"
                )

            except Exception as e:
                print(f"Error calculating metrics for sample {i}: {e}")
                import traceback

                traceback.print_exc()

            # Clean up memory
            del images_dict, masked_fake_B
            if masked_fake_A is not None:
                del masked_fake_A
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Calculate averages
    if num_samples > 0:
        for key in total_metrics:
            total_metrics[key] /= num_samples

    print(f"Calculated metrics over {num_samples} validation samples:")
    for key, value in total_metrics.items():
        print(f"  - {key}: {value:.4f}")

    return total_metrics


def evaluate_slice_based_fid(
    model,
    validation_data,
    device,
    train_exvivo_data=None,
    model_name=None,
    base_dir=None,
):
    """
    Evaluate FID with memory-efficient weighted average for combined score
    and visualization of middle slices. Uses consistent slice extraction
    and masking of outputs.
    """
    print("\n** Starting Enhanced FID Evaluation with Masking **")
    metrics_calc = DomainMetricsCalculator(device=device)

    # Collect ALL validation volumes
    all_real_B = []
    all_fake_B = []
    all_masks = []

    # Process each validation sample
    print("\n== Processing all validation volumes ==")
    for i, data in enumerate(
        tqdm(validation_data, desc="Processing validation volumes")
    ):
        with torch.no_grad():
            model.set_input(data)

            # Create mask from in vivo input
            invivo_mask = create_mask_from_invivo(model.real_A)

            model.test()

            # Apply mask to output
            fake_B_masked = apply_mask_to_exvivo(model.fake_B, invivo_mask)
            model.fake_B = fake_B_masked

            # Get the current validation volume's real and fake data
            real_B = model.real_B  # validation exvivo
            fake_B = model.fake_B  # generated masked exvivo

            # Make sure we have valid data before adding
            if real_B is not None and fake_B is not None:
                all_real_B.append(real_B.detach().clone())
                all_fake_B.append(fake_B.detach().clone())
                all_masks.append(invivo_mask.detach().clone())
                print(f"Added validation volume {i + 1} with shape: {real_B.shape}")

                # Save visualizations if requested
                if model_name is not None and base_dir is not None:
                    from visualization import save_middle_slice_visualizations

                    save_middle_slice_visualizations(
                        model, data, model_name, i, base_dir
                    )
            else:
                print(f"Skipping validation volume {i + 1} due to None output")

    if not all_real_B or not all_fake_B:
        print("Error: No valid validation volumes were processed")
        return {
            "fid_val": float("inf"),
            "fid_train": float("inf"),
            "fid_combined": float("inf"),
        }

    # Combine all volumes
    print(f"\nCombining {len(all_real_B)} validation volumes")

    # Concatenate along batch dimension
    combined_real_B = torch.cat(all_real_B, dim=0)
    combined_fake_B = torch.cat(all_fake_B, dim=0)
    combined_masks = torch.cat(all_masks, dim=0)

    print(f"Combined validation real data shape: {combined_real_B.shape}")
    print(f"Combined validation fake data shape: {combined_fake_B.shape}")
    print(f"Combined validation masks shape: {combined_masks.shape}")

    # Extract 2D slices along the consistent smallest dimension for FID calculation
    print("\n== Extracting slices for FID calculation ==")
    validation_real_slices = extract_slices_from_volumes(all_real_B, device)
    validation_fake_slices = extract_slices_from_volumes(all_fake_B, device)

    print(f"Extracted validation real slices: {validation_real_slices.shape}")
    print(f"Extracted validation fake slices: {validation_fake_slices.shape}")

    # Calculate FID against validation exvivo data
    print("\n== Calculating FID against validation exvivo data ==")
    fid_val_score = metrics_calc.calculate_slice_based_fid(
        validation_real_slices, validation_fake_slices
    )
    result = {"fid_val": fid_val_score}
    print(f"FID against validation exvivo: {fid_val_score:.3f}")

    # Now process training exvivo data (if available)
    if train_exvivo_data is not None:
        print("\n== Processing training exvivo data for additional FID reference ==")

        # Calculate actual slice counts based on the smallest dimension of each volume
        val_slices, train_slices = calculate_slice_counts(all_real_B, train_exvivo_data)

        print(f"Calculated actual slice counts:")
        print(f"  - Validation slices: {val_slices}")
        print(f"  - Training slices: {train_slices}")

        # Process one random training volume for the training FID score
        if len(train_exvivo_data) > 0:
            try:
                # Use a representative volume
                idx = len(train_exvivo_data) // 2
                training_vol = train_exvivo_data[idx : idx + 1]

                # Extract slices from this volume
                training_real_slices = extract_slices_from_volumes(training_vol, device)
                print(
                    f"Extracted {training_real_slices.shape[0]} slices from representative training volume"
                )

                # Calculate FID against training data (using masked fake outputs)
                fid_train_score = metrics_calc.calculate_slice_based_fid(
                    training_real_slices, validation_fake_slices
                )

                result["fid_train"] = fid_train_score

                # Calculate combined FID using weighted average
                total_slices = val_slices + train_slices
                val_weight = val_slices / total_slices
                train_weight = train_slices / total_slices

                print(f"\n== Calculating weighted combined FID score ==")
                print(f"  - Validation slices: {val_slices} (weight: {val_weight:.3f})")
                print(
                    f"  - Training slices: {train_slices} (weight: {train_weight:.3f})"
                )

                combined_fid = (
                    val_weight * fid_val_score + train_weight * fid_train_score
                )

                result["fid_combined"] = combined_fid

                print(f"FID against validation exvivo: {fid_val_score:.3f}")
                print(f"FID against training exvivo: {fid_train_score:.3f}")
                print(f"Weighted combined FID: {combined_fid:.3f}")

                # Clean up
                del training_real_slices

            except Exception as e:
                print(f"Error calculating training FID: {e}")
                import traceback

                traceback.print_exc()

                result["fid_train"] = float("inf")
                result["fid_combined"] = float("inf")
        else:
            print("Warning: No training volumes available")
            result["fid_train"] = float("inf")
            result["fid_combined"] = float("inf")

    # Final cleanup
    del validation_real_slices
    del validation_fake_slices
    del combined_masks, combined_real_B, combined_fake_B, all_masks

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def log_fid_to_wandb(model_name, results, opt):
    """
    Log FID evaluation results and additional metrics to Weights & Biases.
    """
    import wandb
    import os
    from utils.utils import _sanitize_wandb_values

    # Skip if wandb not enabled
    if not hasattr(opt, "use_wandb") or not opt.use_wandb:
        return

    try:
        # Extract the run ID (3-character prefix) from model name
        run_id = model_name.split("_")[0]
        config_name = model_name.split("_")[1]
        full_name = model_name

        print(
            f"Logging evaluation results to WandB for run {run_id}, config {config_name}"
        )

        # Check if there's an existing wandb ID file for this model
        wandb_id_file = os.path.join(opt.checkpoints_dir, model_name, "wandb_id.txt")
        wandb_id = None

        if os.path.exists(wandb_id_file):
            with open(wandb_id_file, "r") as f:
                wandb_id = f.read().strip()
            print(f"Found existing wandb ID: {wandb_id}")

        # Determine if we need to create a new run or resume existing
        if wandb.run is None:
            print("Initializing new wandb connection")

            # Set up wandb config
            config = {
                "run_id": run_id,
                "config_name": config_name,
                "evaluation_type": "full_metrics",
                "model_name": model_name,
            }

            # Add any relevant options from opt
            for key, value in vars(opt).items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    config[key] = value

            # Initialize wandb
            os.environ["WANDB_START_METHOD"] = "thread"
            # Same wandb project as in training
            wandb_project = getattr(opt, "wandb_project", "prostate_SR-domain_cor")

            try:
                # Try to authenticate with the key used in the training code
                wandb.login(key="cde9483f01d3d4c883d033dbde93150f7d5b22d5", timeout=60)
            except Exception as e:
                print(f"Warning: WandB login failed: {e}, trying to proceed anyway")

            init_mode = "online" if opt.use_wandb else "disabled"

            # Initialize with the same run if ID exists
            if wandb_id:
                try:
                    wandb.init(
                        project=wandb_project,
                        group=getattr(opt, "group_name", "experiments"),
                        mode=init_mode,
                        name=full_name,
                        id=wandb_id,
                        resume="allow",
                        config=config,
                    )
                except Exception as e:
                    print(f"Failed to resume wandb run: {e}, creating new run")
                    wandb_id = None

            # Create new run if no ID or resuming failed
            if not wandb_id or wandb.run is None:
                wandb.init(
                    project=wandb_project,
                    group=getattr(opt, "group_name", "experiments"),
                    mode=init_mode,
                    name=full_name,
                    config=config,
                )

        # Prepare metrics for logging with 'eval/' prefix
        metrics = {
            "eval/fid_val": results.get("fid_val", float("inf")),
            "eval/fid_train": results.get("fid_train", float("inf")),
            "eval/fid_combined": results.get("fid_combined", float("inf")),
            "eval/configuration": config_name,
            "eval/run_id": run_id,
        }

        # Add additional metrics with eval/ prefix
        if "psnr" in results:
            metrics["eval/psnr"] = results.get("psnr", 0.0)
        if "ssim" in results:
            metrics["eval/ssim"] = results.get("ssim", 0.0)
        if "lpips" in results:
            metrics["eval/lpips"] = results.get("lpips", 1.0)
        if "ncc" in results:
            metrics["eval/ncc"] = results.get("ncc", 0.0)

        # Add any error messages
        if "error" in results and results["error"]:
            metrics["eval/error"] = results["error"]

        # Sanitize values to prevent NaN/Inf issues
        sanitized_metrics = _sanitize_wandb_values(metrics)

        # Log to wandb
        wandb.log(sanitized_metrics)
        print(f"Successfully logged evaluation metrics to wandb: {sanitized_metrics}")

    except Exception as e:
        print(f"Error logging to wandb: {e}")
        import traceback

        traceback.print_exc()
