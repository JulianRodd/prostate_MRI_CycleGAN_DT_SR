import argparse
import gc
import os
import sys

import wandb

from agent import run_sweep_agent
from config import create_sweep_config


def init_wandb(opt=None):
    temp_wandb_dir = os.path.join(os.path.expanduser("~"), "temp_wandb_logs")
    os.makedirs(temp_wandb_dir, exist_ok=True)

    os.environ["WANDB_DIR"] = temp_wandb_dir
    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_DEBUG"] = "false"

    try:
        wandb.login(key="cde9483f01d3d4c883d033dbde93150f7d5b22d5", timeout=60)
        return True
    except Exception as e:
        print(f"Warning: WandB initialization failed: {str(e)}")
        print("Continuing without WandB logging...")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run W&B sweeps for prostate domain translation"
    )
    parser.add_argument("--init", action="store_true", help="Initialize a new sweep")
    parser.add_argument("--sweep_id", type=str, help="Continue an existing sweep")
    parser.add_argument(
        "--count", type=int, default=400, help="Number of runs for this agent"
    )
    parser.add_argument(
        "--config", type=str, default="sweep_baseline", help="Base configuration to use"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="prostate_SR-domain_cor",
        help="WandB project name",
    )
    parser.add_argument(
        "--entity", type=str, default=None, help="WandB entity (username or team name)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run additional cleanup after sweep completion",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Use train-kaggle instead of train-server command and set up Kaggle paths",
    )
    parser.add_argument(
        "--add_agent",
        action="store_true",
        help="Add an agent to an existing sweep using direct WandB agent call",
    )
    parser.add_argument(
        "--no_wandb_service_reset",
        action="store_true",
        help="Avoid resetting WandB services between runs (helps with Kaggle service errors)",
    )
    args = parser.parse_args()

    if not init_wandb():
        print("Failed to initialize WandB. Exiting.")
        return 1

    if args.kaggle:
        print("Setting up Kaggle environment...")
        os.makedirs("/kaggle/working/patches", exist_ok=True)
        os.makedirs("/kaggle/working/temp", exist_ok=True)
        os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
        os.makedirs("/kaggle/working/configurations", exist_ok=True)

        if args.no_wandb_service_reset:
            print("Using persistent WandB service mode for Kaggle")
            os.environ["WANDB_SERVICE_WAIT"] = "300"
            os.environ["WANDB_CONSOLE"] = "off"

        print("Kaggle environment setup complete")

    try:
        if args.add_agent and args.sweep_id:
            entity = args.entity
            if entity is None:
                try:
                    api = wandb.Api()
                    entity = api.default_entity
                    print(f"Using default entity: {entity}")
                except:
                    print(
                        "Warning: Could not determine default entity. Using current user..."
                    )
                    entity = None

            if entity:
                sweep_path = f"{entity}/{args.project}/{args.sweep_id}"
            else:
                sweep_path = f"{args.project}/{args.sweep_id}"

            print(f"Adding agent to sweep: {sweep_path}")
            print(f"This agent will run {args.count} experiments...")

            if args.kaggle:
                print("Using train-kaggle command for Kaggle environment")

            wandb.agent(
                sweep_path, lambda: run_sweep_agent(args.kaggle), count=args.count
            )
            return 0

        if args.init:
            sweep_config = create_sweep_config(args.config)
            sweep_id = wandb.sweep(
                sweep_config, project=args.project, entity=args.entity
            )
            print(f"Created sweep with ID: {sweep_id}")
            print(f"To run the sweep agent: python {sys.argv[0]} --sweep_id {sweep_id}")
            print(
                f"To add parallel agents: python {sys.argv[0]} --sweep_id {sweep_id} --add_agent"
            )

            if args.count > 0:
                print(f"Starting agent to run {args.count} experiments...")
                if args.kaggle:
                    print("Using train-kaggle command for Kaggle environment")
                wandb.agent(
                    sweep_id, lambda: run_sweep_agent(args.kaggle), count=args.count
                )

        elif args.sweep_id:
            print(
                f"Starting agent for sweep {args.sweep_id} to run {args.count} experiments..."
            )
            if args.kaggle:
                print("Using train-kaggle command for Kaggle environment")

            wandb.agent(
                args.sweep_id,
                lambda: run_sweep_agent(args.kaggle),
                count=args.count,
                project=args.project,
                entity=args.entity,
            )

        else:
            print(
                "Error: Must specify either --init to create a new sweep or --sweep_id to join an existing sweep"
            )
            return 1

    finally:
        if args.cleanup or args.init or args.sweep_id:
            try:
                temp_wandb_dir = os.path.join(
                    os.path.expanduser("~"), "temp_wandb_logs"
                )
                if os.path.exists(temp_wandb_dir):
                    for item in os.listdir(temp_wandb_dir):
                        item_path = os.path.join(temp_wandb_dir, item)
                        try:
                            if os.path.isdir(item_path):
                                if time.time() - os.path.getmtime(item_path) > 86400:
                                    shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                        except Exception as e:
                            print(f"Warning: Could not clean up {item_path}: {str(e)}")
                    print(f"Cleaned up temporary WandB directory: {temp_wandb_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean up WandB directory: {str(e)}")

            gc.collect()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("Final CUDA cache cleanup")
            except:
                pass

    return 0


if __name__ == "__main__":
    import time
    import shutil

    sys.exit(main())
