import os
import time

import torch

from dataset.data_loader import setup_dataloaders
from options.train_options import TrainOptions
from training.trainer import Trainer
from utils.utils import (
    set_seed,
    set_starting_epoch,
    init_wandb,
    setup_sliding_window_validation,
)

torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,max_split_size_mb:1024"
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def main():
    opt = TrainOptions().parse()

    print(opt.use_hinge)
    print(f"CUDA available: {torch.cuda.is_available()}")
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.use_precomputed_patches = True

    set_starting_epoch(opt)

    opt.use_wandb = True
    init_wandb(opt)
    set_seed(0)

    start_time = time.time()
    print("Setting up data loaders with precomputed patches...")

    train_loader, val_loader = setup_dataloaders(opt)

    data_setup_time = time.time() - start_time
    print(f"Data setup completed in {data_setup_time:.2f} seconds")

    trainer = Trainer(opt)

    try:
        if "mini" in opt.name:
            trainer = setup_sliding_window_validation(trainer, mini=True)
            print("Successfully enabled sliding window validation for mini-baseline")
        else:
            setup_sliding_window_validation(trainer)
            print("Successfully enabled sliding window validation")
    except Exception as e:
        print(f"Error setting up sliding window validation: {e}")

    print("\n===== Starting training =====")
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
