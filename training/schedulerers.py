from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def create_lr_schedulers(optimizer_G, optimizer_D, opt):
    """
    Create learning rate schedulers for generators and discriminators.

    Uses CosineAnnealingWarmRestarts for both Generator and Discriminator
    with configurable restart periods, multipliers, and minimum learning rates.
    Handles continuing from a checkpoint by advancing the schedulers.

    Args:
        optimizer_G: Generator optimizer
        optimizer_D: Discriminator optimizer
        opt: Options with scheduler parameters

    Returns:
        Dictionary containing scheduler objects for G and D
    """
    T_0 = getattr(opt, "lr_restart_epochs", 10)
    T_mult = getattr(opt, "lr_restart_mult", 2)
    eta_min_G = getattr(opt, "lr_min_G", 1e-6)
    eta_min_D = getattr(opt, "lr_min_D", 5e-7)

    scheduler_G = CosineAnnealingWarmRestarts(
        optimizer_G, T_0=T_0, T_mult=T_mult, eta_min=eta_min_G
    )

    scheduler_D = CosineAnnealingWarmRestarts(
        optimizer_D, T_0=T_0, T_mult=T_mult, eta_min=eta_min_D
    )

    if (
        hasattr(opt, "continue_train")
        and opt.continue_train
        and hasattr(opt, "epoch_count")
    ):
        starting_epoch = opt.epoch_count - 1
        if starting_epoch > 0:
            print(f"Advancing LR schedulers to epoch {starting_epoch}...")
            for _ in range(starting_epoch):
                scheduler_G.step()
                scheduler_D.step()
            print(
                f"LR after advancement - G: {optimizer_G.param_groups[0]['lr']:.7f}, D: {optimizer_D.param_groups[0]['lr']:.7f}"
            )

    return {"scheduler_G": scheduler_G, "scheduler_D": scheduler_D}


class LambdaWeightScheduler:
    """
    Scheduler for loss term weights (lambdas) throughout training.

    Dynamically adjusts various loss weights over the course of training:
    - Phases out identity loss
    - Gradually increases adversarial loss
    - Scales domain adaptation weight up
    - Adjusts cycle consistency weight

    This enables the model to focus on different aspects of learning at appropriate times.
    """

    def __init__(self, opt):
        """
        Initialize the lambda weight scheduler.

        Args:
            opt: Options containing initialization parameters and schedules
        """
        self.max_epochs = opt.niter + opt.niter_decay
        self.opt = opt

        self.initial_weights = {
            "lambda_identity_A": getattr(opt, "lambda_identity", 0.5),
            "lambda_identity_B": getattr(opt, "lambda_identity", 0.5),
            "lambda_ganloss_A": getattr(opt, "lambda_ganloss_A", 1.0),
            "lambda_ganloss_B": getattr(opt, "lambda_ganloss_B", 1.0),
            "lambda_cycle_A": getattr(opt, "lambda_cycle_A", 10.0),
            "lambda_cycle_B": getattr(opt, "lambda_cycle_B", 10.0),
            "lambda_feature_matching": getattr(opt, "lambda_feature_matching", 10.0),
            "lambda_domain_adaptation": getattr(opt, "lambda_domain_adaptation", 1.0),
        }

        self.current_weights = self.initial_weights.copy()

        self.identity_phase_out_start = getattr(opt, "identity_phase_out_start", 0.1)
        self.identity_phase_out_end = getattr(opt, "identity_phase_out_end", 0.4)
        self.gan_phase_in_start = getattr(opt, "gan_phase_in_start", 0.05)
        self.gan_phase_in_end = getattr(opt, "gan_phase_in_end", 0.3)
        self.domain_adaptation_phase_in_start = getattr(
            opt, "domain_adaptation_phase_in_start", 0.2
        )
        self.domain_adaptation_phase_in_end = getattr(
            opt, "domain_adaptation_phase_in_end", 0.6
        )
        self.domain_adaptation_scale_max = getattr(
            opt, "domain_adaptation_scale_max", 1.5
        )
        self.cycle_adjust_start = getattr(opt, "cycle_adjust_start", 0.3)
        self.cycle_adjust_end = getattr(opt, "cycle_adjust_end", 0.7)
        self.cycle_scale_min = getattr(opt, "cycle_scale_min", 0.7)
        self.min_identity_weight = getattr(opt, "min_identity_weight", 0.05)
        self.enabled = getattr(opt, "use_lambda_scheduler", False)

        if (
            hasattr(opt, "continue_train")
            and opt.continue_train
            and hasattr(opt, "epoch_count")
        ):
            starting_epoch = opt.epoch_count - 1
            if starting_epoch > 0:
                print(
                    f"Initializing lambda weights for continued training from epoch {starting_epoch}..."
                )
                self.update(starting_epoch)
                print("Lambda weights initialized for continued training:")
                for k, v in self.current_weights.items():
                    print(f"  {k}: {v:.4f}")

        for k, v in self.initial_weights.items():
            print(f"  {k}: {v}")
        print(f"  Total epochs: {self.max_epochs}")

    def update(self, current_epoch):
        """
        Update lambda weights based on current training epoch.

        Args:
            current_epoch: Current training epoch

        Returns:
            Dictionary of updated lambda weights
        """
        if not self.enabled:
            return self.initial_weights

        progress = min(1.0, current_epoch / self.max_epochs)

        if progress < self.identity_phase_out_start:
            lambda_identity_A = self.initial_weights["lambda_identity_A"]
            lambda_identity_B = self.initial_weights["lambda_identity_B"]
        elif progress > self.identity_phase_out_end:
            lambda_identity_A = self.min_identity_weight
            lambda_identity_B = self.min_identity_weight
        else:
            phase_out_progress = (progress - self.identity_phase_out_start) / (
                self.identity_phase_out_end - self.identity_phase_out_start
            )
            lambda_identity_A = max(
                self.initial_weights["lambda_identity_A"] * (1 - phase_out_progress),
                self.min_identity_weight,
            )
            lambda_identity_B = max(
                self.initial_weights["lambda_identity_B"] * (1 - phase_out_progress),
                self.min_identity_weight,
            )

        if progress < self.gan_phase_in_start:
            gan_scale = 0.5
        elif progress > self.gan_phase_in_end:
            gan_scale = 1.0
        else:
            phase_in_progress = (progress - self.gan_phase_in_start) / (
                self.gan_phase_in_end - self.gan_phase_in_start
            )
            gan_scale = 0.5 + 0.5 * phase_in_progress

        lambda_ganloss_A = self.initial_weights["lambda_ganloss_A"] * gan_scale
        lambda_ganloss_B = self.initial_weights["lambda_ganloss_B"] * gan_scale

        if progress < self.domain_adaptation_phase_in_start:
            domain_adaptation_scale = 1.0
        elif progress > self.domain_adaptation_phase_in_end:
            domain_adaptation_scale = self.domain_adaptation_scale_max
        else:
            phase_in_progress = (progress - self.domain_adaptation_phase_in_start) / (
                self.domain_adaptation_phase_in_end
                - self.domain_adaptation_phase_in_start
            )
            domain_adaptation_scale = (
                1.0 + (self.domain_adaptation_scale_max - 1.0) * phase_in_progress
            )

        lambda_domain_adaptation = (
            self.initial_weights["lambda_domain_adaptation"] * domain_adaptation_scale
        )

        if progress < self.cycle_adjust_start:
            cycle_scale = 1.0
        elif progress > self.cycle_adjust_end:
            cycle_scale = self.cycle_scale_min
        else:
            adjustment_progress = (progress - self.cycle_adjust_start) / (
                self.cycle_adjust_end - self.cycle_adjust_start
            )
            cycle_scale = 1.0 - (1.0 - self.cycle_scale_min) * adjustment_progress

        lambda_cycle_A = self.initial_weights["lambda_cycle_A"] * cycle_scale
        lambda_cycle_B = self.initial_weights["lambda_cycle_B"] * cycle_scale
        lambda_feature_matching = self.initial_weights["lambda_feature_matching"]

        self.current_weights = {
            "lambda_identity_A": lambda_identity_A,
            "lambda_identity_B": lambda_identity_B,
            "lambda_ganloss_A": lambda_ganloss_A,
            "lambda_ganloss_B": lambda_ganloss_B,
            "lambda_cycle_A": lambda_cycle_A,
            "lambda_cycle_B": lambda_cycle_B,
            "lambda_feature_matching": lambda_feature_matching,
            "lambda_domain_adaptation": lambda_domain_adaptation,
        }

        return self.current_weights

    def get_current_weights(self):
        """
        Get current lambda weights.

        Returns:
            Dictionary of current lambda weights
        """
        return self.current_weights

    def apply_to_options(self, opt):
        """
        Apply current lambda weights to the options object.

        Args:
            opt: Options object to modify

        Returns:
            Modified options object
        """
        for key, value in self.current_weights.items():
            if key == "lambda_identity_A" or key == "lambda_identity_B":
                opt.lambda_identity = value
            else:
                setattr(opt, key, value)
        return opt
