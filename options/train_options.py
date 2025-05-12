from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--use_lsgan", action="store_true", help="use lsgan")
        parser.add_argument("--use_hinge", action="store_true", help="use hinge loss")
        parser.add_argument(
            "--use_wasserstein", action="store_true", help="use wasserstein loss"
        )
        parser.add_argument(
            "--use_relativistic", action="store_true", help="use relativistic loss"
        )
        parser.add_argument(
            "--use_stn",
            action="store_true",
            help="use spatial transformer network (STN) in the generator",
        )
        parser.add_argument(
            "--use_residual",
            action="store_true",
            help="use residual blocks in the generator",
        )
        parser.add_argument(
            "--use_full_attention",
            action="store_true",
            help="use full attention in the generator",
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="frequency of showing training results on console",
        )
        parser.add_argument(
            "--save_latest_freq",
            type=int,
            default=1000,
            help="frequency of saving the latest results",
        )
        # In options/base_options.py
        parser.add_argument(
            "--use_full_validation",
            action="store_true",
            help="Use full images for validation instead of patches",
        )
        parser.add_argument(
            "--full_image_size",
            nargs=3,
            type=int,
            default=None,
            help="Manual override for full image size (z,y,x) if auto-detection fails",
        )
        parser.add_argument(
            "--max_full_plots",
            type=int,
            default=8,
            help="Maximum number of full validation image plots to generate",
        )
        parser.add_argument(
            "--dataset_prefix",
            type=str,
            default=None,
            help='Dataset prefix for full validation image filenames (e.g., "icarus")',
        )
        parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=200,
            help="frequency of saving checkpoints at the end of epochs",
        )
        parser.add_argument(
            "--accumulation_steps",
            type=int,
            default=1,
            help="number of gradient accumulation steps",
        )
        parser.add_argument(
            "--continue_train",
            action="store_true",
            help="continue training: load the latest model",
        )

        parser.add_argument(
            "--continue_training_run",
            type=str,
            help="three digit random string of the run to continue training from",
        )

        parser.add_argument(
            "--epoch_count",
            type=int,
            default=1,
            help="the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...",
        )
        parser.add_argument(
            "--phase", type=str, default="train", help="train, val, test, etc"
        )
        parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="which epoch to load? set to latest to use latest cached model",
        )
        parser.add_argument(
            "--niter", type=int, default=200, help="# of iter at starting learning rate"
        )
        parser.add_argument(
            "--niter_decay",
            type=int,
            default=200,
            help="# of iter to linearly decay learning rate to zero",
        )
        parser.add_argument(
            "--beta1", type=float, default=0.5, help="momentum term of adam"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0001, help="initial learning rate for adam"
        )
        # LR scheduler parameters
        parser.add_argument(
            "--lr_restart_epochs",
            type=int,
            default=10,
            help="initial cycle length for cosine annealing",
        )
        parser.add_argument(
            "--lr_restart_mult",
            type=int,
            default=2,
            help="cycle length multiplier for warm restarts",
        )
        parser.add_argument(
            "--lr_min_G",
            type=float,
            default=1e-6,
            help="minimum learning rate for generator",
        )
        parser.add_argument(
            "--lr_min_D",
            type=float,
            default=5e-7,
            help="minimum learning rate for discriminator",
        )

        # Lambda scheduler control
        parser.add_argument(
            "--use_lambda_scheduler",
            action="store_true",
            help="use lambda weight scheduler",
        )
        parser.add_argument(
            "--lr_D_Constant",
            type=float,
            default=1,
            help="initial learning constant multiplier for discriminator",
        )
        parser.add_argument(
            "--pool_size",
            type=int,
            default=50,
            help="the size of image buffer that stores previously generated images",
        )
        parser.add_argument(
            "--no_html",
            action="store_true",
            help="do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/",
        )
        parser.add_argument(
            "--lr_policy",
            type=str,
            default="lambda",
            help="learning rate policy: lambda|step|plateau|cosine",
        )
        parser.add_argument(
            "--lr_decay_iters",
            type=int,
            default=50,
            help="multiply by a gamma every lr_decay_iters iterations",
        )
        parser.add_argument(
            "--patches_per_image",
            type=int,
            default=8,
            help="number of patches per image",
        )

        parser.add_argument(
            "--validation_chunk_size",
            type=int,
            default=4,
            help="Size of chunks for validation processing (smaller = less memory)",
        )
        parser.add_argument(
            "--use_spectral_norm_G",
            action="store_true",
            help="use spectral normalization in generator",
        )
        parser.add_argument(
            "--max_val_samples",
            type=int,
            default=5,
            help="Maximum number of validation samples to process (limit for memory)",
        )

        # Add lambda weight scheduler parameters
        parser.add_argument(
            "--identity_phase_out_start",
            type=float,
            default=0.1,
            help="start of identity loss phase-out (as fraction of training)",
        )
        parser.add_argument(
            "--identity_phase_out_end",
            type=float,
            default=0.4,
            help="end of identity loss phase-out (as fraction of training)",
        )
        parser.add_argument(
            "--gan_phase_in_start",
            type=float,
            default=0.05,
            help="start of GAN loss phase-in (as fraction of training)",
        )
        parser.add_argument(
            "--gan_phase_in_end",
            type=float,
            default=0.3,
            help="end of GAN loss phase-in (as fraction of training)",
        )
        parser.add_argument(
            "--domain_adaptation_phase_in_start",
            type=float,
            default=0.2,
            help="start of domain adaptation loss phase-in (as fraction of training)",
        )
        parser.add_argument(
            "--domain_adaptation_phase_in_end",
            type=float,
            default=0.6,
            help="end of domain adaptation loss phase-in (as fraction of training)",
        )
        parser.add_argument(
            "--domain_adaptation_scale_max",
            type=float,
            default=1.5,
            help="maximum scale factor for domain adaptation loss",
        )
        parser.add_argument(
            "--cycle_adjust_start",
            type=float,
            default=0.3,
            help="start of cycle consistency loss adjustment (as fraction of training)",
        )
        parser.add_argument(
            "--cycle_adjust_end",
            type=float,
            default=0.7,
            help="end of cycle consistency loss adjustment (as fraction of training)",
        )
        parser.add_argument(
            "--cycle_scale_min",
            type=float,
            default=0.7,
            help="minimum scale factor for cycle consistency loss",
        )
        parser.add_argument(
            "--min_identity_weight",
            type=float,
            default=0.05,
            help="minimum weight for identity loss after phase-out",
        )

        parser.add_argument(
            "--run_validation_interval",
            type=int,
            default=1,
            help="Number of epochs between validation runs",
        )

        self.isTrain = True
        return parser
