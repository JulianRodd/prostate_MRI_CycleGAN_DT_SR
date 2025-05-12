import os
import torch
import torch.nn as nn
from collections import OrderedDict
import utils.model_utils


class BaseModel:
    """
    BaseModel is an abstract base class for all models.
    It provides common functionality such as network initialization,
    loading/saving networks, and setting up training schedulers.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add model-specific commandline options.
        To be overridden by child classes.

        Args:
            parser: Option parser
            is_train: Whether the model is in training mode

        Returns:
            Modified parser
        """
        return parser

    def name(self):
        """Return the name of the model"""
        return "BaseModel"

    def initialize(self, opt):
        """
        Initialize the BaseModel.
        Sets up device, directories, and basic model attributes.

        Args:
            opt: Command line options
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = self._get_device()
        self._setup_directories(opt)
        self._init_model_attributes()

    def _get_device(self):
        """
        Get the device (GPU or CPU) to run the model on.

        Returns:
            torch.device: The device to use
        """
        return (
            torch.device(f"cuda:{self.gpu_ids[0]}")
            if self.gpu_ids
            else torch.device("cpu")
        )

    def _setup_directories(self, opt):
        """
        Create the directory for saving checkpoints.

        Args:
            opt: Command line options
        """
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    def _init_model_attributes(self):
        """Initialize model attributes with empty values"""
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def setup(self, opt, parser=None):
        """
        Setup the model.
        Initialize schedulers, load networks if needed, and print network info.

        Args:
            opt: Command line options
            parser: Option parser (optional)
        """
        if self.isTrain:
            self.schedulers = [
                utils.model_utils.get_scheduler(optimizer, opt)
                for optimizer in self.optimizers
            ]
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def eval(self):
        """Switch all networks to evaluation mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """Run forward pass in evaluation mode"""
        with torch.no_grad():
            self.forward()

    def get_image_paths(self):
        """Return image paths used in current batch"""
        return self.image_paths

    def get_current_visuals(self):
        """
        Return visualization images.

        Returns:
            OrderedDict of visualization tensors
        """
        return OrderedDict(
            (name, getattr(self, name))
            for name in self.visual_names
            if isinstance(name, str)
        )

    def get_current_losses(self):
        """
        Return training losses.

        Returns:
            OrderedDict of training loss values
        """
        return OrderedDict(
            (name, float(getattr(self, "loss_" + name)))
            for name in self.loss_names
            if isinstance(name, str)
        )

    def save_networks(self, which_epoch):
        """
        Save all networks to disk.

        Args:
            which_epoch: Current epoch number
        """
        for name in self.model_names:
            if not isinstance(name, str):
                continue

            save_filename = f"{which_epoch}_net_{name}.pth"
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, "net" + name)

            # Handle DataParallel properly
            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            # Save the state dictionary
            torch.save(state_dict, save_path)

            # For backward compatibility, save G_A as G when appropriate
            if name == "G_A" and not self.isTrain:
                g_save_filename = f"{which_epoch}_net_G.pth"
                g_save_path = os.path.join(self.save_dir, g_save_filename)
                torch.save(state_dict, g_save_path)

    def load_networks(self, which_epoch):
        """
        Load all networks from disk with error handling.

        Args:
            which_epoch: Epoch to load models from
        """
        for name in self.model_names:
            if not isinstance(name, str):
                continue

            # Get the correct filename based on model type and direction
            load_filename = self._get_load_filename(name, which_epoch)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, "net" + name)

            # Get the network outside of DataParallel wrapper if needed
            if isinstance(net, torch.nn.DataParallel):
                net = net.module

            # Check if the model file exists
            if not os.path.exists(load_path):
                print(f"Warning: Model file not found: {load_path}")
                print(f"Starting {name} from scratch")
                continue

            print(f"Loading model from {load_path}")
            try:
                # Load the state dict
                state_dict = torch.load(load_path, map_location=str(self.device))

                # Handle metadata if present
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # Clean up state dict for compatibility
                clean_state_dict = self._clean_state_dict(state_dict, net)

                # Load the state dict with relaxed strictness
                missing_keys, unexpected_keys = [], []
                if hasattr(net, "load_state_dict"):
                    incompatible = net.load_state_dict(clean_state_dict, strict=False)
                    missing_keys = incompatible.missing_keys
                    unexpected_keys = incompatible.unexpected_keys

                # Log any incompatibilities
                self._log_loading_status(name, missing_keys, unexpected_keys)

                print(f"Successfully loaded model {name}")

            except Exception as e:
                print(f"Error loading network {name}: {e}")
                import traceback

                traceback.print_exc()
                print(f"Starting {name} from scratch")

    def _clean_state_dict(self, state_dict, net):
        """
        Clean up state dict to improve compatibility.

        Args:
            state_dict: Loaded state dict
            net: Target network

        Returns:
            Cleaned state dict
        """
        cleaned_dict = {}

        # Get current network state dict for reference
        try:
            current_dict = net.state_dict()
        except Exception:
            current_dict = {}

        # Process each key in the state dict
        for key, value in state_dict.items():
            # Skip instance norm running stats that don't exist in target
            if (
                "running_mean" in key
                or "running_var" in key
                or "num_batches_tracked" in key
            ):
                # Find the corresponding module
                module = self._find_module_by_key(net, key)
                # Skip if module is InstanceNorm and doesn't use these stats
                if (
                    module
                    and isinstance(
                        module,
                        (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d),
                    )
                    and getattr(module, key.split(".")[-1], None) is None
                ):
                    continue

            # Handle shape mismatches
            if key in current_dict and value.shape != current_dict[key].shape:
                # Skip parameters with mismatched shapes
                continue

            cleaned_dict[key] = value

        return cleaned_dict

    def _find_module_by_key(self, net, key):
        """
        Find a module given a state dict key.

        Args:
            net: Network to search in
            key: State dict key

        Returns:
            The module corresponding to the key, or None if not found
        """
        module = net
        keyparts = key.split(".")

        # Walk down the module hierarchy
        for part in keyparts[:-1]:  # Skip the parameter name
            if not hasattr(module, part):
                return None
            module = getattr(module, part)

        return module

    def _log_loading_status(self, name, missing_keys, unexpected_keys):
        """
        Log information about missing and unexpected keys.

        Args:
            name: Name of the network
            missing_keys: List of missing keys
            unexpected_keys: List of unexpected keys
        """
        if missing_keys:
            max_keys_to_show = 10
            print(f"Missing keys in {name} ({len(missing_keys)} total):")
            for key in missing_keys[:max_keys_to_show]:
                print(f"  - {key}")
            if len(missing_keys) > max_keys_to_show:
                print(f"  ... and {len(missing_keys) - max_keys_to_show} more")

        if unexpected_keys:
            max_keys_to_show = 10
            print(f"Unexpected keys in {name} ({len(unexpected_keys)} total):")
            for key in unexpected_keys[:max_keys_to_show]:
                print(f"  - {key}")
            if len(unexpected_keys) > max_keys_to_show:
                print(f"  ... and {len(unexpected_keys) - max_keys_to_show} more")

    def _get_load_filename(self, name, which_epoch):
        """
        Get the filename to load based on model name and epoch.

        Args:
            name: Network name
            which_epoch: Epoch to load from

        Returns:
            Filename to load
        """
        if not self.isTrain and name == "G":
            return (
                f"{which_epoch}_net_G_B.pth"
                if self.opt.which_direction == "BtoA"
                else f"{which_epoch}_net_G_A.pth"
            )
        return f"{which_epoch}_net_{name}.pth"

    def print_networks(self, verbose):
        """
        Print network information.

        Args:
            verbose: Whether to print detailed network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = sum(p.numel() for p in net.parameters())
                if verbose:
                    print(net)
                print(f"[Network {name}] Total parameters : {num_params / 1e6:.3f} M")
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requires_grad for specified networks.

        Args:
            nets: Network or list of networks
            requires_grad: Whether to enable gradients
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        """Update learning rates for all optimizers and print the current learning rate"""
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]["lr"]
        print(f"learning rate = {lr:.7f}")
