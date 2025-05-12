import os
import torch
import itertools
import numpy as np

def optimize_model_for_memory(opt, using_cpu=False):
    """
    Apply memory optimizations to model configuration.
    Configures mixed precision, batch size, gradient accumulation,
    and other parameters based on available resources.
    
    Args:
        opt: Options object with model configuration
        using_cpu: Boolean flag indicating if running on CPU
        
    Returns:
        opt: Updated options object
    """
    # Enable mixed precision for all CUDA operations
    if torch.cuda.is_available() and not using_cpu:
        opt.mixed_precision = True
        print("Enabled mixed precision for all CUDA operations")

    # Use batch size based on available memory
    if torch.cuda.is_available() and not using_cpu:
        if torch.cuda.get_device_properties(0).total_memory < 8 * (1024**3):  # <8GB VRAM
            opt.batch_size = min(opt.batch_size, 1)
            print(f"Limited batch size to {opt.batch_size} due to limited GPU memory")
        else:
            opt.batch_size = min(opt.batch_size, 2)
            print(f"Limited batch size to {opt.batch_size} for medical imaging")

    # Configure gradient accumulation
    if not hasattr(opt, "accumulation_steps"):
        opt.accumulation_steps = 4
    else:
        opt.accumulation_steps = max(opt.accumulation_steps, 4)
    print(f"Using gradient accumulation with {opt.accumulation_steps} steps")

    # Configure discriminator update frequency
    if not hasattr(opt, "disc_update_freq"):
        opt.disc_update_freq = 2
    print(f"Setting discriminator update frequency to {opt.disc_update_freq}")

    # Use fewer discriminator layers for medical images
    if hasattr(opt, "n_layers_D") and opt.n_layers_D > 3:
        opt.n_layers_D = 3
        print(f"Limiting discriminator layers to {opt.n_layers_D} for efficiency")

    # Set appropriate display frequency for memory efficiency
    if not hasattr(opt, "display_freq"):
        opt.display_freq = 100 if not using_cpu else 10

    # Set conservative checkpoint frequency
    if not hasattr(opt, "save_epoch_freq"):
        opt.save_epoch_freq = 5 if not using_cpu else 1

    print(f"Memory-optimized configuration: batch_size={opt.batch_size}, "
          f"accumulation_steps={opt.accumulation_steps}, "
          f"mixed_precision={opt.mixed_precision}")
    
    return opt

def setup_memory_optimizations(using_cpu=False):
    """
    Apply memory optimization settings for CPU and GPU.
    Configures thread limits, memory fractions, and optimized
    backend operations based on the execution environment.
    
    Args:
        using_cpu: Boolean flag indicating if running on CPU
    """
    if using_cpu:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.backends.cudnn.enabled = False
        print("CPU Mode: Limited parallel threads for better memory usage")
    else:
        # Memory fraction management
        if hasattr(torch.cuda, "memory"):
            try:
                # Use a slightly lower fraction to avoid OOM
                torch.cuda.memory.set_per_process_memory_fraction(0.75)
                print("Set CUDA memory fraction to 0.75 to prevent OOM errors")
            except:
                pass

        # Modern PyTorch optimizations
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                try:
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    print("Enabled memory efficient scaled dot product attention")
                except:
                    pass
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    print("Enabled flash scaled dot product attention")
                except:
                    pass

        # Set optimal cudnn flags
        torch.backends.cudnn.benchmark = True  # Good for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Better performance
        print("Set cuDNN benchmark=True for optimized conv performance")

        # Disable gradient synchronization - only needed for multi-GPU
        if torch.cuda.device_count() <= 1:
            torch.cuda.set_device(0)
            print("Single GPU mode: disabled gradient synchronization")

def clip_generator_gradients(netG_A, netG_B):
    """
    Enhanced gradient clipping with better memory management.
    Detects and handles NaN/Inf values and applies appropriate
    clipping strategies.
    
    Args:
        netG_A: Generator A network
        netG_B: Generator B network
    """
    # First identify if we have any problematic gradients
    has_nan_inf = False
    for param in itertools.chain(netG_A.parameters(), netG_B.parameters()):
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_nan_inf = True
                break

    # Only proceed with clipping if needed
    if has_nan_inf:
        print("Detected NaN/Inf gradients in generator, applying specialized clipping")
        # Use a memory-efficient approach: process each parameter individually
        for param in itertools.chain(netG_A.parameters(), netG_B.parameters()):
            if param.grad is not None:
                # Replace NaN/Inf with zeros directly (memory efficient)
                mask_nan = torch.isnan(param.grad)
                mask_inf = torch.isinf(param.grad)

                if mask_nan.any() or mask_inf.any():
                    param.grad.data.masked_fill_(mask_nan | mask_inf, 0.0)

                # Clip large values without creating new tensors
                with torch.no_grad():
                    param.grad.data.clamp_(-10.0, 10.0)
    else:
        # If no NaN/Inf values, use standard clipping which is more efficient
        try:
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(netG_A.parameters(), netG_B.parameters()),
                max_norm=1.0,  # Use more conservative norm
            )
        except RuntimeError:
            # Fallback to per-parameter clipping
            for param in itertools.chain(netG_A.parameters(), netG_B.parameters()):
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)

def clip_discriminator_gradients(netD_A, netD_B):
    """
    Apply conservative gradient clipping to discriminator.
    Uses more stringent clipping norms for discriminator stability.
    
    Args:
        netD_A: Discriminator A network
        netD_B: Discriminator B network
    """
    try:
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(netD_A.parameters(), netD_B.parameters()),
            max_norm=0.5,  # Even more conservative for discriminator
        )
    except RuntimeError as e:
        print(f"Standard discriminator clipping failed: {e}, using per-parameter clipping")
        for param in itertools.chain(netD_A.parameters(), netD_B.parameters()):
            if param.grad is not None:
                # Handle NaN/Inf
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    param.grad.data.masked_fill_(torch.isnan(param.grad), 0.0)
                    param.grad.data.masked_fill_(torch.isinf(param.grad), 0.0)
                # Apply clipping
                param.grad.data.clamp_(-0.5, 0.5)

def debug_gradients(model):
    """
    Print detailed gradient information to pinpoint problematic layers.
    Analyzes all model gradients and provides recommendations for
    stabilizing training.
    
    Args:
        model: Model instance to debug
    """
    print("\n--- Detailed Gradient Analysis ---")
    problematic_layers = []
    total_params = 0

    # Check Generator A
    print("Generator A gradients:")
    for name, param in model.netG_A.named_parameters():
        if param.grad is not None:
            total_params += 1
            grad_norm = param.grad.data.norm(2).item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()

            if has_nan or has_inf or grad_norm > 100:
                problematic_layers.append((name, grad_norm, has_nan, has_inf))
                print(f"  PROBLEM - {name}: norm={grad_norm:.4f}, nan={has_nan}, inf={has_inf}")

    # Check Generator B
    print("\nGenerator B gradients:")
    for name, param in model.netG_B.named_parameters():
        if param.grad is not None:
            total_params += 1
            grad_norm = param.grad.data.norm(2).item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()

            if has_nan or has_inf or grad_norm > 100:
                problematic_layers.append((name, grad_norm, has_nan, has_inf))
                print(f"  PROBLEM - {name}: norm={grad_norm:.4f}, nan={has_nan}, inf={has_inf}")

    print(f"Found {len(problematic_layers)}/{total_params} problematic gradient tensors")

    # Recommendation based on analysis
    if problematic_layers:
        print("\nRecommendations:")

        # Check if feature matching layers are problematic
        if any("feat" in layer[0] for layer in problematic_layers):
            print("- Reduce feature matching loss weight (lambda_feat)")

        # Check if identity mapping layers are problematic
        if any("idt" in layer[0] for layer in problematic_layers):
            print("- Reduce identity loss weight (lambda_identity)")

        # Check if specific layers are problematic
        resnet_block_issues = sum(1 for layer in problematic_layers if "resnet" in layer[0])
        if resnet_block_issues > 2:
            print("- Consider reducing ResNet block depth or adding more skip connections")

    print("----------------------------\n")
