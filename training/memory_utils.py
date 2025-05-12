import gc
import os
import torch
import psutil

def release_graph_memory(model):
    """
    Aggressively releases computation graph memory by clearing model tensors
    and forcing garbage collection.
    
    Args:
        model: Model instance containing tensors to be cleared
    """
    # Clear intermediate tensors
    for tensor_name in [
        "fake_A", "fake_B", "rec_A", "rec_B", "idt_A", "idt_B",
        "feat_real_A", "feat_real_B", "feat_fake_A", "feat_fake_B",
    ]:
        if hasattr(model, tensor_name):
            setattr(model, tensor_name, None)

    # Clear loss values
    if hasattr(model, "loss_G"):
        model.loss_G = 0
    if hasattr(model, "loss_D_A"):
        model.loss_D_A = 0
    if hasattr(model, "loss_D_B"):
        model.loss_D_B = 0

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def reset_memory(model, scaler_G=None, scaler_D=None, force_cuda_empty=True):
    """
    Aggressively clean up memory between runs to prevent memory leaks.
    Clears model tensors, cached data, resets pools, zeroes gradients, and
    forces memory cleanup.
    
    Args:
        model: The model instance to clean
        scaler_G: Generator gradient scaler for mixed precision
        scaler_D: Discriminator gradient scaler for mixed precision
        force_cuda_empty: Whether to explicitly empty CUDA cache
        
    Returns:
        bool: Success status
    """
    # Clear all model-specific tensors that might be holding references
    for tensor_name in [
        "fake_A", "fake_B", "rec_A", "rec_B", "real_A", "real_B", 
        "idt_A", "idt_B", "feat_real_A", "feat_real_B", "feat_fake_A", "feat_fake_B",
    ]:
        if hasattr(model, tensor_name):
            setattr(model, tensor_name, None)

    # Clear any cached tensors in the model
    if hasattr(model, "cached_tensors"):
        model.cached_tensors = {}

    # Clear fake pools if they exist
    if hasattr(model, "fake_A_pool"):
        model.fake_A_pool.reset()
    if hasattr(model, "fake_B_pool"):
        model.fake_B_pool.reset()

    # Reset any gradient scalers
    if scaler_G is not None:
        del scaler_G
        scaler_G = torch.amp.GradScaler(enabled=True)
    if scaler_D is not None:
        del scaler_D
        scaler_D = torch.amp.GradScaler(enabled=True)

    # Zero gradients with set_to_none=True to free memory
    if hasattr(model, "optimizer_G"):
        model.optimizer_G.zero_grad(set_to_none=True)
    if hasattr(model, "optimizer_D"):
        model.optimizer_D.zero_grad(set_to_none=True)

    # Clear CUDA cache
    if force_cuda_empty and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "memory_stats"):
            current_allocated = torch.cuda.memory_allocated() / (1024**2)
            current_reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"CUDA memory before cleanup: {current_allocated:.2f}MB allocated, {current_reserved:.2f}MB reserved")

        # Force CUDA synchronization
        torch.cuda.synchronize()

        if hasattr(torch.cuda, "memory_stats"):
            current_allocated = torch.cuda.memory_allocated() / (1024**2)
            current_reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"CUDA memory after cleanup: {current_allocated:.2f}MB allocated, {current_reserved:.2f}MB reserved")

    # Clear CPU cache and run garbage collection multiple times
    gc.collect()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    gc.collect()

    # Try to release Python memory back to the OS
    if hasattr(gc, "collect"):
        gc.collect()

    # Report memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Process memory usage: {memory_info.rss / (1024 ** 2):.2f}MB")

    # Clear matplotlib figures if any
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except:
        pass

    # For PyTorch 1.11+ with memory diagnostics
    if hasattr(torch.cuda, "memory_summary"):
        try:
            print(torch.cuda.memory_summary(abbreviated=True))
        except:
            pass

    return True

def aggressive_memory_cleanup(model=None):
    """
    Enhanced version of memory cleanup with better CUDA handling.
    Synchronizes CUDA, empties cache, clears model tensors, and 
    performs multiple rounds of garbage collection.
    
    Args:
        model: Optional model instance to clean
    """
    if torch.cuda.is_available():
        # First synchronize CUDA
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Clear all model tensor references if model is provided
    if model is not None:
        for tensor_name in ["fake_A", "fake_B", "rec_A", "rec_B", "idt_A", "idt_B"]:
            if hasattr(model, tensor_name):
                setattr(model, tensor_name, None)

        # Zero gradients for all models
        for net_name in ["netG_A", "netG_B", "netD_A", "netD_B"]:
            if hasattr(model, net_name):
                net = getattr(model, net_name)
                if hasattr(net, "zero_grad"):
                    net.zero_grad(set_to_none=True)

    # Multiple rounds of garbage collection
    using_cpu = not torch.cuda.is_available()
    if using_cpu:
        for _ in range(3):
            gc.collect()
    else:
        gc.collect()
        torch.cuda.synchronize()
        gc.collect()

def analyze_memory(model=None):
    """
    Analyze memory usage and identify bottlenecks.
    Reports GPU memory usage, model tensor sizes, and provides
    optimization recommendations.
    
    Args:
        model: Optional model instance to analyze
    """
    if torch.cuda.is_available():
        print(f"\nCUDA Memory Analysis:")
        print(f"  Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        # Tensor analysis (approximate)
        if model is not None:
            model_tensors = {
                "Generator A": sum(p.numel() * p.element_size() for p in model.netG_A.parameters()),
                "Generator B": sum(p.numel() * p.element_size() for p in model.netG_B.parameters()),
                "Discriminator A": sum(p.numel() * p.element_size() for p in model.netD_A.parameters()),
                "Discriminator B": sum(p.numel() * p.element_size() for p in model.netD_B.parameters()),
            }

            print("\nModel Memory Requirements:")
            for name, size in model_tensors.items():
                print(f"  {name}: {size / 1e6:.2f} MB")

            # Estimate batch memory
            if hasattr(model, "real_A") and model.real_A is not None:
                batch_memory = model.real_A.numel() * model.real_A.element_size() * 8  # Heuristic multiplier
                print(f"\nEstimated memory per batch: {batch_memory / 1e6:.2f} MB")

            # Recommendations
            print("\nMemory Optimization Recommendations:")
            if torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.8:
                print("  - Reduce batch size by half")
                print("  - Increase gradient accumulation steps")
            else:
                print("  - Current memory usage is acceptable")
