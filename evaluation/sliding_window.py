import torch
import gc
import math
from tqdm import tqdm

# Import from other modules
from masking import create_mask_from_invivo, apply_mask_to_exvivo


def predict_with_sliding_window(
    model, validation_data, device, patch_size=[64, 64, 32], min_patch_size=[16, 16, 8]
):
    """
    Process validation data using a sliding window approach for memory efficiency,
    with automatic stride calculation.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize results collectors
    all_real_B = []
    all_fake_B = []

    # Process each validation sample
    for i, data in enumerate(
        tqdm(validation_data, desc="Processing validation samples")
    ):
        with torch.no_grad():
            # Process current sample with sliding window
            real_B, fake_B = process_sample_with_sliding_window(
                model, data, device, patch_size, min_patch_size
            )

            # Store results if valid
            if real_B is not None and fake_B is not None:
                all_real_B.append(real_B.detach().cpu())
                all_fake_B.append(fake_B.detach().cpu())

            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Return the model's results (should be set by process_sample_with_sliding_window)
    return model.real_B, model.fake_B


def process_sample_with_sliding_window(
    model, data, device, patch_size=[64, 64, 32], min_patch_size=[16, 16, 8]
):
    """
    Process a single sample with sliding window to handle large 3D volumes efficiently.
    Uses automatic stride calculation based on patch size.
    Applies masking to the generated outputs.
    """
    with torch.no_grad():
        # Set the input data
        model.set_input(data)

        # Get input tensors
        real_A = model.real_A  # invivo
        real_B = model.real_B  # exvivo

        # Check if we have valid inputs
        if real_A is None:
            print("Warning: real_A is None, skipping sample")
            return None, None

        # Create mask from in vivo input
        invivo_mask = create_mask_from_invivo(real_A)

        # Process differently based on dimensions
        if real_A.dim() == 5:  # 3D volume [B, C, W, H, D]
            B, C, W, H, D = real_A.shape
            print(f"Original data shape: [B={B}, C={C}, W={W}, H={H}, D={D}]")

            # Calculate stride as half of patch dimensions
            stride_inplane = max(patch_size[1] // 2, 1)  # Half of H dimension
            stride_layer = max(patch_size[0] // 2, 1)  # Half of D dimension

            print(
                f"Using patch size: {patch_size}, stride_inplane: {stride_inplane}, stride_layer: {stride_layer}"
            )

            # Generate patch indices for sliding window
            patch_indices = generate_patch_indices(
                real_A.shape[2:],
                patch_size,
                stride_inplane,
                stride_layer,
                min_patch_size,
            )

            print(f"Generated {len(patch_indices)} patches for processing")

            # Initialize output tensors with same shape as input
            fake_B = torch.zeros_like(real_A, device="cpu")
            weight_map = torch.zeros_like(real_A, device="cpu")

            # Process patches one by one to save memory
            for idx, patch_idx in enumerate(
                tqdm(patch_indices, desc="Processing patches")
            ):
                istart, iend, jstart, jend, kstart, kend = patch_idx

                # Extract patch and corresponding mask patch
                patch_A = real_A[:, :, istart:iend, jstart:jend, kstart:kend].to(device)
                patch_mask = invivo_mask[
                    :, :, istart:iend, jstart:jend, kstart:kend
                ].to(device)

                try:
                    # Forward pass for this patch
                    patch_fake_B = model.netG_A(patch_A)

                    # Apply mask to patch result
                    patch_fake_B = apply_mask_to_exvivo(patch_fake_B, patch_mask)

                    # Add result to output tensor (on CPU to save GPU memory)
                    fake_B[
                        :, :, istart:iend, jstart:jend, kstart:kend
                    ] += patch_fake_B.cpu()
                    weight_map[:, :, istart:iend, jstart:jend, kstart:kend] += 1.0

                except Exception as e:
                    print(f"Error processing patch {idx}: {e}")

                # Clean up to free memory
                del patch_A, patch_mask
                if "patch_fake_B" in locals():
                    del patch_fake_B

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Average overlapping regions
            epsilon = 1e-8
            weight_map = torch.where(
                weight_map > 0,
                weight_map,
                torch.tensor(epsilon, device=weight_map.device),
            )
            fake_B = fake_B / weight_map

            # Apply final mask to entire volume
            masked_fake_B = apply_mask_to_exvivo(fake_B, invivo_mask.cpu())

            # Move final results to device and store in model
            model.fake_B = masked_fake_B.to(device)

            # Clean up processing tensors
            del weight_map, fake_B

        else:  # Already 2D [B, C, H, W]
            # Standard forward pass for 2D data
            model.test()

            # Apply mask to the result
            model.fake_B = apply_mask_to_exvivo(model.fake_B, invivo_mask.to(device))

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model.real_B, model.fake_B


def generate_patch_indices(
    image_shape, patch_size, stride_inplane, stride_layer, min_patch_size
):
    """
    Generate patch indices for sliding window processing.
    """
    inum = max(
        1,
        int(math.ceil((image_shape[0] - patch_size[0]) / float(stride_inplane))) + 1,
    )
    jnum = max(
        1,
        int(math.ceil((image_shape[1] - patch_size[1]) / float(stride_inplane))) + 1,
    )
    knum = max(
        1,
        int(math.ceil((image_shape[2] - patch_size[2]) / float(stride_layer))) + 1,
    )

    patch_indices = []

    for k in range(knum):
        for i in range(inum):
            for j in range(jnum):
                istart = min(i * stride_inplane, max(0, image_shape[0] - patch_size[0]))
                iend = min(istart + patch_size[0], image_shape[0])

                if iend - istart < min_patch_size[0]:
                    continue

                jstart = min(j * stride_inplane, max(0, image_shape[1] - patch_size[1]))
                jend = min(jstart + patch_size[1], image_shape[1])

                if jend - jstart < min_patch_size[1]:
                    continue

                kstart = min(k * stride_layer, max(0, image_shape[2] - patch_size[2]))
                kend = min(kstart + patch_size[2], image_shape[2])

                if kend - kstart < min_patch_size[2]:
                    continue

                if (
                    (iend - istart) >= min_patch_size[0]
                    and (jend - jstart) >= min_patch_size[1]
                    and (kend - kstart) >= min_patch_size[2]
                ):
                    patch_indices.append([istart, iend, jstart, jend, kstart, kend])

    if not patch_indices:
        print(f"Warning: No valid patches found for image of shape {image_shape}.")
        print(f"Creating a single central patch with minimum sizes {min_patch_size}")

        istart = max(0, image_shape[0] // 2 - min_patch_size[0] // 2)
        iend = min(image_shape[0], istart + min_patch_size[0])

        jstart = max(0, image_shape[1] // 2 - min_patch_size[1] // 2)
        jend = min(image_shape[1], jstart + min_patch_size[1])

        kstart = max(0, image_shape[2] // 2 - min_patch_size[2] // 2)
        kend = min(image_shape[2], kstart + min_patch_size[2])

        if (iend - istart) >= 4 and (jend - jstart) >= 4 and (kend - kstart) >= 4:
            patch_indices.append([istart, iend, jstart, jend, kstart, kend])
        else:
            print(f"Cannot create valid patches for this image - dimensions too small")

    return patch_indices
