import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import filters, registration
from scipy import ndimage
from tqdm import tqdm
import traceback


def perform_registration_analysis(scan_data, output_dir):
    """Register in-vivo to ex-vivo images and visualize deformation fields."""
    from statistics import compute_similarity_metrics

    if not scan_data:
        print("No scan data available for registration analysis")
        return

    os.makedirs(output_dir, exist_ok=True)

    metrics_file = os.path.join(output_dir, "registration_metrics.csv")
    with open(metrics_file, "w") as f:
        f.write(
            "sample_id,psnr_before,psnr_after,psnr_improvement,ssim_before,ssim_after,ssim_improvement,avg_displacement,max_displacement\n"
        )

    summary_stats = {
        "psnr_improvement": [],
        "ssim_improvement": [],
        "avg_displacement": [],
        "max_displacement": [],
    }

    try:
        print("Starting registration analysis...")
        for i, scan_info in enumerate(tqdm(scan_data, desc="Processing registrations")):
            if scan_info["invivo_data"] is None or scan_info["exvivo_data"] is None:
                continue

            sample_id = scan_info.get("sample_id", f"sample_{i}")
            output_path = os.path.join(output_dir, f"{sample_id}_registration.png")

            print(f"Processing registration for {sample_id}")

            if (
                scan_info["invivo_data"].shape[2] > 0
                and scan_info["exvivo_data"].shape[2] > 0
            ):
                invivo_slice = scan_info["invivo_data"][
                    :, :, scan_info["invivo_data"].shape[2] // 2
                ]
                exvivo_slice = scan_info["exvivo_data"][
                    :, :, scan_info["exvivo_data"].shape[2] // 2
                ]
            else:
                print(f"Skip {sample_id}: Invalid dimensions")
                continue

            invivo_norm = np.zeros_like(invivo_slice)
            exvivo_norm = np.zeros_like(exvivo_slice)

            invivo_mask = invivo_slice > 0
            exvivo_mask = exvivo_slice > 0

            if np.sum(invivo_mask) > 0 and np.sum(exvivo_mask) > 0:
                invivo_norm[invivo_mask] = (
                    invivo_slice[invivo_mask] - np.min(invivo_slice[invivo_mask])
                ) / (
                    np.max(invivo_slice[invivo_mask])
                    - np.min(invivo_slice[invivo_mask])
                )
                exvivo_norm[exvivo_mask] = (
                    exvivo_slice[exvivo_mask] - np.min(exvivo_slice[exvivo_mask])
                ) / (
                    np.max(exvivo_slice[exvivo_mask])
                    - np.min(exvivo_slice[exvivo_mask])
                )
            else:
                print(f"Skip {sample_id}: No foreground pixels")
                continue

            invivo_blur = filters.gaussian(invivo_norm, sigma=1.0)
            exvivo_blur = filters.gaussian(exvivo_norm, sigma=1.0)

            try:
                print(f"Calculating optical flow for {sample_id}")
                flow = registration.optical_flow_tvl1(invivo_blur, exvivo_blur)

                warped = np.zeros_like(invivo_blur)

                h, w = invivo_blur.shape
                y_coords, x_coords = np.mgrid[0:h, 0:w]

                y_new = y_coords + flow[0]
                x_new = x_coords + flow[1]

                y_new = np.clip(y_new, 0, h - 1)
                x_new = np.clip(x_new, 0, w - 1)

                print(f"Applying warping for {sample_id}")
                coordinates = np.stack([y_new.flatten(), x_new.flatten()])
                warped = ndimage.map_coordinates(
                    invivo_blur, coordinates, order=1
                ).reshape(invivo_blur.shape)

                diff_before = np.abs(invivo_blur - exvivo_blur)
                diff_after = np.abs(warped - exvivo_blur)

                print(f"Computing metrics for {sample_id}")
                metrics_before = compute_similarity_metrics(invivo_blur, exvivo_blur)
                metrics_after = compute_similarity_metrics(warped, exvivo_blur)

                displacement = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
                avg_displacement = np.mean(displacement)
                max_displacement = np.max(displacement)

                print(f"Creating visualization for {sample_id}")
                fig = plt.figure(figsize=(15, 12))
                gs = gridspec.GridSpec(2, 3, figure=fig)

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(invivo_blur, cmap="gray")
                ax1.set_title("In-vivo (Source)")
                ax1.axis("off")

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(exvivo_blur, cmap="gray")
                ax2.set_title("Ex-vivo (Target)")
                ax2.axis("off")

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(warped, cmap="gray")
                ax3.set_title("Registered In-vivo")
                ax3.axis("off")

                ax4 = fig.add_subplot(gs[1, 0])
                ax4.imshow(diff_before, cmap="viridis")
                ax4.set_title(
                    f'Before: PSNR={metrics_before["psnr"]:.2f}, SSIM={metrics_before["ssim"]:.2f}'
                )
                ax4.axis("off")

                ax5 = fig.add_subplot(gs[1, 1])
                ax5.imshow(diff_after, cmap="viridis")
                ax5.set_title(
                    f'After: PSNR={metrics_after["psnr"]:.2f}, SSIM={metrics_after["ssim"]:.2f}'
                )
                ax5.axis("off")

                ax6 = fig.add_subplot(gs[1, 2])
                step = 10
                y, x = np.mgrid[
                    : invivo_blur.shape[0] : step, : invivo_blur.shape[1] : step
                ]
                dx = flow[1][::step, ::step]
                dy = flow[0][::step, ::step]

                ax6.quiver(
                    x, y, dx, dy, angles="xy", scale_units="xy", scale=1, color="r"
                )
                ax6.imshow(invivo_blur, cmap="gray", alpha=0.3)
                ax6.set_title(f"Deformation Field (Avg: {avg_displacement:.2f}px)")
                ax6.axis("off")

                plt.figtext(
                    0.5,
                    0.01,
                    f"Registration Analysis: PSNR improved by {metrics_after['psnr'] - metrics_before['psnr']:.2f}dB, "
                    f"SSIM improved by {metrics_after['ssim'] - metrics_before['ssim']:.2f}, "
                    f"Max displacement: {max_displacement:.2f}px",
                    ha="center",
                    fontsize=12,
                    bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
                )

                plt.tight_layout()
                plt.savefig(output_path, dpi=150)
                plt.close()

                psnr_improvement = metrics_after["psnr"] - metrics_before["psnr"]
                ssim_improvement = metrics_after["ssim"] - metrics_before["ssim"]

                summary_stats["psnr_improvement"].append(psnr_improvement)
                summary_stats["ssim_improvement"].append(ssim_improvement)
                summary_stats["avg_displacement"].append(avg_displacement)
                summary_stats["max_displacement"].append(max_displacement)

                print(f"Writing metrics for {sample_id}")
                with open(metrics_file, "a") as f:
                    f.write(
                        f"{sample_id},{metrics_before['psnr']:.4f},{metrics_after['psnr']:.4f},"
                        f"{psnr_improvement:.4f},{metrics_before['ssim']:.4f},"
                        f"{metrics_after['ssim']:.4f},{ssim_improvement:.4f},"
                        f"{avg_displacement:.4f},{max_displacement:.4f}\n"
                    )

                print(f"Saved registration analysis for {sample_id}")
            except Exception as e:
                print(f"Error in registration for {sample_id}: {str(e)}")
                traceback.print_exc()
                continue

        if summary_stats["psnr_improvement"]:
            print("Creating registration summary statistics...")
            summary_df = pd.DataFrame(
                {
                    "metric": [
                        "PSNR Improvement (Mean)",
                        "PSNR Improvement (Median)",
                        "PSNR Improvement (Max)",
                        "SSIM Improvement (Mean)",
                        "SSIM Improvement (Median)",
                        "SSIM Improvement (Max)",
                        "Average Displacement (Mean)",
                        "Average Displacement (Median)",
                        "Maximum Displacement (Mean)",
                        "Maximum Displacement (Max)",
                    ],
                    "value": [
                        np.mean(summary_stats["psnr_improvement"]),
                        np.median(summary_stats["psnr_improvement"]),
                        np.max(summary_stats["psnr_improvement"]),
                        np.mean(summary_stats["ssim_improvement"]),
                        np.median(summary_stats["ssim_improvement"]),
                        np.max(summary_stats["ssim_improvement"]),
                        np.mean(summary_stats["avg_displacement"]),
                        np.median(summary_stats["avg_displacement"]),
                        np.mean(summary_stats["max_displacement"]),
                        np.max(summary_stats["max_displacement"]),
                    ],
                }
            )
            summary_df.to_csv(
                os.path.join(output_dir, "registration_summary.csv"), index=False
            )
            print(
                f"Saved registration summary to {os.path.join(output_dir, 'registration_summary.csv')}"
            )

            plt.figure(figsize=(12, 8))
            plt.boxplot(
                [summary_stats["psnr_improvement"], summary_stats["ssim_improvement"]],
                labels=["PSNR Improvement", "SSIM Improvement"],
            )
            plt.title("Registration Quality Metrics")
            plt.ylabel("Improvement")
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(output_dir, "registration_improvements.png"), dpi=150
            )
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.boxplot(
                [summary_stats["avg_displacement"], summary_stats["max_displacement"]],
                labels=["Average Displacement", "Maximum Displacement"],
            )
            plt.title("Registration Displacement Metrics")
            plt.ylabel("Displacement (pixels)")
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(output_dir, "registration_displacements.png"), dpi=150
            )
            plt.close()

    except Exception as e:
        print(f"Error in registration analysis: {e}")
        traceback.print_exc()
