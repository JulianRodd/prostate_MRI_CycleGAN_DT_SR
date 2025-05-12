import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import traceback
from scipy.stats import pearsonr


def visualize_paired_scans(
    invivo_data, exvivo_data, invivo_img, exvivo_img, sample_id, output_path
):
    """Create visualizations comparing in-vivo and ex-vivo scans."""
    if (
        invivo_data is None
        or exvivo_data is None
        or invivo_img is None
        or exvivo_img is None
    ):
        print(f"Skipping visualization for {sample_id} due to missing data")
        return

    try:
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)

        if invivo_data.shape[2] > 0 and exvivo_data.shape[2] > 0:
            invivo_slice_z = invivo_data[:, :, invivo_data.shape[2] // 2]
            exvivo_slice_z = exvivo_data[:, :, exvivo_data.shape[2] // 2]
        else:
            print(f"Warning: Invalid Z dimensions for {sample_id}")
            return

        if invivo_data.shape[1] > 0 and exvivo_data.shape[1] > 0:
            invivo_slice_y = invivo_data[:, invivo_data.shape[1] // 2, :]
            exvivo_slice_y = exvivo_data[:, exvivo_data.shape[1] // 2, :]
        else:
            print(f"Warning: Invalid Y dimensions for {sample_id}")
            return

        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.imshow(invivo_slice_z.T, cmap="gray", origin="lower")
        ax1.set_title("In-vivo Axial Slice")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.imshow(exvivo_slice_z.T, cmap="gray", origin="lower")
        ax2.set_title("Ex-vivo Axial Slice")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.imshow(invivo_slice_y.T, cmap="gray", origin="lower")
        ax3.set_title("In-vivo Coronal Slice")
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[1, 2:4])
        ax4.imshow(exvivo_slice_y.T, cmap="gray", origin="lower")
        ax4.set_title("Ex-vivo Coronal Slice")
        ax4.axis("off")

        ax5 = fig.add_subplot(gs[2, 0:2])
        invivo_nonzero = invivo_data[invivo_data > 0]
        exvivo_nonzero = exvivo_data[exvivo_data > 0]

        if len(invivo_nonzero) > 0:
            sns.histplot(
                invivo_nonzero.flatten(),
                bins=50,
                kde=True,
                color="blue",
                stat="density",
                label="In-vivo",
                ax=ax5,
            )
        if len(exvivo_nonzero) > 0:
            sns.histplot(
                exvivo_nonzero.flatten(),
                bins=50,
                kde=True,
                color="red",
                stat="density",
                label="Ex-vivo",
                ax=ax5,
            )

        ax5.set_title("Intensity Histograms")
        ax5.set_xlabel("Intensity")
        ax5.set_ylabel("Density")
        ax5.legend()

        ax6 = fig.add_subplot(gs[2, 2:4])

        if (
            len(invivo_nonzero) > 0
            and np.max(invivo_nonzero) - np.min(invivo_nonzero) > 0
        ):
            invivo_norm = (invivo_nonzero - np.min(invivo_nonzero)) / (
                np.max(invivo_nonzero) - np.min(invivo_nonzero)
            )
            sns.histplot(
                invivo_norm,
                bins=50,
                kde=True,
                color="blue",
                stat="density",
                label="In-vivo (normalized)",
                ax=ax6,
            )

        if (
            len(exvivo_nonzero) > 0
            and np.max(exvivo_nonzero) - np.min(exvivo_nonzero) > 0
        ):
            exvivo_norm = (exvivo_nonzero - np.min(exvivo_nonzero)) / (
                np.max(exvivo_nonzero) - np.min(exvivo_nonzero)
            )
            sns.histplot(
                exvivo_norm,
                bins=50,
                kde=True,
                color="red",
                stat="density",
                label="Ex-vivo (normalized)",
                ax=ax6,
            )

        ax6.set_title("Normalized Intensity Histograms")
        ax6.set_xlabel("Normalized Intensity (0-1)")
        ax6.set_ylabel("Density")
        ax6.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error visualizing paired scans for {sample_id}: {e}")
        traceback.print_exc()


def create_average_volumes(scan_data):
    """Create average volumes for in-vivo and ex-vivo scans."""
    if not scan_data:
        print("No scan data available for creating average volumes")
        return None, None

    invivo_count = exvivo_count = 0
    invivo_sum = None
    exvivo_sum = None
    invivo_shape = exvivo_shape = None

    for scan_info in scan_data:
        if scan_info["invivo_data"] is None or scan_info["exvivo_data"] is None:
            continue

        curr_invivo_shape = scan_info["invivo_data"].shape
        curr_exvivo_shape = scan_info["exvivo_data"].shape

        if invivo_sum is None:
            invivo_shape = curr_invivo_shape
            invivo_sum = np.zeros(invivo_shape)
            invivo_count = 0

        if exvivo_sum is None:
            exvivo_shape = curr_exvivo_shape
            exvivo_sum = np.zeros(exvivo_shape)
            exvivo_count = 0

        if curr_invivo_shape != invivo_shape or curr_exvivo_shape != exvivo_shape:
            print(
                f"Warning: Skipping scan with mismatched dimensions. "
                f"Expected {invivo_shape}/{exvivo_shape}, got {curr_invivo_shape}/{curr_exvivo_shape}"
            )
            continue

        invivo_sum += scan_info["invivo_data"]
        invivo_count += 1

        exvivo_sum += scan_info["exvivo_data"]
        exvivo_count += 1

    invivo_avg = invivo_sum / invivo_count if invivo_count > 0 else None
    exvivo_avg = exvivo_sum / exvivo_count if exvivo_count > 0 else None

    print(
        f"Created average volumes from {invivo_count} in-vivo and {exvivo_count} ex-vivo scans"
    )

    return invivo_avg, exvivo_avg


def visualize_average_volumes(invivo_avg, exvivo_avg, output_path):
    """Create visualizations of average in-vivo and ex-vivo volumes."""
    if invivo_avg is None or exvivo_avg is None:
        print("Cannot visualize average volumes: missing data")
        return

    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        invivo_slice_z = invivo_avg[:, :, invivo_avg.shape[2] // 2]
        exvivo_slice_z = exvivo_avg[:, :, exvivo_avg.shape[2] // 2]

        invivo_slice_y = invivo_avg[:, invivo_avg.shape[1] // 2, :]
        exvivo_slice_y = exvivo_avg[:, exvivo_avg.shape[1] // 2, :]

        invivo_slice_x = invivo_avg[invivo_avg.shape[0] // 2, :, :]
        exvivo_slice_x = exvivo_avg[exvivo_avg.shape[0] // 2, :, :]

        axes[0, 0].imshow(invivo_slice_z.T, cmap="gray", origin="lower")
        axes[0, 0].set_title("In-vivo Axial (Z)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(invivo_slice_y.T, cmap="gray", origin="lower")
        axes[0, 1].set_title("In-vivo Coronal (Y)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(invivo_slice_x.T, cmap="gray", origin="lower")
        axes[0, 2].set_title("In-vivo Sagittal (X)")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(exvivo_slice_z.T, cmap="gray", origin="lower")
        axes[1, 0].set_title("Ex-vivo Axial (Z)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(exvivo_slice_y.T, cmap="gray", origin="lower")
        axes[1, 1].set_title("Ex-vivo Coronal (Y)")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(exvivo_slice_x.T, cmap="gray", origin="lower")
        axes[1, 2].set_title("Ex-vivo Sagittal (X)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved average volume visualization to {output_path}")
    except Exception as e:
        print(f"Error visualizing average volumes: {e}")
        traceback.print_exc()


def create_intensity_correlation_heatmap(scan_data, output_path):
    """Create a heatmap showing the correlation between in-vivo and ex-vivo intensities."""
    import os
    import pandas as pd
    from data_loading import preprocess_scan

    if not scan_data:
        print("No scan data available for intensity correlation heatmap")
        return

    try:
        invivo_intensities = []
        exvivo_intensities = []

        for scan_info in scan_data:
            if scan_info["invivo_data"] is None or scan_info["exvivo_data"] is None:
                continue

            invivo_mask = scan_info["invivo_data"] > 0
            exvivo_mask = scan_info["exvivo_data"] > 0

            if np.sum(invivo_mask) == 0 or np.sum(exvivo_mask) == 0:
                continue

            invivo_norm = preprocess_scan(scan_info["invivo_data"], normalize=True)
            exvivo_norm = preprocess_scan(scan_info["exvivo_data"], normalize=True)

            invivo_samples = invivo_norm[invivo_mask].flatten()
            exvivo_samples = exvivo_norm[exvivo_mask].flatten()

            max_samples = 10000
            if len(invivo_samples) > max_samples:
                indices = np.random.choice(
                    len(invivo_samples), max_samples, replace=False
                )
                invivo_samples = invivo_samples[indices]

            if len(exvivo_samples) > max_samples:
                indices = np.random.choice(
                    len(exvivo_samples), max_samples, replace=False
                )
                exvivo_samples = exvivo_samples[indices]

            invivo_intensities.extend(invivo_samples)
            exvivo_intensities.extend(exvivo_samples)

        if invivo_intensities and exvivo_intensities:
            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

            ax1 = plt.subplot(gs[0])

            hist, x_edges, y_edges = np.histogram2d(
                invivo_intensities, exvivo_intensities, bins=50, range=[[0, 1], [0, 1]]
            )

            hist_log = np.log1p(hist)

            im = ax1.imshow(
                hist_log.T,
                origin="lower",
                extent=[0, 1, 0, 1],
                aspect="auto",
                cmap="viridis",
            )

            fig.colorbar(im, ax=ax1, label="Log(count + 1)")

            ax1.set_xlabel("In-vivo Normalized Intensity")
            ax1.set_ylabel("Ex-vivo Normalized Intensity")
            ax1.set_title("Intensity Correlation Heatmap")

            ax1.plot([0, 1], [0, 1], "r--", alpha=0.7)

            correlation = pearsonr(invivo_intensities, exvivo_intensities)[0]
            ax1.text(
                0.1,
                0.9,
                f"Correlation: {correlation:.3f}",
                bbox=dict(facecolor="white", alpha=0.7),
            )

            invivo_array = np.array(invivo_intensities)
            exvivo_array = np.array(exvivo_intensities)
            mask = invivo_array > 0.01
            ratios = exvivo_array[mask] / invivo_array[mask]
            mean_ratio = np.mean(ratios)
            median_ratio = np.median(ratios)

            ax2 = plt.subplot(gs[1])
            ax2.axis("off")

            explanation_text = (
                "INTENSITY CORRELATION ANALYSIS\n\n"
                "This heatmap shows how intensity values map between in-vivo and ex-vivo scans.\n\n"
                f"• Mean Intensity Ratio: {mean_ratio:.2f}\n"
                f"• Median Intensity Ratio: {median_ratio:.2f}\n"
                f"• Pearson Correlation: {correlation:.3f}\n\n"
                "INTERPRETATION:\n\n"
                "- Perfect linear mapping would follow the red diagonal line\n"
                "- Vertical bands indicate in-vivo values that map to multiple ex-vivo intensities\n"
                "- Horizontal bands indicate ex-vivo values that map to multiple in-vivo intensities\n"
                "- The intensity ratio represents how much brighter/darker ex-vivo is compared to in-vivo\n"
                "- The correlation measures how predictable the mapping is\n\n"
                "ML MODEL IMPLICATIONS:\n\n"
                "- Consider intensity transformation as part of your model\n"
                "- A ratio > 1 means ex-vivo is generally brighter than in-vivo\n"
                "- Non-linear mapping suggests using non-linear activation functions"
            )

            ax2.text(
                0,
                1,
                explanation_text,
                va="top",
                ha="left",
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=1"),
            )

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            print(f"Saved intensity correlation heatmap to {output_path}")

            ratio_df = pd.DataFrame(
                {
                    "metric": [
                        "Mean Intensity Ratio (Ex/In)",
                        "Median Intensity Ratio (Ex/In)",
                        "Intensity Correlation",
                    ],
                    "value": [mean_ratio, median_ratio, correlation],
                }
            )
            ratio_df.to_csv(
                os.path.join(os.path.dirname(output_path), "intensity_ratios.csv"),
                index=False,
            )
        else:
            print("No valid intensity data for correlation heatmap")
    except Exception as e:
        print(f"Error creating intensity correlation heatmap: {e}")
        traceback.print_exc()


def create_histogram_grid(scan_data, output_path):
    """Create a grid of histograms showing intensity distributions."""
    if not scan_data:
        print("No scan data available for histogram grid")
        return

    try:
        sample_limit = min(9, len(scan_data))
        samples = scan_data[:sample_limit]

        fig = plt.figure(figsize=(15, 15))
        rows = int(np.ceil(np.sqrt(sample_limit)))
        cols = int(np.ceil(sample_limit / rows))

        for i, scan_info in enumerate(samples):
            if i >= sample_limit:
                break

            sample_id = scan_info.get("sample_id", f"sample_{i}")

            if scan_info["invivo_data"] is None or scan_info["exvivo_data"] is None:
                continue

            invivo_data = scan_info["invivo_data"].flatten()
            exvivo_data = scan_info["exvivo_data"].flatten()

            invivo_fg = invivo_data[invivo_data > 0]
            exvivo_fg = exvivo_data[exvivo_data > 0]

            ax = fig.add_subplot(rows, cols, i + 1)

            ax.hist(invivo_fg, bins=50, alpha=0.5, label="In-vivo", density=True)
            ax.hist(exvivo_fg, bins=50, alpha=0.5, label="Ex-vivo", density=True)

            ax.set_title(f"{sample_id}")
            if i == 0:
                ax.legend()

            invivo_mean = np.mean(invivo_fg)
            exvivo_mean = np.mean(exvivo_fg)
            ratio = exvivo_mean / invivo_mean if invivo_mean > 0 else 0

            ax.text(
                0.05,
                0.95,
                f"Ratio: {ratio:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.7),
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved histogram grid to {output_path}")

    except Exception as e:
        print(f"Error creating histogram grid: {e}")
        traceback.print_exc()
