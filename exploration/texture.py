import numpy as np
from skimage.feature import graycomatrix, graycoprops


def calculate_texture_features(data):
    """Calculate texture features using GLCM."""
    if data is None or data.size == 0:
        return {
            "contrast": 0,
            "dissimilarity": 0,
            "homogeneity": 0,
            "energy": 0,
            "correlation": 0,
            "ASM": 0,
        }

    if np.max(data) - np.min(data) > 0:
        data_norm = (
            (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        ).astype(np.uint8)
    else:
        data_norm = np.zeros_like(data, dtype=np.uint8)

    if data_norm.shape[2] > 0:
        middle_slice = data_norm[:, :, data_norm.shape[2] // 2]
    else:
        return {
            "contrast": 0,
            "dissimilarity": 0,
            "homogeneity": 0,
            "energy": 0,
            "correlation": 0,
            "ASM": 0,
        }

    if np.sum(middle_slice) == 0:
        return {
            "contrast": 0,
            "dissimilarity": 0,
            "homogeneity": 0,
            "energy": 0,
            "correlation": 0,
            "ASM": 0,
        }

    try:
        glcm = graycomatrix(
            middle_slice,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        contrast = graycoprops(glcm, "contrast").mean()
        dissimilarity = graycoprops(glcm, "dissimilarity").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        energy = graycoprops(glcm, "energy").mean()
        correlation = graycoprops(glcm, "correlation").mean()
        asm = graycoprops(glcm, "ASM").mean()
    except Exception as e:
        print(f"Error calculating texture features: {e}")
        contrast = dissimilarity = homogeneity = energy = correlation = asm = 0

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "ASM": asm,
    }


def edge_detection_comparison(scan_data, output_dir):
    """Detect edges in both modalities and compare them."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from skimage import filters, feature

    if not scan_data:
        print("No scan data available for edge detection comparison")
        return

    try:
        for i, scan_info in enumerate(scan_data):
            if scan_info["invivo_data"] is None or scan_info["exvivo_data"] is None:
                continue

            sample_id = scan_info.get("sample_id", f"sample_{i}")
            output_path = os.path.join(output_dir, f"{sample_id}_edge_comparison.png")

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
                continue

            invivo_blur = filters.gaussian(invivo_norm, sigma=1.0)
            exvivo_blur = filters.gaussian(exvivo_norm, sigma=1.0)

            invivo_edges = feature.canny(invivo_blur, sigma=2.0)
            exvivo_edges = feature.canny(exvivo_blur, sigma=2.0)

            overlap = np.logical_and(invivo_edges, exvivo_edges)
            overlap_percentage = (
                100
                * np.sum(overlap)
                / (np.sum(invivo_edges) + np.sum(exvivo_edges) - np.sum(overlap))
            )

            edge_comparison = np.zeros(
                (invivo_edges.shape[0], invivo_edges.shape[1], 3)
            )
            edge_comparison[:, :, 0] = invivo_edges
            edge_comparison[:, :, 1] = exvivo_edges

            fig = plt.figure(figsize=(15, 12))
            gs = gridspec.GridSpec(2, 3, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(invivo_norm, cmap="gray")
            ax1.set_title("In-vivo")
            ax1.axis("off")

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(exvivo_norm, cmap="gray")
            ax2.set_title("Ex-vivo")
            ax2.axis("off")

            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(edge_comparison)
            ax3.set_title(f"Edge Overlay (Overlap: {overlap_percentage:.1f}%)")
            ax3.axis("off")

            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(invivo_edges, cmap="gray")
            ax4.set_title("In-vivo Edges")
            ax4.axis("off")

            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(exvivo_edges, cmap="gray")
            ax5.set_title("Ex-vivo Edges")
            ax5.axis("off")

            ax6 = fig.add_subplot(gs[1, 2])
            ax6.axis("off")

            edge_explanation = (
                "EDGE COMPARISON ANALYSIS\n\n"
                f"Edge Overlap: {overlap_percentage:.1f}%\n\n"
                "INTERPRETATION:\n\n"
                "- RED: Edges only present in in-vivo images\n"
                "- GREEN: Edges only present in ex-vivo images\n"
                "- YELLOW: Edges present in both modalities\n\n"
                "This analysis reveals:\n"
                "- Which structural features are preserved across modalities\n"
                "- Areas where ex-vivo shows more detail (green edges)\n"
                "- Features that are lost in the ex-vivo preparation (red edges)\n\n"
                "ML MODEL IMPLICATIONS:\n\n"
                "- Focus on preserving yellow edges (common features)\n"
                "- Consider edge-aware loss functions\n"
                "- Lower overlap suggests more complex transformation needed"
            )

            ax6.text(
                0,
                1,
                edge_explanation,
                va="top",
                ha="left",
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=1"),
            )

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            print(f"Saved edge comparison for {sample_id}")

            with open(
                os.path.join(output_dir, "edge_overlap_percentages.csv"), "a"
            ) as f:
                if (
                    os.path.getsize(
                        os.path.join(output_dir, "edge_overlap_percentages.csv")
                    )
                    == 0
                ):
                    f.write("sample_id,overlap_percentage\n")
                f.write(f"{sample_id},{overlap_percentage:.1f}\n")
    except Exception as e:
        print(f"Error in edge detection comparison: {e}")
        import traceback

        traceback.print_exc()


def advanced_texture_analysis(scan_data, output_dir):
    """Analyze texture features of in-vivo and ex-vivo scans using advanced metrics."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from skimage import filters
    from tqdm import tqdm

    if not scan_data:
        print("No scan data available for texture analysis")
        return

    try:
        texture_data = []

        for i, scan_info in enumerate(tqdm(scan_data, desc="Analyzing textures")):
            if scan_info["invivo_data"] is None or scan_info["exvivo_data"] is None:
                continue

            sample_id = scan_info.get("sample_id", f"sample_{i}")

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
                continue

            invivo_uint8 = (invivo_norm * 255).astype(np.uint8)
            exvivo_uint8 = (exvivo_norm * 255).astype(np.uint8)

            try:
                invivo_entropy = filters.rank.entropy(
                    invivo_uint8, np.ones((7, 7), dtype=np.uint8)
                )
                invivo_entropy = invivo_entropy[invivo_mask]
                invivo_entropy_mean = np.mean(invivo_entropy)

                invivo_variance = filters.rank.variance(
                    invivo_uint8, np.ones((7, 7), dtype=np.uint8)
                )
                invivo_variance = invivo_variance[invivo_mask]
                invivo_variance_mean = np.mean(invivo_variance)

                exvivo_entropy = filters.rank.entropy(
                    exvivo_uint8, np.ones((7, 7), dtype=np.uint8)
                )
                exvivo_entropy = exvivo_entropy[exvivo_mask]
                exvivo_entropy_mean = np.mean(exvivo_entropy)

                exvivo_variance = filters.rank.variance(
                    exvivo_uint8, np.ones((7, 7), dtype=np.uint8)
                )
                exvivo_variance = exvivo_variance[exvivo_mask]
                exvivo_variance_mean = np.mean(exvivo_variance)

                texture_data.append(
                    {
                        "sample_id": sample_id,
                        "invivo_entropy": invivo_entropy_mean,
                        "exvivo_entropy": exvivo_entropy_mean,
                        "entropy_ratio": (
                            exvivo_entropy_mean / invivo_entropy_mean
                            if invivo_entropy_mean > 0
                            else 0
                        ),
                        "invivo_variance": invivo_variance_mean,
                        "exvivo_variance": exvivo_variance_mean,
                        "variance_ratio": (
                            exvivo_variance_mean / invivo_variance_mean
                            if invivo_variance_mean > 0
                            else 0
                        ),
                    }
                )

                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                axes[0, 0].imshow(invivo_norm, cmap="gray")
                axes[0, 0].set_title("In-vivo")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(exvivo_norm, cmap="gray")
                axes[0, 1].set_title("Ex-vivo")
                axes[0, 1].axis("off")

                im1 = axes[1, 0].imshow(invivo_entropy, cmap="inferno")
                axes[1, 0].set_title(
                    f"In-vivo Entropy (Mean: {invivo_entropy_mean:.2f})"
                )
                axes[1, 0].axis("off")
                fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

                im2 = axes[1, 1].imshow(exvivo_entropy, cmap="inferno")
                axes[1, 1].set_title(
                    f"Ex-vivo Entropy (Mean: {exvivo_entropy_mean:.2f})"
                )
                axes[1, 1].axis("off")
                fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

                im3 = axes[0, 2].imshow(invivo_variance, cmap="viridis")
                axes[0, 2].set_title(
                    f"In-vivo Variance (Mean: {invivo_variance_mean:.2f})"
                )
                axes[0, 2].axis("off")
                fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

                im4 = axes[1, 2].imshow(exvivo_variance, cmap="viridis")
                axes[1, 2].set_title(
                    f"Ex-vivo Variance (Mean: {exvivo_variance_mean:.2f})"
                )
                axes[1, 2].axis("off")
                fig.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"{sample_id}_texture.png"), dpi=150
                )
                plt.close()

            except Exception as e:
                print(f"Error in texture analysis for {sample_id}: {e}")
                continue

        if texture_data:
            texture_df = pd.DataFrame(texture_data)
            texture_df.to_csv(
                os.path.join(output_dir, "advanced_texture_features.csv"), index=False
            )

            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.scatter(texture_df["invivo_entropy"], texture_df["exvivo_entropy"])
            plt.plot(
                [
                    texture_df["invivo_entropy"].min(),
                    texture_df["invivo_entropy"].max(),
                ],
                [
                    texture_df["invivo_entropy"].min(),
                    texture_df["invivo_entropy"].max(),
                ],
                "r--",
            )
            plt.xlabel("In-vivo Entropy")
            plt.ylabel("Ex-vivo Entropy")
            plt.title("Entropy Comparison")
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.scatter(texture_df["invivo_variance"], texture_df["exvivo_variance"])
            plt.plot(
                [
                    texture_df["invivo_variance"].min(),
                    texture_df["invivo_variance"].max(),
                ],
                [
                    texture_df["invivo_variance"].min(),
                    texture_df["invivo_variance"].max(),
                ],
                "r--",
            )
            plt.xlabel("In-vivo Variance")
            plt.ylabel("Ex-vivo Variance")
            plt.title("Variance Comparison")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "texture_comparison.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.boxplot(
                [texture_df["entropy_ratio"], texture_df["variance_ratio"]],
                labels=["Entropy Ratio (Ex/In)", "Variance Ratio (Ex/In)"],
            )
            plt.axhline(y=1.0, color="r", linestyle="--")
            plt.title("Texture Feature Ratios")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "texture_ratios.png"), dpi=150)
            plt.close()

            print(f"Saved texture analysis results to {output_dir}")

    except Exception as e:
        print(f"Error in texture analysis: {e}")
        import traceback

        traceback.print_exc()
