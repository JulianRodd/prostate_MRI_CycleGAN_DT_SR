import numpy as np
from scipy.stats import pearsonr


def calculate_statistics(data):
    """Calculate basic statistics for a scan."""
    if data is None:
        return {
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "p5": 0,
            "p25": 0,
            "p75": 0,
            "p95": 0,
            "volume": 0,
        }

    non_zero = data[data > 0]
    if len(non_zero) == 0:
        return {
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "p5": 0,
            "p25": 0,
            "p75": 0,
            "p95": 0,
            "volume": 0,
        }

    stats = {
        "mean": np.mean(non_zero),
        "median": np.median(non_zero),
        "std": np.std(non_zero),
        "min": np.min(non_zero),
        "max": np.max(non_zero),
        "p5": np.percentile(non_zero, 5),
        "p25": np.percentile(non_zero, 25),
        "p75": np.percentile(non_zero, 75),
        "p95": np.percentile(non_zero, 95),
        "volume": len(non_zero),
    }
    return stats


def compute_similarity_metrics(image1, image2, mask1=None, mask2=None):
    """Compute PSNR, SSIM, and correlation between two images."""
    from skimage import metrics

    if image1 is None or image2 is None:
        return {"psnr": 0, "ssim": 0, "correlation": 0, "mse": 0, "nrmse": 0}

    if mask1 is not None:
        image1 = image1 * (mask1 > 0)
    if mask2 is not None:
        image2 = image2 * (mask2 > 0)

    foreground1 = image1 > 0
    foreground2 = image2 > 0
    combined_mask = foreground1 & foreground2

    if np.sum(combined_mask) == 0:
        return {"psnr": 0, "ssim": 0, "correlation": 0, "mse": 0, "nrmse": 0}

    img1_fg = image1[combined_mask]
    img2_fg = image2[combined_mask]

    img1_norm = (
        (img1_fg - np.min(img1_fg)) / (np.max(img1_fg) - np.min(img1_fg))
        if np.max(img1_fg) > np.min(img1_fg)
        else np.zeros_like(img1_fg)
    )
    img2_norm = (
        (img2_fg - np.min(img2_fg)) / (np.max(img2_fg) - np.min(img2_fg))
        if np.max(img2_fg) > np.min(img2_fg)
        else np.zeros_like(img2_fg)
    )

    img1_full_norm = np.zeros_like(image1)
    img2_full_norm = np.zeros_like(image2)
    img1_full_norm[combined_mask] = img1_norm
    img2_full_norm[combined_mask] = img2_norm

    try:
        mse = np.mean((img1_norm - img2_norm) ** 2)
        nrmse = (
            np.sqrt(mse) / (np.max(img1_norm) - np.min(img1_norm))
            if np.max(img1_norm) > np.min(img1_norm)
            else 1.0
        )

        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))

        if len(image1.shape) == 3:
            middle_slice_idx = image1.shape[2] // 2
            ssim = metrics.structural_similarity(
                img1_full_norm[:, :, middle_slice_idx],
                img2_full_norm[:, :, middle_slice_idx],
                data_range=1.0,
            )
        else:
            ssim = metrics.structural_similarity(
                img1_full_norm,
                img2_full_norm,
                data_range=1.0,
            )

        try:
            if len(img1_norm) > 1 and np.std(img1_norm) > 0 and np.std(img2_norm) > 0:
                corr_result = pearsonr(img1_norm, img2_norm)
                if hasattr(corr_result, "statistic"):
                    corr = corr_result.statistic
                elif isinstance(corr_result, tuple) and len(corr_result) > 0:
                    corr = corr_result[0]
                else:
                    corr = 0
            else:
                corr = 0
        except Exception as e:
            print(f"Correlation calculation error: {e}")
            corr = 0
    except Exception as e:
        print(f"Error calculating similarity metrics: {e}")
        psnr = ssim = corr = mse = nrmse = 0

    return {"psnr": psnr, "ssim": ssim, "correlation": corr, "mse": mse, "nrmse": nrmse}


def perform_pca_analysis(scan_data, output_dir):
    """Perform PCA analysis on image features from both modalities."""
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    if not scan_data:
        print("No scan data available for PCA analysis")
        return

    try:
        invivo_features = []
        exvivo_features = []
        sample_ids = []

        for i, scan_info in enumerate(scan_data):
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

            invivo_mask = invivo_slice > 0
            exvivo_mask = exvivo_slice > 0

            if np.sum(invivo_mask) == 0 or np.sum(exvivo_mask) == 0:
                continue

            invivo_values = invivo_slice[invivo_mask]
            exvivo_values = exvivo_slice[exvivo_mask]

            invivo_feat = [
                np.mean(invivo_values),
                np.std(invivo_values),
                np.percentile(invivo_values, 25),
                np.percentile(invivo_values, 50),
                np.percentile(invivo_values, 75),
                np.sum(invivo_mask),
            ]

            exvivo_feat = [
                np.mean(exvivo_values),
                np.std(exvivo_values),
                np.percentile(exvivo_values, 25),
                np.percentile(exvivo_values, 50),
                np.percentile(exvivo_values, 75),
                np.sum(exvivo_mask),
            ]

            invivo_features.append(invivo_feat)
            exvivo_features.append(exvivo_feat)
            sample_ids.append(sample_id)

        if not invivo_features or not exvivo_features:
            print("No features extracted for PCA analysis")
            return

        invivo_features = np.array(invivo_features)
        exvivo_features = np.array(exvivo_features)

        scaler = StandardScaler()
        invivo_scaled = scaler.fit_transform(invivo_features)
        exvivo_scaled = scaler.fit_transform(exvivo_features)

        pca_invivo = PCA(n_components=2)
        invivo_pca = pca_invivo.fit_transform(invivo_scaled)

        pca_exvivo = PCA(n_components=2)
        exvivo_pca = pca_exvivo.fit_transform(exvivo_scaled)

        combined_features = np.concatenate([invivo_scaled, exvivo_scaled], axis=0)
        pca_combined = PCA(n_components=2)
        combined_pca = pca_combined.fit_transform(combined_features)

        invivo_combined_pca = combined_pca[: len(invivo_features)]
        exvivo_combined_pca = combined_pca[len(invivo_features) :]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        axes[0].scatter(invivo_pca[:, 0], invivo_pca[:, 1], c="blue", label="In-vivo")
        axes[0].scatter(exvivo_pca[:, 0], exvivo_pca[:, 1], c="red", label="Ex-vivo")
        for i, txt in enumerate(sample_ids):
            axes[0].annotate(
                txt, (invivo_pca[i, 0], invivo_pca[i, 1]), fontsize=8, alpha=0.7
            )
            axes[0].annotate(
                txt, (exvivo_pca[i, 0], exvivo_pca[i, 1]), fontsize=8, alpha=0.7
            )
        axes[0].set_title("Separate PCA for In-vivo and Ex-vivo")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(
            invivo_combined_pca[:, 0],
            invivo_combined_pca[:, 1],
            c="blue",
            label="In-vivo",
        )
        axes[1].scatter(
            exvivo_combined_pca[:, 0],
            exvivo_combined_pca[:, 1],
            c="red",
            label="Ex-vivo",
        )

        for i in range(len(sample_ids)):
            axes[1].plot(
                [invivo_combined_pca[i, 0], exvivo_combined_pca[i, 0]],
                [invivo_combined_pca[i, 1], exvivo_combined_pca[i, 1]],
                "k-",
                alpha=0.3,
            )
            axes[1].annotate(
                sample_ids[i],
                (invivo_combined_pca[i, 0], invivo_combined_pca[i, 1]),
                fontsize=8,
                alpha=0.7,
            )

        axes[1].set_title("Combined PCA with Paired Samples")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pca_analysis.png"), dpi=150)
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        feature_names = ["Mean", "Std Dev", "P25", "P50", "P75", "Area"]

        pc1_importance = np.abs(pca_invivo.components_[0])
        pc2_importance = np.abs(pca_invivo.components_[1])

        x = np.arange(len(feature_names))
        width = 0.35

        axes[0].bar(x - width / 2, pc1_importance, width, label="PC1")
        axes[0].bar(x + width / 2, pc2_importance, width, label="PC2")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(feature_names)
        axes[0].set_title("In-vivo PCA Feature Importance")
        axes[0].legend()

        pc1_importance = np.abs(pca_exvivo.components_[0])
        pc2_importance = np.abs(pca_exvivo.components_[1])

        axes[1].bar(x - width / 2, pc1_importance, width, label="PC1")
        axes[1].bar(x + width / 2, pc2_importance, width, label="PC2")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(feature_names)
        axes[1].set_title("Ex-vivo PCA Feature Importance")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pca_feature_importance.png"), dpi=150)
        plt.close()

        print(f"Saved PCA analysis results to {output_dir}")

        if len(invivo_combined_pca) > 0 and len(exvivo_combined_pca) > 0:
            transformation_vectors = exvivo_combined_pca - invivo_combined_pca

            plt.figure(figsize=(10, 10))
            plt.quiver(
                invivo_combined_pca[:, 0],
                invivo_combined_pca[:, 1],
                transformation_vectors[:, 0],
                transformation_vectors[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="r",
                width=0.005,
            )

            plt.scatter(
                invivo_combined_pca[:, 0],
                invivo_combined_pca[:, 1],
                c="blue",
                label="In-vivo",
            )
            plt.scatter(
                exvivo_combined_pca[:, 0],
                exvivo_combined_pca[:, 1],
                c="red",
                label="Ex-vivo",
            )

            for i, txt in enumerate(sample_ids):
                plt.annotate(
                    txt,
                    (invivo_combined_pca[i, 0], invivo_combined_pca[i, 1]),
                    fontsize=8,
                    alpha=0.7,
                )

            plt.title("PCA Transformation Vectors (In-vivo to Ex-vivo)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "pca_transformation_vectors.png"), dpi=150
            )
            plt.close()

    except Exception as e:
        print(f"Error in PCA analysis: {e}")
        import traceback

        traceback.print_exc()
