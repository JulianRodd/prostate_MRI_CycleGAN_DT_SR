import os
import glob
import pandas as pd
import traceback
from tqdm import tqdm
from pathlib import Path

from data_loading import load_scan, print_debug_info
from statistics import calculate_statistics
from structural import calculate_structural_properties
from texture import calculate_texture_features
from visualization import visualize_paired_scans


def analyze_before_processing_dataset(
    invivo_dir, exvivo_dir, output_dir="before_processing"
):
    """Analyze dataset using direct paths to .mha files."""
    print(f"Analyzing in before_processing mode")
    print_debug_info(invivo_dir)
    print_debug_info(exvivo_dir)

    stats_df = pd.DataFrame()
    struct_df = pd.DataFrame()
    texture_df = pd.DataFrame()

    if not os.path.exists(invivo_dir):
        print(f"Error: In-vivo directory '{invivo_dir}' not found")
        return stats_df, struct_df, texture_df

    if not os.path.exists(exvivo_dir):
        print(f"Error: Ex-vivo directory '{exvivo_dir}' not found")
        return stats_df, struct_df, texture_df

    invivo_files = sorted(glob.glob(os.path.join(invivo_dir, "*.mha")))
    print(f"Found {len(invivo_files)} in-vivo files in {invivo_dir}")

    data_found = False

    for invivo_file in tqdm(invivo_files, desc="Processing .mha files"):
        sample_id = os.path.basename(invivo_file).replace(".mha", "")
        exvivo_file = os.path.join(exvivo_dir, f"{sample_id}.mha")

        if not os.path.exists(exvivo_file):
            print(f"Warning: No matching ex-vivo file for {sample_id}")
            continue

        try:
            invivo_img, invivo_data = load_scan(invivo_file)
            exvivo_img, exvivo_data = load_scan(exvivo_file)

            if invivo_data is None or exvivo_data is None:
                print(f"Skipping {sample_id} due to loading errors")
                continue

            invivo_voxel_dims = invivo_img.header.get_zooms()
            exvivo_voxel_dims = exvivo_img.header.get_zooms()

            invivo_stats = calculate_statistics(invivo_data)
            exvivo_stats = calculate_statistics(exvivo_data)

            stats_row = {
                "sample_id": sample_id,
                "dataset": "before_processing",
                "invivo_mean": invivo_stats["mean"],
                "invivo_median": invivo_stats["median"],
                "invivo_std": invivo_stats["std"],
                "invivo_min": invivo_stats["min"],
                "invivo_max": invivo_stats["max"],
                "invivo_p5": invivo_stats["p5"],
                "invivo_p95": invivo_stats["p95"],
                "exvivo_mean": exvivo_stats["mean"],
                "exvivo_median": exvivo_stats["median"],
                "exvivo_std": exvivo_stats["std"],
                "exvivo_min": exvivo_stats["min"],
                "exvivo_max": exvivo_stats["max"],
                "exvivo_p5": exvivo_stats["p5"],
                "exvivo_p95": exvivo_stats["p95"],
                "intensity_ratio": (
                    exvivo_stats["mean"] / invivo_stats["mean"]
                    if invivo_stats["mean"] > 0
                    else 0
                ),
                "voxel_size_ratio_x": (
                    exvivo_voxel_dims[0] / invivo_voxel_dims[0]
                    if invivo_voxel_dims[0] > 0
                    else 0
                ),
                "voxel_size_ratio_y": (
                    exvivo_voxel_dims[1] / invivo_voxel_dims[1]
                    if invivo_voxel_dims[1] > 0
                    else 0
                ),
                "voxel_size_ratio_z": (
                    exvivo_voxel_dims[2] / invivo_voxel_dims[2]
                    if invivo_voxel_dims[2] > 0
                    else 0
                ),
            }
            stats_df = pd.concat(
                [stats_df, pd.DataFrame([stats_row])], ignore_index=True
            )

            invivo_struct = calculate_structural_properties(
                invivo_data, invivo_voxel_dims
            )
            exvivo_struct = calculate_structural_properties(
                exvivo_data, exvivo_voxel_dims
            )

            struct_row = {
                "sample_id": sample_id,
                "dataset": "before_processing",
                "invivo_volume_mm3": invivo_struct["volume_mm3"],
                "exvivo_volume_mm3": exvivo_struct["volume_mm3"],
                "volume_ratio": (
                    exvivo_struct["volume_mm3"] / invivo_struct["volume_mm3"]
                    if invivo_struct["volume_mm3"] > 0
                    else 0
                ),
                "invivo_centroid_x": invivo_struct["centroid_x"],
                "invivo_centroid_y": invivo_struct["centroid_y"],
                "invivo_centroid_z": invivo_struct["centroid_z"],
                "exvivo_centroid_x": exvivo_struct["centroid_x"],
                "exvivo_centroid_y": exvivo_struct["centroid_y"],
                "exvivo_centroid_z": exvivo_struct["centroid_z"],
                "invivo_surface_area": invivo_struct["surface_area"],
                "exvivo_surface_area": exvivo_struct["surface_area"],
                "invivo_surface_to_volume": invivo_struct["surface_to_volume"],
                "exvivo_surface_to_volume": exvivo_struct["surface_to_volume"],
            }
            struct_df = pd.concat(
                [struct_df, pd.DataFrame([struct_row])], ignore_index=True
            )

            invivo_texture = calculate_texture_features(invivo_data)
            exvivo_texture = calculate_texture_features(exvivo_data)

            texture_row = {
                "sample_id": sample_id,
                "dataset": "before_processing",
                "invivo_contrast": invivo_texture["contrast"],
                "invivo_homogeneity": invivo_texture["homogeneity"],
                "invivo_energy": invivo_texture["energy"],
                "invivo_correlation": invivo_texture["correlation"],
                "exvivo_contrast": exvivo_texture["contrast"],
                "exvivo_homogeneity": exvivo_texture["homogeneity"],
                "exvivo_energy": exvivo_texture["energy"],
                "exvivo_correlation": exvivo_texture["correlation"],
                "contrast_ratio": (
                    exvivo_texture["contrast"] / invivo_texture["contrast"]
                    if invivo_texture["contrast"] > 0
                    else 0
                ),
                "homogeneity_ratio": (
                    exvivo_texture["homogeneity"] / invivo_texture["homogeneity"]
                    if invivo_texture["homogeneity"] > 0
                    else 0
                ),
            }
            texture_df = pd.concat(
                [texture_df, pd.DataFrame([texture_row])], ignore_index=True
            )

            output_path = os.path.join(
                output_dir, "paired_visualizations", f"{sample_id}.png"
            )
            visualize_paired_scans(
                invivo_data, exvivo_data, invivo_img, exvivo_img, sample_id, output_path
            )

            data_found = True
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            traceback.print_exc()

    if data_found:
        stats_df.to_csv(
            os.path.join(output_dir, "intensity_statistics.csv"), index=False
        )
        struct_df.to_csv(
            os.path.join(output_dir, "structural_properties.csv"), index=False
        )
        texture_df.to_csv(os.path.join(output_dir, "texture_features.csv"), index=False)
        print(f"Processed {len(stats_df)} samples successfully")
    else:
        print("No data was processed successfully. Check file paths and formats.")

    return stats_df, struct_df, texture_df


def analyze_dataset(base_dir="organized_data", output_dir="data_exploration_results"):
    """Analyze the entire dataset and generate statistics and visualizations."""
    print_debug_info(base_dir)
    sets = ["train", "test"]

    stats_df = pd.DataFrame()
    struct_df = pd.DataFrame()
    texture_df = pd.DataFrame()

    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' not found")
        return stats_df, struct_df, texture_df

    data_found = False

    for set_name in sets:
        set_dir = os.path.join(base_dir, set_name)
        if not os.path.exists(set_dir):
            print(f"Warning: Set directory '{set_dir}' not found, skipping")
            continue

        invivo_dir = os.path.join(set_dir, "invivo")
        exvivo_dir = os.path.join(set_dir, "exvivo")

        if not os.path.exists(invivo_dir):
            print(f"Warning: In-vivo directory '{invivo_dir}' not found, skipping")
            continue

        if not os.path.exists(exvivo_dir):
            print(f"Warning: Ex-vivo directory '{exvivo_dir}' not found, skipping")
            continue

        invivo_files = sorted(glob.glob(os.path.join(invivo_dir, "*.nii")))
        if not invivo_files:
            invivo_files = sorted(glob.glob(os.path.join(invivo_dir, "*.nii.gz")))

        print(f"Found {len(invivo_files)} in-vivo files in {invivo_dir}")

        for invivo_file in tqdm(invivo_files, desc=f"Processing {set_name} set"):
            sample_id = (
                os.path.basename(invivo_file).replace(".nii.gz", "").replace(".nii", "")
            )

            exvivo_file = os.path.join(exvivo_dir, f"{sample_id}.nii")
            if not os.path.exists(exvivo_file):
                exvivo_file = os.path.join(exvivo_dir, f"{sample_id}.nii.gz")

            if not os.path.exists(exvivo_file):
                print(f"Warning: No matching ex-vivo file for {sample_id}")
                continue

            try:
                invivo_img, invivo_data = load_scan(invivo_file)
                exvivo_img, exvivo_data = load_scan(exvivo_file)

                if invivo_data is None or exvivo_data is None:
                    print(f"Skipping {sample_id} due to loading errors")
                    continue

                invivo_voxel_dims = invivo_img.header.get_zooms()
                exvivo_voxel_dims = exvivo_img.header.get_zooms()

                invivo_stats = calculate_statistics(invivo_data)
                exvivo_stats = calculate_statistics(exvivo_data)

                stats_row = {
                    "sample_id": sample_id,
                    "dataset": set_name,
                    "invivo_mean": invivo_stats["mean"],
                    "invivo_median": invivo_stats["median"],
                    "invivo_std": invivo_stats["std"],
                    "invivo_min": invivo_stats["min"],
                    "invivo_max": invivo_stats["max"],
                    "invivo_p5": invivo_stats["p5"],
                    "invivo_p95": invivo_stats["p95"],
                    "exvivo_mean": exvivo_stats["mean"],
                    "exvivo_median": exvivo_stats["median"],
                    "exvivo_std": exvivo_stats["std"],
                    "exvivo_min": exvivo_stats["min"],
                    "exvivo_max": exvivo_stats["max"],
                    "exvivo_p5": exvivo_stats["p5"],
                    "exvivo_p95": exvivo_stats["p95"],
                    "intensity_ratio": (
                        exvivo_stats["mean"] / invivo_stats["mean"]
                        if invivo_stats["mean"] > 0
                        else 0
                    ),
                    "voxel_size_ratio_x": (
                        exvivo_voxel_dims[0] / invivo_voxel_dims[0]
                        if invivo_voxel_dims[0] > 0
                        else 0
                    ),
                    "voxel_size_ratio_y": (
                        exvivo_voxel_dims[1] / invivo_voxel_dims[1]
                        if invivo_voxel_dims[1] > 0
                        else 0
                    ),
                    "voxel_size_ratio_z": (
                        exvivo_voxel_dims[2] / invivo_voxel_dims[2]
                        if invivo_voxel_dims[2] > 0
                        else 0
                    ),
                }
                stats_df = pd.concat(
                    [stats_df, pd.DataFrame([stats_row])], ignore_index=True
                )

                invivo_struct = calculate_structural_properties(
                    invivo_data, invivo_voxel_dims
                )
                exvivo_struct = calculate_structural_properties(
                    exvivo_data, exvivo_voxel_dims
                )

                struct_row = {
                    "sample_id": sample_id,
                    "dataset": set_name,
                    "invivo_volume_mm3": invivo_struct["volume_mm3"],
                    "exvivo_volume_mm3": exvivo_struct["volume_mm3"],
                    "volume_ratio": (
                        exvivo_struct["volume_mm3"] / invivo_struct["volume_mm3"]
                        if invivo_struct["volume_mm3"] > 0
                        else 0
                    ),
                    "invivo_centroid_x": invivo_struct["centroid_x"],
                    "invivo_centroid_y": invivo_struct["centroid_y"],
                    "invivo_centroid_z": invivo_struct["centroid_z"],
                    "exvivo_centroid_x": exvivo_struct["centroid_x"],
                    "exvivo_centroid_y": exvivo_struct["centroid_y"],
                    "exvivo_centroid_z": exvivo_struct["centroid_z"],
                    "invivo_surface_area": invivo_struct["surface_area"],
                    "exvivo_surface_area": exvivo_struct["surface_area"],
                    "invivo_surface_to_volume": invivo_struct["surface_to_volume"],
                    "exvivo_surface_to_volume": exvivo_struct["surface_to_volume"],
                }
                struct_df = pd.concat(
                    [struct_df, pd.DataFrame([struct_row])], ignore_index=True
                )

                invivo_texture = calculate_texture_features(invivo_data)
                exvivo_texture = calculate_texture_features(exvivo_data)

                texture_row = {
                    "sample_id": sample_id,
                    "dataset": set_name,
                    "invivo_contrast": invivo_texture["contrast"],
                    "invivo_homogeneity": invivo_texture["homogeneity"],
                    "invivo_energy": invivo_texture["energy"],
                    "invivo_correlation": invivo_texture["correlation"],
                    "exvivo_contrast": exvivo_texture["contrast"],
                    "exvivo_homogeneity": exvivo_texture["homogeneity"],
                    "exvivo_energy": exvivo_texture["energy"],
                    "exvivo_correlation": exvivo_texture["correlation"],
                    "contrast_ratio": (
                        exvivo_texture["contrast"] / invivo_texture["contrast"]
                        if invivo_texture["contrast"] > 0
                        else 0
                    ),
                    "homogeneity_ratio": (
                        exvivo_texture["homogeneity"] / invivo_texture["homogeneity"]
                        if invivo_texture["homogeneity"] > 0
                        else 0
                    ),
                }
                texture_df = pd.concat(
                    [texture_df, pd.DataFrame([texture_row])], ignore_index=True
                )

                output_path = os.path.join(
                    output_dir, "paired_visualizations", f"{sample_id}.png"
                )
                visualize_paired_scans(
                    invivo_data,
                    exvivo_data,
                    invivo_img,
                    exvivo_img,
                    sample_id,
                    output_path,
                )

                data_found = True
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                traceback.print_exc()

    if data_found:
        stats_df.to_csv(
            os.path.join(output_dir, "intensity_statistics.csv"), index=False
        )
        struct_df.to_csv(
            os.path.join(output_dir, "structural_properties.csv"), index=False
        )
        texture_df.to_csv(os.path.join(output_dir, "texture_features.csv"), index=False)
        print(f"Processed {len(stats_df)} samples successfully")
    else:
        print("No data was processed successfully. Check file paths and formats.")

    return stats_df, struct_df, texture_df
