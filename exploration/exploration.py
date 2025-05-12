import os
import argparse
from analysis import analyze_dataset, analyze_before_processing_dataset
from visualization import (
    create_average_volumes,
    visualize_average_volumes,
    create_intensity_correlation_heatmap,
    create_histogram_grid,
)
from statistics import perform_pca_analysis
from texture import edge_detection_comparison, advanced_texture_analysis
from registration import perform_registration_analysis
from residual import calculate_residual

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prostate MRI Data Exploration")
    parser.add_argument(
        "--before-processing",
        action="store_true",
        help="Use raw .mha files instead of organized data",
    )
    parser.add_argument(
        "--invivo-dir",
        type=str,
        default="/Users/julianroddeman/Desktop/invivos",
        help="Directory containing in-vivo scans (before processing mode)",
    )
    parser.add_argument(
        "--exvivo-dir",
        type=str,
        default="/Users/julianroddeman/Desktop/exvivo_to_check/masked_scans",
        help="Directory containing ex-vivo scans (before processing mode)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="organized_data",
        help="Base directory for organized data",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Perform advanced analysis (PCA, registration, edge detection, etc.)",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Calculate residual between two specific scans",
    )
    parser.add_argument(
        "--img1",
        type=str,
        default=None,
        help="First image for residual analysis",
    )
    parser.add_argument(
        "--img2",
        type=str,
        default=None,
        help="Second image for residual analysis",
    )
    parser.add_argument(
        "--abs-diff",
        action="store_true",
        help="Use absolute difference for residual analysis",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            "before_processing"
            if args.before_processing
            else "data_exploration_results"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "histograms"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "structural"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "paired_visualizations"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "texture"), exist_ok=True)

    if args.advanced:
        os.makedirs(os.path.join(args.output_dir, "edge_detection"), exist_ok=True)
        os.makedirs(
            os.path.join(args.output_dir, "intensity_correlation"), exist_ok=True
        )
        os.makedirs(os.path.join(args.output_dir, "pca"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "registration"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "advanced_texture"), exist_ok=True)

    if args.residual:
        if args.img1 is None or args.img2 is None:
            print("Error: --img1 and --img2 must be specified for residual analysis")
            exit(1)
        os.makedirs(os.path.join(args.output_dir, "residual"), exist_ok=True)
        calculate_residual(
            args.img1,
            args.img2,
            os.path.join(args.output_dir, "residual"),
            abs_diff=args.abs_diff,
        )
    else:
        scan_data = []

        if args.before_processing:
            stats_df, struct_df, texture_df = analyze_before_processing_dataset(
                args.invivo_dir, args.exvivo_dir, args.output_dir
            )
        else:
            stats_df, struct_df, texture_df = analyze_dataset(
                args.base_dir, args.output_dir
            )

        if args.advanced:
            print("Performing advanced analysis...")
            # Collect scan data for advanced analysis
            if args.before_processing:
                from glob import glob
                from data_loading import load_scan

                invivo_files = sorted(glob(os.path.join(args.invivo_dir, "*.mha")))
                for invivo_file in invivo_files:
                    sample_id = os.path.basename(invivo_file).replace(".mha", "")
                    exvivo_file = os.path.join(args.exvivo_dir, f"{sample_id}.mha")

                    if os.path.exists(exvivo_file):
                        invivo_img, invivo_data = load_scan(invivo_file)
                        exvivo_img, exvivo_data = load_scan(exvivo_file)

                        if invivo_data is not None and exvivo_data is not None:
                            scan_data.append(
                                {
                                    "sample_id": sample_id,
                                    "invivo_data": invivo_data,
                                    "exvivo_data": exvivo_data,
                                    "invivo_img": invivo_img,
                                    "exvivo_img": exvivo_img,
                                }
                            )
            else:
                from glob import glob
                from data_loading import load_scan

                sets = ["train", "test"]
                for set_name in sets:
                    invivo_dir = os.path.join(args.base_dir, set_name, "invivo")
                    exvivo_dir = os.path.join(args.base_dir, set_name, "exvivo")

                    if os.path.exists(invivo_dir) and os.path.exists(exvivo_dir):
                        invivo_files = sorted(glob(os.path.join(invivo_dir, "*.nii*")))

                        for invivo_file in invivo_files:
                            sample_id = (
                                os.path.basename(invivo_file)
                                .replace(".nii.gz", "")
                                .replace(".nii", "")
                            )
                            exvivo_file = os.path.join(exvivo_dir, f"{sample_id}.nii")

                            if not os.path.exists(exvivo_file):
                                exvivo_file = os.path.join(
                                    exvivo_dir, f"{sample_id}.nii.gz"
                                )

                            if os.path.exists(exvivo_file):
                                invivo_img, invivo_data = load_scan(invivo_file)
                                exvivo_img, exvivo_data = load_scan(exvivo_file)

                                if invivo_data is not None and exvivo_data is not None:
                                    scan_data.append(
                                        {
                                            "sample_id": sample_id,
                                            "invivo_data": invivo_data,
                                            "exvivo_data": exvivo_data,
                                            "invivo_img": invivo_img,
                                            "exvivo_img": exvivo_img,
                                        }
                                    )

            # Perform advanced analysis if scan data is available
            if scan_data:
                print(f"Found {len(scan_data)} paired scans for advanced analysis")

                # Create average volumes
                invivo_avg, exvivo_avg = create_average_volumes(scan_data)
                if invivo_avg is not None and exvivo_avg is not None:
                    visualize_average_volumes(
                        invivo_avg,
                        exvivo_avg,
                        os.path.join(args.output_dir, "average_volumes.png"),
                    )

                # Create intensity correlation heatmap
                create_intensity_correlation_heatmap(
                    scan_data,
                    os.path.join(
                        args.output_dir, "intensity_correlation", "heatmap.png"
                    ),
                )

                # Create histogram grid
                create_histogram_grid(
                    scan_data,
                    os.path.join(args.output_dir, "histograms", "histogram_grid.png"),
                )

                # Perform edge detection comparison
                edge_detection_comparison(
                    scan_data, os.path.join(args.output_dir, "edge_detection")
                )

                # Perform advanced texture analysis
                advanced_texture_analysis(
                    scan_data, os.path.join(args.output_dir, "advanced_texture")
                )

                # Perform PCA analysis
                perform_pca_analysis(scan_data, os.path.join(args.output_dir, "pca"))

                # Perform registration analysis
                perform_registration_analysis(
                    scan_data, os.path.join(args.output_dir, "registration")
                )
            else:
                print("No scan data available for advanced analysis")
