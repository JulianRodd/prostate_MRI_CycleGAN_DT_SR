import glob
import os
import shutil


def cleanup_sweep_resources(
    run_id, temp_config_file, temp_output_dir, use_kaggle=False
):
    try:
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)
            print(f"Removed temporary config file: {temp_config_file}")
    except Exception as e:
        print(f"Warning: Failed to remove config file: {str(e)}")

    try:
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            print(f"Removed output directory: {temp_output_dir}")
    except Exception as e:
        print(f"Warning: Failed to remove output directory: {str(e)}")

    try:
        checkpoint_pattern = f"*{run_id}*.pth"
        for checkpoint in glob.glob(os.path.join("../checkpoints", checkpoint_pattern)):
            os.remove(checkpoint)
            print(f"Removed checkpoint: {checkpoint}")
    except Exception as e:
        print(f"Warning: Failed to clean up checkpoints: {str(e)}")

    try:
        temp_pattern = f"*{run_id}*"
        for temp_file in glob.glob(temp_pattern):
            if os.path.isfile(temp_file):
                os.remove(temp_file)
                print(f"Removed temporary file: {temp_file}")
    except Exception as e:
        print(f"Warning: Failed to clean up temporary files: {str(e)}")

    try:
        cache_dir = os.path.join("cache", f"*{run_id}*")
        for cached_item in glob.glob(cache_dir):
            if os.path.isdir(cached_item):
                shutil.rmtree(cached_item)
                print(f"Removed cache directory: {cached_item}")
            elif os.path.isfile(cached_item):
                os.remove(cached_item)
                print(f"Removed cache file: {cached_item}")
    except Exception as e:
        print(f"Warning: Failed to clean up cache: {str(e)}")

    if use_kaggle:
        try:
            for cleanup_dir in ["/kaggle/working/patches", "/kaggle/working/temp"]:
                if os.path.exists(cleanup_dir):
                    for item in os.listdir(cleanup_dir):
                        item_path = os.path.join(cleanup_dir, item)
                        if f"{run_id}" in item:
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
            print("Cleaned up Kaggle working directories")
        except Exception as e:
            print(f"Warning: Failed to clean up Kaggle directories: {str(e)}")
