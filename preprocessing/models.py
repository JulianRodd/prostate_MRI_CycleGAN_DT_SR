import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import SimpleITK as sitk
import numpy as np


@dataclass
class ImageData:
    """Class to store an image and its associated metadata."""

    # Core data
    sitk_image: Optional[sitk.Image] = None
    numpy_array: Optional[np.ndarray] = None

    # Metadata
    spacing: Tuple[float, float, float] = field(default_factory=lambda: (1.0, 1.0, 1.0))
    origin: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    direction: Tuple[float, ...] = field(
        default_factory=lambda: (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )
    dimensions: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))

    # Additional info
    source_path: Optional[Path] = None
    modality: Optional[str] = None
    is_processed: bool = False

    def __post_init__(self):
        """Initialize derived properties after instance creation."""
        self.logger = logging.getLogger(f"{__name__}.ImageData")

        # Extract metadata from SimpleITK image if available
        if self.sitk_image is not None and self.numpy_array is None:
            self.numpy_array = sitk.GetArrayFromImage(self.sitk_image)
            self.spacing = self.sitk_image.GetSpacing()
            self.origin = self.sitk_image.GetOrigin()
            self.direction = self.sitk_image.GetDirection()
            self.dimensions = self.sitk_image.GetSize()

        # Create SimpleITK image from NumPy array if needed
        elif self.numpy_array is not None and self.sitk_image is None:
            self.sitk_image = sitk.GetImageFromArray(self.numpy_array)
            self.sitk_image.SetSpacing(self.spacing)
            self.sitk_image.SetOrigin(self.origin)
            self.sitk_image.SetDirection(self.direction)
            self.dimensions = self.sitk_image.GetSize()

    def update_from_sitk(self, new_image: sitk.Image) -> None:
        """Update the image data from a new SimpleITK image."""
        self.sitk_image = new_image
        self.numpy_array = sitk.GetArrayFromImage(new_image)
        self.spacing = new_image.GetSpacing()
        self.origin = new_image.GetOrigin()
        self.direction = new_image.GetDirection()
        self.dimensions = new_image.GetSize()
        self.is_processed = True

    def update_from_numpy(
        self, new_array: np.ndarray, update_metadata: bool = False
    ) -> None:
        """Update the image data from a new NumPy array."""
        self.numpy_array = new_array

        # Create a new SimpleITK image with the updated array
        new_image = sitk.GetImageFromArray(new_array)

        # Preserve metadata unless explicitly updating
        if not update_metadata and self.sitk_image is not None:
            new_image.SetSpacing(self.spacing)
            new_image.SetOrigin(self.origin)
            new_image.SetDirection(self.direction)
        else:
            self.spacing = new_image.GetSpacing()
            self.origin = new_image.GetOrigin()
            self.direction = new_image.GetDirection()

        self.sitk_image = new_image
        self.dimensions = new_image.GetSize()
        self.is_processed = True

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the image."""
        if self.numpy_array is None:
            return {}

        stats = {
            "shape": self.numpy_array.shape,
            "dimensions": self.dimensions,
            "spacing": self.spacing,
            "min": float(np.min(self.numpy_array)),
            "max": float(np.max(self.numpy_array)),
            "mean": float(np.mean(self.numpy_array)),
            "std": float(np.std(self.numpy_array)),
            "non_zero_voxels": int(np.count_nonzero(self.numpy_array)),
            "total_voxels": int(self.numpy_array.size),
        }

        if stats["total_voxels"] > 0:
            stats["non_zero_percentage"] = (
                stats["non_zero_voxels"] / stats["total_voxels"]
            ) * 100
        else:
            stats["non_zero_percentage"] = 0.0

        return stats

    def save(self, output_path: Union[str, Path], compress: bool = False) -> None:
        """Save the image to disk."""
        if self.sitk_image is None:
            raise ValueError("Cannot save image: No SimpleITK image available")

        output_path = Path(output_path)

        # Make sure extension is .nii instead of .nii.gz if compress is False
        if not compress and str(output_path).endswith(".nii.gz"):
            output_path = Path(str(output_path).replace(".nii.gz", ".nii"))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(self.sitk_image, str(output_path), compress)
        self.logger.debug(f"Saved image to {output_path}")


@dataclass
class ImagePair:
    """Class to store a paired in-vivo and ex-vivo dataset."""

    # Images
    invivo: ImageData = field(default_factory=ImageData)
    exvivo: ImageData = field(default_factory=ImageData)

    # Additional information
    pair_id: str = "unknown"
    flip_anteroposterior: bool = False

    # Bounding box from cropping (min_z, min_y, min_x, max_z, max_y, max_x)
    bbox: Optional[Tuple[int, int, int, int, int, int]] = None

    # Output images
    invivo_hr: Optional[ImageData] = None
    invivo_lr: Optional[ImageData] = None
    exvivo_hr: Optional[ImageData] = None
    exvivo_lr: Optional[ImageData] = None

    def __post_init__(self):
        """Initialize after instance creation."""
        self.logger = logging.getLogger(f"{__name__}.ImagePair")

        # Set pair ID from filenames if available
        if self.pair_id == "unknown" and self.invivo.source_path is not None:
            self.pair_id = Path(self.invivo.source_path).stem

    def set_pair_id(self, pair_id: str) -> None:
        """Set the pair ID."""
        self.pair_id = pair_id
        self.logger = logging.getLogger(f"{__name__}.ImagePair.{pair_id}")

    def save_results(
        self, output_dir: Union[str, Path], split: str = "train"
    ) -> Dict[str, Path]:
        """Save all processed images to appropriate directories."""
        output_dir = Path(output_dir)
        output_paths = {}

        # Save main images
        if self.invivo.is_processed:
            invivo_path = output_dir / split / "invivo" / f"{self.pair_id}.nii.gz"
            self.invivo.save(invivo_path)
            output_paths["invivo"] = invivo_path

        if self.exvivo.is_processed:
            exvivo_path = output_dir / split / "exvivo" / f"{self.pair_id}.nii.gz"
            self.exvivo.save(exvivo_path)
            output_paths["exvivo"] = exvivo_path

        # Save super-resolution pairs
        if self.invivo_hr is not None and self.invivo_lr is not None:
            hr_path = output_dir / split / "sr_hr" / f"{self.pair_id}.nii.gz"
            lr_path = output_dir / split / "sr_lr" / f"{self.pair_id}.nii.gz"
            self.invivo_hr.save(hr_path)
            self.invivo_lr.save(lr_path)
            output_paths["sr_hr"] = hr_path
            output_paths["sr_lr"] = lr_path

        if self.exvivo_hr is not None and self.exvivo_lr is not None:
            hr_path = output_dir / split / "sr_hr" / f"{self.pair_id}.nii.gz"
            lr_path = output_dir / split / "sr_lr" / f"{self.pair_id}.nii.gz"
            self.exvivo_hr.save(hr_path)
            self.exvivo_lr.save(lr_path)
            output_paths["sr_hr"] = hr_path
            output_paths["sr_lr"] = lr_path

        return output_paths


@dataclass
class ProcessingResult:
    """Results and metrics from image processing."""

    # Processed image pair
    image_pair: ImagePair

    # Processing success status
    success: bool = True
    error_message: Optional[str] = None

    # Timestamps
    start_time: float = 0.0
    end_time: float = 0.0

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Registration results
    registration_transform: Optional[sitk.Transform] = None
    registration_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def processing_time(self) -> float:
        """Return total processing time in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert processing result to a dictionary for serialization."""
        result = {
            "pair_id": self.image_pair.pair_id,
            "success": self.success,
            "processing_time": self.processing_time,
            "metrics": self.metrics,
            "registration_metrics": self.registration_metrics,
        }

        if self.error_message:
            result["error_message"] = self.error_message

        return result
