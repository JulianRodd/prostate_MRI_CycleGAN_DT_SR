import numpy as np
from skimage import measure


def calculate_structural_properties(data, voxel_dims):
    """Calculate structural properties like volume, surface area, etc."""
    if data is None or voxel_dims is None:
        return {
            "volume_mm3": 0,
            "centroid_x": 0,
            "centroid_y": 0,
            "centroid_z": 0,
            "surface_area": 0,
            "surface_to_volume": 0,
        }

    mask = data > 0
    volume_voxels = np.sum(mask)
    volume_mm3 = volume_voxels * voxel_dims[0] * voxel_dims[1] * voxel_dims[2]

    coords = np.array(np.where(mask)).T
    if len(coords) == 0:
        return {
            "volume_mm3": 0,
            "centroid_x": 0,
            "centroid_y": 0,
            "centroid_z": 0,
            "surface_area": 0,
            "surface_to_volume": 0,
        }

    centroid = np.mean(coords, axis=0)

    try:
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
        surface_area = measure.mesh_surface_area(verts, faces)
        surface_area *= voxel_dims[0] * voxel_dims[1]
        surface_to_volume = surface_area / volume_mm3 if volume_mm3 > 0 else 0
    except Exception as e:
        print(f"Error calculating surface area: {e}")
        surface_area = 0
        surface_to_volume = 0

    return {
        "volume_mm3": volume_mm3,
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "centroid_z": centroid[2],
        "surface_area": surface_area,
        "surface_to_volume": surface_to_volume,
    }
