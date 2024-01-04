import os
import _init_paths
import mesh_to_sdf
import pathlib
import numpy as np
import trimesh


cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

if __name__ == "__main__":

    filename = os.path.join(cwd, "robots", "fetch", "meshes", "gripper_link.obj")
    mesh = trimesh.load(filename)

    pc = mesh_to_sdf.get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=1, scan_count=100, scan_resolution=400, sample_point_count=10000000, calculate_normals=True)
    print(pc)