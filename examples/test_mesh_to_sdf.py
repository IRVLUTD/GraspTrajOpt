import os
import _init_paths
import mesh_to_sdf
import pathlib
import numpy as np
import trimesh
import optas
import pyrender
import casadi as cs
from optas.spatialmath import rt2tr, rpy2r

cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

if __name__ == "__main__":

    model_dir = os.path.join(cwd, "robots", "fetch")
    urdf_filename = os.path.join(model_dir, "fetch.urdf")
    robot_model = optas.RobotModel(urdf_filename=urdf_filename)

    collision_link_names = ["shoulder_pan_link", "shoulder_lift_link", "upperarm_roll_link",
                  "elbow_flex_link", "forearm_roll_link", "wrist_flex_link", "wrist_roll_link", "gripper_link",
                  "l_gripper_finger_link", "r_gripper_finger_link"]
    
    # loop over all the links
    urdf = robot_model.get_urdf()
    surface_pc_map = {}
    for urdf_link in urdf.links:
        name = urdf_link.name
        if urdf_link.visual is not None and name in collision_link_names:
            filename = os.path.join(model_dir, urdf_link.visual.geometry.filename)
            print(filename)

            mesh = trimesh.load(filename)
            surface_pc = mesh_to_sdf.get_surface_point_cloud(mesh, surface_point_method='sample', bounding_radius=1, scan_count=100, scan_resolution=400, sample_point_count=1000, calculate_normals=True)
            surface_pc_map[name] = surface_pc
            print('surface points', surface_pc.points.shape)

    # Setup functions to compute visual origins in global frame
    q = cs.SX.sym("q", robot_model.ndof)
    link_tf = {}
    visual_tf = {}
    for urdf_link in urdf.links:
        name = urdf_link.name

        lnk_tf = robot_model.get_global_link_transform(urdf_link.name, q)
        link_tf[name] = cs.Function(f"link_tf_{name}", [q], [lnk_tf])

        xyz, rpy = robot_model.get_link_visual_origin(urdf_link)
        visl_tf = rt2tr(rpy2r(rpy), xyz)

        # move robot base as well
        tf = lnk_tf @ visl_tf
        visual_tf[name] = cs.Function(f"visual_tf_{name}", [q], [tf])

    # forward kinematics
    q_user_input = [0.0] * robot_model.ndof
    points_base_all = np.zeros((3, 0), dtype=np.float32)
    normals_base_all = np.zeros((3, 0), dtype=np.float32)
    for name in surface_pc_map.keys():
        tf = visual_tf[name](q_user_input).toarray()
        surface_pc = surface_pc_map[name]
        points = surface_pc.points
        normals = surface_pc.normals

        # transform points
        points_base = tf[:3, :3] @ np.transpose(points) + tf[:3, 3].reshape((3, 1))
        normals_base = tf[:3, :3] @ np.transpose(normals)

        points_base_all = np.concatenate((points_base_all, points_base), axis=1)
        normals_base_all = np.concatenate((normals_base_all, normals_base), axis=1)

    # show points
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_points(points_base_all.transpose(), normals=normals_base_all.transpose()))
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)