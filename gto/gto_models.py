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


class GTORobotModel():

    # robot model from optas
    def __init__(self, model_dir, robot_model):
        self.model_dir = model_dir
        self.robot_model = robot_model
        self.surface_pc_map = self.compute_link_surface_points()
        self.visual_tf = self.setup_fk_functions()


    def compute_link_surface_points(self):
        # loop over all the links
        urdf = self.robot_model.get_urdf()
        surface_pc_map = {}
        for urdf_link in urdf.links:
            name = urdf_link.name
            if urdf_link.visual is not None:
                filename = os.path.join(self.model_dir, urdf_link.visual.geometry.filename)
                print(filename)

                mesh = trimesh.load(filename)
                surface_pc = mesh_to_sdf.get_surface_point_cloud(mesh, surface_point_method='sample', bounding_radius=1, scan_count=100, scan_resolution=400, sample_point_count=1000, calculate_normals=True)
                surface_pc_map[name] = surface_pc
                print('surface points', surface_pc.points.shape)
        return surface_pc_map
    
    
    def setup_fk_functions(self):
        # Setup functions to compute visual origins in global frame
        urdf = self.robot_model.get_urdf()
        q = cs.SX.sym("q", self.robot_model.ndof)
        link_tf = {}
        visual_tf = {}
        for urdf_link in urdf.links:
            name = urdf_link.name

            lnk_tf = self.robot_model.get_global_link_transform(urdf_link.name, q)
            link_tf[name] = cs.Function(f"link_tf_{name}", [q], [lnk_tf])

            xyz, rpy = self.robot_model.get_link_visual_origin(urdf_link)
            visl_tf = rt2tr(rpy2r(rpy), xyz)

            # move robot base as well
            tf = lnk_tf @ visl_tf
            visual_tf[name] = cs.Function(f"visual_tf_{name}", [q], [tf])
        return visual_tf
    

    def compute_fk_surface_points(self, q_user_input, tf_base=None):
        points_base_all = np.zeros((3, 0), dtype=np.float32)
        normals_base_all = np.zeros((3, 0), dtype=np.float32)
        for name in self.surface_pc_map.keys():
            tf = self.visual_tf[name](q_user_input).toarray()
            if tf_base is not None:
                tf = tf_base @ tf
            surface_pc = self.surface_pc_map[name]
            points = surface_pc.points
            normals = surface_pc.normals

            # transform points
            points_base = tf[:3, :3] @ np.transpose(points) + tf[:3, 3].reshape((3, 1))
            normals_base = tf[:3, :3] @ np.transpose(normals)

            points_base_all = np.concatenate((points_base_all, points_base), axis=1)
            normals_base_all = np.concatenate((normals_base_all, normals_base), axis=1)
        return points_base_all.T, normals_base_all.T


if __name__ == "__main__":

    model_dir = os.path.join(cwd, "../examples/robots", "fetch")
    urdf_filename = os.path.join(model_dir, "fetch_gripper.urdf")
    robot_model = optas.RobotModel(urdf_filename=urdf_filename)
    gto_robot_model = GTORobotModel(model_dir, robot_model)

    # forward kinematics
    q_user_input = [0.0] * robot_model.ndof
    points_base_all, normals_base_all = gto_robot_model.compute_fk_surface_points(q_user_input)

    # show points
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_points(points_base_all, normals=normals_base_all))
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)