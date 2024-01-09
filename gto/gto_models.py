import dis
import os
import _init_paths
import mesh_to_sdf
import pathlib
import numpy as np
from typing import Union, List
import trimesh
import optas
import pyrender
import casadi as cs
from optas.spatialmath import rt2tr, rpy2r, ArrayType
from optas.models import RobotModel

cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory


class GTORobotModel(RobotModel):

    # robot model from optas
    def __init__(
            self,
            model_dir,
            urdf_filename: Union[None, str] = None,
            urdf_string: Union[None, str] = None,
            xacro_filename: Union[None, str] = None,
            name: Union[None, str] = None,
            time_derivs: List[int] = [0],
            qddlim: Union[None, ArrayType] = None,
            T: Union[None, int] = None,
            param_joints: List[str] = [],
            collision_link_names=None,
        ):
        
        super().__init__(urdf_filename, urdf_string, xacro_filename, name, time_derivs, qddlim, T, param_joints)
        self.model_dir = model_dir
        self.collision_link_names = collision_link_names
        self.surface_pc_map = self.compute_link_surface_points()
        self.visual_tf = self.setup_fk_functions()


    def compute_link_surface_points(self):
        # loop over all the links
        urdf = self.get_urdf()
        surface_pc_map = {}
        for urdf_link in urdf.links:
            name = urdf_link.name
            if urdf_link.visual is None:
                continue

            if self.collision_link_names is None or name in self.collision_link_names:
                filename = os.path.join(self.model_dir, urdf_link.visual.geometry.filename)
                print(filename)

                mesh = trimesh.load(filename)
                surface_pc = mesh_to_sdf.get_surface_point_cloud(mesh, surface_point_method='sample', bounding_radius=1, scan_count=100, scan_resolution=400, sample_point_count=100, calculate_normals=True)
                surface_pc_map[name] = surface_pc
                print('surface points', surface_pc.points.shape)
        return surface_pc_map
    
    
    def setup_fk_functions(self):
        # Setup functions to compute visual origins in global frame
        urdf = self.get_urdf()
        q = cs.SX.sym("q", self.ndof)
        link_tf = {}
        visual_tf = {}
        for urdf_link in urdf.links:
            name = urdf_link.name
            if self.collision_link_names is None or name in self.collision_link_names:
                lnk_tf = self.get_global_link_transform(urdf_link.name, q)
                link_tf[name] = cs.Function(f"link_tf_{name}", [q], [lnk_tf])

                xyz, rpy = self.get_link_visual_origin(urdf_link)
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
    

    def setup_workspace_field(self, arm_len, arm_height):
        self.xlim = [0, arm_len]
        self.ylim = [-arm_len, arm_len]
        self.zlim = [0, arm_height + arm_len]

        self.field_margin = 0.2
        self.grid_resolution = 0.02
        self.origin = np.array([self.xlim[0] - self.field_margin, self.ylim[0] - self.field_margin, self.zlim[0] - self.field_margin]).reshape((1, 3))
        workspace_points = np.array(np.meshgrid(
                                np.arange(self.xlim[0] - self.field_margin, self.xlim[1] + self.field_margin, self.grid_resolution),
                                np.arange(self.ylim[0] - self.field_margin, self.ylim[1] + self.field_margin, self.grid_resolution),
                                np.arange(self.zlim[0] - self.field_margin, self.zlim[1] + self.field_margin, self.grid_resolution),
                                indexing='ij'))
        self.field_shape = workspace_points.shape[1:]
        self.workspace_points = workspace_points.reshape((3, -1)).T
        self.field_size = self.workspace_points.shape[0]
        print('workspace field', self.field_size)
        print('workspace points', self.workspace_points.shape)


    def points_to_offsets(self, points):
        n = points.shape[0]
        origin = np.repeat(self.origin, n, axis=0)
        idxes = optas.floor((points - origin) / self.grid_resolution)
        # offset = n_3 + N_3 * (n_2 + N_2 * n_1)
        # https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays
        offsets = idxes[:, 2] + self.field_shape[2] * (idxes[:, 1] + self.field_shape[1] + idxes[:, 0])
        return offsets


if __name__ == "__main__":

    collision_link_names = ["shoulder_pan_link", "shoulder_lift_link", "upperarm_roll_link",
                  "elbow_flex_link", "forearm_roll_link", "wrist_flex_link", "wrist_roll_link", "gripper_link",
                  "l_gripper_finger_link", "r_gripper_finger_link"]
    param_joints = ['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 
                    'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']     

    model_dir = os.path.join(cwd, "../examples/robots", "fetch")
    urdf_filename = os.path.join(model_dir, "fetch.urdf")
    robot_model = GTORobotModel(model_dir, urdf_filename=urdf_filename, 
                                param_joints=param_joints, collision_link_names=None)
    robot_model.setup_workspace_field(arm_len=1.1, arm_height=1.1)

    # forward kinematics
    q_user_input = [0.0] * robot_model.ndof
    q_user_input[2] = 0.4
    points_base_all, normals_base_all = robot_model.compute_fk_surface_points(q_user_input)

    # show points
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_points(points_base_all, normals=normals_base_all))
    scene.add(pyrender.Mesh.from_points(robot_model.workspace_points))
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)