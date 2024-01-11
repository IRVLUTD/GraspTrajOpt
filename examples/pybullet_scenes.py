import os
import time
import numpy as np
import pybullet as p
import argparse
import scipy
import matplotlib.pyplot as plt
from pybullet_api import Fetch
from utils import *
import _init_paths
from mesh_to_sdf.depth_point_cloud import DepthPointCloud
import optas
from optas.visualize import Visualizer
from gto.gto_models import GTORobotModel
from gto.gto_planner import GTOPlanner
from gto.ik_solver import IKSolver
from transforms3d.quaternions import mat2quat
import pyrender

import pathlib
cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory


def pybullet_show_frame(RT):

    origin = RT[:3, 3]
    frame = np.eye(3)
    frame_new = RT[:3, :3] @ frame + origin.reshape((3, 1))

    x_axis = p.addUserDebugLine(lineFromXYZ = origin,
                            lineToXYZ = frame_new[:, 0],
                            lineColorRGB = [1, 0, 0],
                            lineWidth = 0.1,
                            lifeTime = 0
                            )    

    y_axis = p.addUserDebugLine(lineFromXYZ = origin,
                            lineToXYZ = frame_new[:, 1],
                            lineColorRGB = [0, 1, 0],
                            lineWidth = 0.1,
                            lifeTime = 0
                            )
    
    z_axis = p.addUserDebugLine(lineFromXYZ = origin,
                            lineToXYZ = frame_new[:, 2],
                            lineColorRGB = [0, 0, 1],
                            lineWidth = 0.1,
                            lifeTime = 0
                            )    


class SceneReplicaEnv():

    def __init__(self):

        self._renders = True
        self._egl_render = False
        self.connected = False

        self._window_width = 640
        self._window_height = 480
        self.object_uids = []
        self._timeStep = 1. / 1000.
        self.root_dir = os.path.dirname(os.path.abspath(__file__))

        self.connect()
        self.reset()


    def connect(self):
        """
        Connect pybullet.
        """
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.0, 90.0, -41.0, [0.45, 0, 0.75])
        else:
            self.cid = p.connect(p.DIRECT)

        if self._egl_render:
            import pkgutil
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.connected = True


    def reset(self):

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)

        p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        # set robot 
        self.robot = Fetch()
        self.robot.retract()

        # Set table and plane
        self.object_uids = []
        self.object_names = []
        plane_file = os.path.join(self.root_dir, 'objects/floor/model_normalized.urdf') # _white
        table_file = os.path.join(self.root_dir, 'objects/cafe_table/cafe_table.urdf')

        self.obj_path = [plane_file, table_file]
        self.plane_id = p.loadURDF(plane_file, [0, 0, 0])
        # z_offset = -0.03  # difference between Real World and table CAD model
        z_offset = 0
        self.table_pos = np.array([0.8, 0, z_offset])
        self.table_id = p.loadURDF(table_file, self.table_pos)
        self.light_position = np.array([-1.0, 0, 2.5])


    def _add_mesh(self, obj_file, trans, quat, scale=1):
        """
        Add a mesh with URDF file.
        """
        bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        return bid


    def place_objects(self, urdf_filename, name, position, orientation):
        """
        Place of an object to the scene
        """
        uid = self._add_mesh(urdf_filename, position, orientation)  # xyzw
        self.object_uids.append(uid)
        self.object_names.append(name)
        p.resetBaseVelocity(uid, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for _ in range(100):
            p.stepSimulation()


    def get_object_pose(self, name):
        index = self.object_names.index(name)
        return p.getBasePositionAndOrientation(self.object_uids[index])     


    def get_camera_view(self):
        """
        Get head camera view
        """
        head_camera_rgb_optical_frame = 7
        pos, orn = p.getLinkState(self.robot._id, head_camera_rgb_optical_frame)[:2]
        cam_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        cam_pose_mat = unpack_pose(cam_pose)

        fov = 45
        aspect = float(self._window_width) / (self._window_height)
        head_near = 0.1
        head_far = 10
        head_proj_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, head_near, head_far
        )

        # z backward
        RT = cam_pose_mat.dot(rotX(-np.pi))
        # show frame for debugging
        # pybullet_show_frame(RT)
        head_cam_view_matrix = se3_inverse(RT).T.flatten().tolist()

        lightDistance = 2.0
        light_position = np.array([-1.0, 0, 2.5])
        lightDirection = self.table_pos - light_position
        lightColor = np.array([1.0, 1.0, 1.0])
        return (
            head_cam_view_matrix,
            head_proj_matrix,
            lightDistance,
            lightColor,
            lightDirection,
            head_near,
            head_far,
            cam_pose_mat,
        )            


    def get_observation(self):
        """
        Get observation and visualize
        """

        (
            head_cam_view_matrix,
            head_proj_matrix,
            lightDistance,
            lightColor,
            lightDirection,
            near,
            far,
            cam_pose,
        ) = self.get_camera_view()        

        _, _, rgba, depth, mask = p.getCameraImage(width=self._window_width,
                                                   height=self._window_height,
                                                   viewMatrix=head_cam_view_matrix,
                                                   projectionMatrix=head_proj_matrix,
                                                   lightDirection = lightDirection,
                                                   lightColor=lightColor,
                                                   lightDistance=lightDistance,
                                                   physicsClientId=self.cid,
                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        # transform depth from NDC to actual depth        
        depth = far * near / (far - (far - near) * depth)
        intrinsic_matrix = projection_to_intrinsics(head_proj_matrix, self._window_width, self._window_height)
        # self.visualize_data(rgba, depth, mask, depth_pc)
        return rgba, depth, mask, cam_pose, intrinsic_matrix


    def visualize_data(self, rgba, depth, mask, depth_pc):                                               
        # visualization
        fig = plt.figure()
        
        # show RGB image
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(rgba[:, :, :3])
        ax.set_title('RGB image')
        
        # show depth image
        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(depth)
        ax.set_title('depth image')
        
        # show segmentation mask
        ax = fig.add_subplot(2, 2, 3)
        plt.imshow(mask)
        ax.set_title('segmentation mask')   

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        pc_base = depth_pc.points
        ax.scatter(pc_base[:, 0], pc_base[:, 1], pc_base[:, 2], marker='.', color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D ploud cloud in robot base')   
        set_axes_equal(ax)                    
        plt.show()
    

def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="/home/yuxiang/Projects/SceneReplica/data",
        help="SceneReplica data directory",
    )
    parser.add_argument(
        "-s",
        "--scene_id",
        type=int,
        default=10,
        help="SceneReplica scene id",
    )         
    args = parser.parse_args()
    return args
        

# main function
if __name__ == '__main__':
    args = make_args()
    data_dir = args.data_dir
    scene_id = args.scene_id
    model_dir = os.path.join(args.data_dir, "models")
    grasp_dir = os.path.join(args.data_dir, "grasp_data", "refined_grasps")
    scenes_path = os.path.join(args.data_dir, "final_scenes", "scene_data")    

    # load robot model
    robot_model_dir = os.path.join(cwd, "robots", "fetch")
    urdf_filename = os.path.join(robot_model_dir, "fetch.urdf")  
    param_joints = ['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 
                    'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']
    collision_link_names = ["shoulder_pan_link", "shoulder_lift_link", "upperarm_roll_link",
                  "elbow_flex_link", "forearm_roll_link", "wrist_flex_link", "wrist_roll_link", "gripper_link",
                  "l_gripper_finger_link", "r_gripper_finger_link"]      
    robot = GTORobotModel(robot_model_dir,
                          urdf_filename=urdf_filename, 
                          time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
                          param_joints=param_joints,
                          collision_link_names=collision_link_names)
    robot.setup_workspace_field(arm_len=1.1, arm_height=1.1)
    link_ee = "wrist_roll_link"  # end-effector link name
    link_gripper = 'gripper_link'

    # load robot gripper model
    urdf_filename = os.path.join(robot_model_dir, "fetch_gripper.urdf")
    gripper_model = GTORobotModel(robot_model_dir, urdf_filename=urdf_filename)

    # create the table environment
    env = SceneReplicaEnv()
    # add objects
    print(f"-----------Scene: {scene_id}---------------")
    meta_f = "meta-%06d.mat" % scene_id
    meta = scipy.io.loadmat(os.path.join(data_dir, "final_scenes", "metadata", meta_f))
    meta_obj_names = meta["object_names"]
    meta_poses = {}
    for i, obj in enumerate(meta_obj_names):
        obj = obj.strip()
        filename = os.path.join(model_dir, obj, f'{obj}.urdf')
        position = meta["poses"][i][:3]
        position[2] += 0.04
        quat = meta["poses"][i][3:]
        # scalar-last (x, y, z, w) format in optas
        orientation = [quat[1], quat[2], quat[3], quat[0]]
        env.place_objects(filename, obj, position, orientation)
        print(obj, position, orientation)

    # Initialize planner
    print('Initialize planner')
    planner = GTOPlanner(robot, link_ee, link_gripper)    
    ik_solver = IKSolver(robot, link_ee, link_gripper)

    # define the standoff pose
    offset = -0.01
    pose_standoff = np.eye(4, dtype=np.float32)
    pose_standoff[0, 3] = offset
    
    # two orderings
    for ordering in ["nearest_first", "random"]:
        object_order = meta[ordering][0].split(",")
        # for each object
        for object_name in object_order:
                                          
            # render image and compute sdf cost field
            rgba, depth, mask, cam_pose, intrinsic_matrix = env.get_observation()
            idx = env.object_uids[env.object_names.index(object_name)]
            target_mask = mask == idx
            depth_pc = DepthPointCloud(depth, intrinsic_matrix, cam_pose, target_mask)
            sdf_cost = depth_pc.get_sdf_cost(robot.workspace_points, epsilon=0.05)

            # load grasps
            grasp_file = os.path.join(grasp_dir, f"fetch_gripper-{object_name}.json")
            RT_grasps = parse_grasps(grasp_file)

            # query object pose
            pos, orn = env.get_object_pose(object_name)
            obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
            RT_obj = unpack_pose(obj_pose)
            print(object_name, RT_obj)

            # transform grasps to robot base
            n = RT_grasps.shape[0]
            RT_grasps_base = np.zeros_like(RT_grasps)
            in_collision = np.zeros((n, ), dtype=np.int32)
            for i in range(n):
                RT_g = RT_grasps[i]
                RT = RT_obj @ RT_g
                RT_grasps_base[i] = RT

                # check if the grasp is in collision
                RT_off = RT_grasps_base[i] @ pose_standoff
                q_user_input = [0.05] * gripper_model.ndof
                gripper_points, normals = gripper_model.compute_fk_surface_points(q_user_input, tf_base=RT_off)
                sdf = depth_pc.get_sdf(gripper_points)

                ratio = np.sum(sdf < 0) / len(sdf)
                print(f'grasp {i}, collision ratio {ratio}')
                if ratio > 0.01:
                    in_collision[i] = 1

                # visualization
                # colors = np.zeros(gripper_points.shape)
                # colors[sdf < 0, 2] = 1
                # colors[sdf > 0, 0] = 1
                # cloud = pyrender.Mesh.from_points(gripper_points, colors=colors)
                # scene = pyrender.Scene()
                # scene.add(cloud)
                # scene.add(pyrender.Mesh.from_points(depth_pc.points))
                # pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
            
            RT_grasps_base = RT_grasps_base[in_collision == 0] 
            print('Among %d grasps, %d in collision, %d collision-free' % (n, np.sum(in_collision), RT_grasps_base.shape[0]))
            if RT_grasps_base.shape[0] == 0:
                continue

            # test IK for remaining grasps
            n = RT_grasps_base.shape[0]
            found_ik = np.zeros((n, ), dtype=np.int32)
            q0 = np.array(env.robot.q()).reshape((env.robot.ndof, 1))
            for i in range(n):
                RT = RT_grasps_base[i]
                q_solution, err_pos, err_rot = ik_solver.solve_ik(q0, RT)
                if err_pos < 0.01 and err_rot < 5:
                    found_ik[i] = 1
            RT_grasps_base = RT_grasps_base[found_ik == 1] 
            print('Among %d grasps, %d found IK' % (n, np.sum(found_ik)))
            if RT_grasps_base.shape[0] == 0:
                continue

            # plan to a grasp
            qc = env.robot.q()
            # RT = RT_grasps_base[0]
            # plan = planner.plan(qc, RT, sdf_cost)
            plan, index = planner.plan_goalset(qc, RT_grasps_base, sdf_cost)
            RT = RT_grasps_base[index]

            # visualize grasps
            vis = Visualizer(camera_position=[3, 0, 3])
            vis.grid_floor()
            vis.points(
                depth_pc.points,
            )
            q = [0, 0]
            position = RT[:3, 3]
            # scalar-last (x, y, z, w) format in optas
            quat = mat2quat(RT[:3, :3])
            orientation = [quat[1], quat[2], quat[3], quat[0]]
            vis.robot(
                gripper_model,
                base_position=position,
                base_orientation=orientation,
                q=q
            )
            # robot trajectory
            # sample plan
            n = plan.shape[1]
            index = list(range(0, n, 10))
            if index[-1] != n - 1:
                index += [n - 1]
            vis.robot_traj(robot, plan[:, index], alpha_spec={'style': 'A'})
            vis.start()

            env.robot.execute_plan(plan)
            time.sleep(1.0)
            env.robot.close_gripper()
            # list object
            # gipper pose
            pos, orn = p.getLinkState(env.robot._id, env.robot.ee_index)[:2]
            pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
            pose_mat = unpack_pose(pose)
            pose_mat[2, 3] += 0.4
            qc = env.robot.q()
            plan = planner.plan(qc, pose_mat, sdf_cost)
            env.robot.execute_plan(plan)
            # retract
            env.robot.retract()            