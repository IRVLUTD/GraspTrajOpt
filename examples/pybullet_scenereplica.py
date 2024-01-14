import os, sys
import time
import numpy as np
import pybullet as p
import argparse
import scipy
import matplotlib.pyplot as plt
from pybullet_api import Fetch, Panda
from utils import *


class SceneReplicaEnv():

    def __init__(self, data_dir, robot_name='fetch'):

        self.data_dir = data_dir
        self.model_dir = os.path.join(data_dir, "objects")
        self.scenes_path = os.path.join(data_dir, "final_scenes", "scene_data")

        self._renders = True
        self._egl_render = False
        self.connected = False

        self._window_width = 640
        self._window_height = 480
        self.object_uids = []
        self.hz = 50
        self._timeStep = 1. / float(self.hz)
        self.root_dir = os.path.dirname(os.path.abspath(__file__))

        self.connect()
        if robot_name == 'fetch':
            base_position = np.array([0.0, 0.0, 0.0])
        elif robot_name == 'panda':
            base_position = np.array([0.05, 0, 0.7])
        else:
            print(f'robot {robot_name} not supported')
            sys.exit(1)
        self.base_position = base_position

        # load scene ids
        filename = os.path.join(data_dir, 'final_scenes', 'scene_ids.txt')
        self.all_scene_ids = sorted(np.loadtxt(filename).astype(int))

        # all 16 YCB objects in SceneReplica
        self.ycb_object_names = (
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "006_mustard_bottle",
            "007_tuna_fish_can",
            "008_pudding_box",
            "009_gelatin_box",
            "010_potted_meat_can",
            "011_banana",
            "021_bleach_cleanser",
            "024_bowl",
            "025_mug",
            "035_power_drill",
            "037_scissors",
            "040_large_marker",
            "052_extra_large_clamp",
        )
        self.reset(robot_name, base_position)


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

    def start(self):
        p.setRealTimeSimulation(1)

    def stop(self):
        p.setRealTimeSimulation(0)        


    def reset(self, robot_name, base_position):

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)

        p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        # set robot
        if robot_name == 'fetch':
            base_position = np.zeros((3, ))
            self.robot = Fetch(base_position)
        elif robot_name == 'panda':
            self.robot = Panda(base_position)        
        self.robot.retract()    

        # Set table and plane
        self.object_uids = []
        self.object_names = []
        self.cache_object_poses = []
        self.cache_objects()

        plane_file = os.path.join(self.root_dir, 'objects/floor/model_normalized.urdf') # _white
        table_file = os.path.join(self.root_dir, 'objects/cafe_table/cafe_table.urdf')

        self.obj_path = [plane_file, table_file]
        self.plane_id = p.loadURDF(plane_file, [0, 0, 0])
        # z_offset = -0.03  # difference between Real World and table CAD model
        z_offset = 0
        self.table_pos = np.array([0.8, 0, z_offset])
        self.table_id = p.loadURDF(table_file, self.table_pos)
        self.table_height = 0.75
        self.light_position = np.array([-1.0, 0, 2.5])


    def cache_objects(self):
        """
        Load all YCB objects and set up (only work for single apperance)
        """
        num = len(self.ycb_object_names)
        pose = np.zeros([num, 3])
        pose[:, 0] = -2.0 - np.linspace(0, 4, num)  # place in the back
        pose[:, 1] = 2
        for i, name in enumerate(self.ycb_object_names):
            print(f'loading {name}')
            trans = pose[i]
            orn = [0, 0, 0, 1]
            self.cache_object_poses.append((trans.copy(), np.array(orn).copy()))
            uid = self._add_mesh(
                os.path.join(self.model_dir, name, "model_normalized.urdf"), trans, orn
            )  # xyzw
            self.object_uids.append(uid)
            self.object_names.append(name)
            p.changeDynamics(
                uid,
                -1,
                restitution=0.1,
                mass=0.1,
                spinningFriction=1.0,
                rollingFriction=1.0,
                lateralFriction=1.0,
            )      


    def setup_scene(self, scene_id):
        # add objects
        print(f"-----------Scene: {scene_id}---------------")
        meta_f = "meta-%06d.mat" % scene_id
        meta = scipy.io.loadmat(os.path.join(self.data_dir, "final_scenes", "metadata", meta_f))
        meta_obj_names = meta["object_names"]
        names = []
        meta_poses = {}
        for i, obj in enumerate(meta_obj_names):
            obj = obj.strip()
            names.append(obj)
            position = meta["poses"][i][:3]
            position[2] += 0.02
            quat = meta["poses"][i][3:]
            # scalar-last (x, y, z, w) format in optas
            orientation = [quat[1], quat[2], quat[3], quat[0]]
            self.set_object_pose(obj, position, orientation)
            print(obj, position, orientation)
            meta_poses[obj] = [position, orientation]
        self.meta_poses = meta_poses            
        # other objects
        for i, name in enumerate(self.ycb_object_names):
            if name not in names:
                position, orientation = self.cache_object_poses[i]
                self.set_object_pose(name, position, orientation)
        # start simulation
        self.start()
        time.sleep(1.0)
        return meta
    
    
    def reset_scene(self, set_objects):
        # reset scene
        for obj in set_objects:
            pos, orn = self.meta_poses[obj]
            self.set_object_pose(obj, pos, orn)        


    def _add_mesh(self, obj_file, trans, quat, scale=1):
        """
        Add a mesh with URDF file.
        """
        bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        return bid


    def reset_objects(self, object_name):
        idx = self.object_uids[self.object_names.index(object_name)]
        p.resetBasePositionAndOrientation(
            idx,
            [0, 1, 0.1],
            [0, 0, 0, 1],
        )


    def get_object_pose(self, name):
        index = self.object_names.index(name)
        return p.getBasePositionAndOrientation(self.object_uids[index])


    def set_object_pose(self, name, pos, orn):
        idx = self.object_uids[self.object_names.index(name)]
        p.resetBasePositionAndOrientation(idx, pos, orn)        


    def get_camera_view(self):
        """
        Get camera view from the robot
        """
        cam_view_matrix, cam_pose = self.robot.get_camera_pose()     

        fov = 45
        aspect = float(self._window_width) / (self._window_height)
        z_near = 0.1
        z_far = 10
        proj_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, z_near, z_far
        )

        lightDistance = 2.0
        light_position = np.array([-1.0, 0, 2.5])
        lightDirection = self.table_pos - light_position
        lightColor = np.array([1.0, 1.0, 1.0])
        return (
            cam_view_matrix,
            proj_matrix,
            lightDistance,
            lightColor,
            lightDirection,
            z_near,
            z_far,
            cam_pose,
        )            


    def get_observation(self):
        """
        Get observation and visualize
        """

        (
            cam_view_matrix,
            proj_matrix,
            lightDistance,
            lightColor,
            lightDirection,
            near,
            far,
            cam_pose,
        ) = self.get_camera_view()        

        _, _, rgba, depth, mask = p.getCameraImage(width=self._window_width,
                                                   height=self._window_height,
                                                   viewMatrix=cam_view_matrix,
                                                   projectionMatrix=proj_matrix,
                                                   lightDirection = lightDirection,
                                                   lightColor=lightColor,
                                                   lightDistance=lightDistance,
                                                   physicsClientId=self.cid,
                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        # transform depth from NDC to actual depth        
        depth = far * near / (far - (far - near) * depth)
        intrinsic_matrix = projection_to_intrinsics(proj_matrix, self._window_width, self._window_height)
        # self.visualize_data(rgba, depth, mask, depth_pc)
        return rgba, depth, mask, cam_pose, intrinsic_matrix


    def visualize_data(self, rgba, depth, mask, depth_pc):                                               
        # visualization
        fig = plt.figure()
        print(self.object_names)
        print(self.object_uids)
        
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


    def step(self, action):
        self.robot.cmd(action)
        for _ in range(400):
            p.stepSimulation()


    def compute_reward(self, object_name):
        """Calculates the reward for the episode.

        The reward is 1 if one of the objects is above height
        """
        reward = 0
        pos, _ = self.get_object_pose(object_name)
        if pos[2] > self.table_height + 0.2:
            reward = 1
        return reward            


    def retract(self):
        """Retract step."""
        qc = self.robot.q()
        # keep gripper closed
        for idx in self.robot.finger_index:
            qc[idx] = 0        

        self.step(qc)  # grasp
        pos, orn = p.getLinkState(self.robot._id, self.robot.ee_index)[:2]
        for i in range(10):
            pos = (pos[0], pos[1], pos[2] + 0.03)
            jointPoses = np.array(p.calculateInverseKinematics(self.robot._id, self.robot.ee_index, pos))
            for idx in self.robot.finger_index:
                jointPoses[idx] = 0.0
            self.step(jointPoses.tolist())
    

def load_grasps(data_dir, robot_name, model):

    if robot_name == 'fetch':
        # parse grasps
        grasp_dir = os.path.join(data_dir, "grasp_data", "refined_grasps")
        grasp_file = os.path.join(grasp_dir, f"fetch_gripper-{model}.json")
        RT_grasps = parse_grasps(grasp_file)
    elif robot_name == 'panda':
        grasp_dir = os.path.join(data_dir, "grasp_data", "panda_simulated")
        grasp_file = os.path.join(grasp_dir, f"{model}.npy")
        try:
            simulator_grasp = np.load(grasp_file, allow_pickle=True)
            RT_grasps = simulator_grasp.item()["transforms"]
        except:
            simulator_grasp = np.load(
                grasp_file,
                allow_pickle=True,
                fix_imports=True,
                encoding="bytes",
            )
            RT_grasps = simulator_grasp.item()[b"transforms"]
        offset_pose = np.array(rotZ(np.pi / 2))  # and
        RT_grasps = np.matmul(RT_grasps, offset_pose)  # flip x, y 
    return RT_grasps  


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-r",
        "--robot",
        type=str,
        default="fetch",
        help="Robot name",
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
    robot_name = args.robot
    data_dir = args.data_dir
    scene_id = args.scene_id

    # create the table environment
    env = SceneReplicaEnv(data_dir, robot_name)
    env.setup_scene(scene_id)
