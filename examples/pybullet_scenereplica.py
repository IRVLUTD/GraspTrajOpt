import os, sys
import time
import numpy as np
import pybullet as p
import open3d as o3d
import argparse
import scipy
import matplotlib.pyplot as plt
from pybullet_api import Fetch, Panda
from transforms3d.quaternions import mat2quat
from utils import *


class SceneReplicaEnv():

    def __init__(self, data_dir, robot_name='fetch', scene_type='tabletop'):

        self.data_dir = data_dir
        self.model_dir = os.path.join(data_dir, "objects")
        self.scenes_path = os.path.join(data_dir, "final_scenes", "scene_data")
        self.scene_type = scene_type

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
            arm_height = 1.1
        elif robot_name == 'panda':
            base_position = np.array([0.05, 0, 0.7])
            arm_height = 0
        else:
            print(f'robot {robot_name} not supported')
            sys.exit(1)
        self.base_position = base_position
        self.arm_height = arm_height
        self.recorded_gripper_position = None

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
            if self.scene_type == 'shelf':
                p.resetDebugVisualizerCamera(2.0, -45.0, -41.0, [0.45, 0, 0.45])
            else:
                p.resetDebugVisualizerCamera(1.8, 90.0, -45.0, [0.8, 0, 0.8])
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

        # set camera for recording propurse
        # Set the camera settings.
        if self.scene_type == 'tabletop':
            look = [0.8, 0, 0.8]   
            distance = 1.8
            pitch = -45.0   
            yaw = 90
        else:
            look = [0.45, 0, 1.0]   
            distance = 1.6
            pitch = -25.0   
            yaw = -45.0
        roll = 0
        fov = 60.0
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            look, distance, yaw, pitch, roll, 2
        )
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, self.near, self.far
        )                

        # set robot
        if robot_name == 'fetch':
            base_position = np.zeros((3, ))
            self.robot = Fetch(base_position, self.scene_type)
        elif robot_name == 'panda':
            self.robot = Panda(base_position, self.scene_type)        
        self.robot.retract()    

        # Set table and plane
        plane_file = os.path.join(self.root_dir, '../data/objects/floor/model_normalized.urdf') # _white
        self.plane_id = p.loadURDF(plane_file, [0, 0, 0])
        self.light_position = np.array([-1.0, 0, 2.5])
        if self.scene_type == 'tabletop':
            table_file = os.path.join(self.root_dir, '../data/objects/cafe_table/cafe_table.urdf')
            texture_file = os.path.join(self.root_dir, '../data/objects/cafe_table/materials/textures/Maple.jpg')
            texture_id = p.loadTexture(texture_file)
            self.obj_path = [plane_file, table_file]
            # z_offset = -0.03  # difference between Real World and table CAD model
            z_offset = 0
            self.table_or_shelf_pos = np.array([0.8, 0, z_offset])
            self.table_id = p.loadURDF(table_file, self.table_or_shelf_pos)
            p.changeVisualShape(self.table_id, -1, textureUniqueId=texture_id)
            self.table_height = 0.75
            p.changeDynamics(
                self.table_id,
                -1,
                restitution=0.1,
                spinningFriction=1.0,
                rollingFriction=1.0,
                lateralFriction=1.0,
            )
        elif self.scene_type == 'shelf':
            shelf_file = os.path.join(self.root_dir, '../data/objects/shelf/shelf.urdf')
            self.obj_path = [plane_file, shelf_file]
            # z_offset = -0.03  # difference between Real World and table CAD model
            z_offset = 0.7 + 0.25
            self.table_or_shelf_pos = np.array([0.9, 0, z_offset])
            self.shelf_orn = [0, 0, 1, 0]
            self.shelf_id = p.loadURDF(shelf_file, self.table_or_shelf_pos, self.shelf_orn)
            self.shelf_height = 0.8
            self.shelf_interval = 0.2
            p.changeDynamics(
                self.shelf_id,
                -1,
                restitution=0.1,
                spinningFriction=1.0,
                rollingFriction=1.0,
                lateralFriction=1.0,
            )

        # cache objects
        self.object_uids = []
        self.object_names = []
        self.cache_object_poses = []
        self.cache_objects()     


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
                mass=0.05,
                spinningFriction=1.0,
                rollingFriction=1.0,
                lateralFriction=1.0,
            )      


    def setup_scene(self, scene_id):
        # add objects
        print(f"-----------Scene: {scene_id}---------------")
        meta_f = "meta-%06d.mat" % scene_id
        if self.scene_type == 'tabletop':
            meta = scipy.io.loadmat(os.path.join(self.data_dir, "final_scenes", "metadata", meta_f))
        elif self.scene_type == 'shelf':
            filename_meta = os.path.join(self.data_dir, "shelf_scenes", "metadata", meta_f)
            if os.path.exists(filename_meta):
                meta = scipy.io.loadmat(filename_meta)
            else:
                # create scene
                dirname = os.path.join(self.data_dir, "shelf_scenes", "metadata")
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                # randomly sample 6 objects
                num = 6
                index = np.random.permutation(len(self.ycb_object_names))[:num]
                meta = {}
                meta_obj_names = [self.ycb_object_names[i] for i in index]
                meta["object_names"] = meta_obj_names
                # set ordering
                for ordering in ["nearest_first", "random"]:
                    if ordering == 'nearest_first':
                        index = np.arange(num)
                    else:
                        index = np.random.permutation(num)
                    s = ''
                    for i in index:
                        s += f'{meta_obj_names[i]},'
                    meta[ordering] = []
                    meta[ordering].append(s[:-1])

                # place object
                poses = np.zeros((num, 7))
                for i, obj in enumerate(meta_obj_names):
                    obj = obj.strip()
                    filename = os.path.join(self.model_dir, obj, 'textured_simple.obj')
                    mesh = o3d.io.read_triangle_mesh(filename)
                    w, h, l = mesh.get_axis_aligned_bounding_box().get_extent()
                    x, y, z = self.table_or_shelf_pos
                    x -= 0.1
                    y = y - self.shelf_interval + (i % 3) * self.shelf_interval
                    z = z + int(i / 3) * self.shelf_height / 2 + l / 2 + 0.01
                    if obj == '035_power_drill' or '024_bowl':
                        z += 0.05
                    position = [x, y, z]
                    poses[i, :3] = position

                    if obj == '010_potted_meat_can' or obj == '021_bleach_cleanser':
                        quat = [1, 0, 0, 0]
                    elif obj == '009_gelatin_box':
                        quat = [0.42352419407869757, -0.647429444803661, 0.2853495846186877, 0.5657189987875211]
                    elif obj =='008_pudding_box':
                        quat = [0.3433036352820681, 0.3820507270105041, 0.5692984998692289, -0.6419338548787622]
                    elif obj == '035_power_drill':
                        quat = [0.15407648515185643, 0.17465462786331165, -0.6933749354238233, -0.6818998435365614]
                    elif obj == "003_cracker_box" or obj == "004_sugar_box":
                        angle = np.pi / 2
                        RT = rotZ(angle)
                        quat = mat2quat(RT[:3, :3])
                    elif obj == '006_mustard_bottle':
                        angle = np.pi / 4
                        RT = rotZ(angle)
                        quat = mat2quat(RT[:3, :3])
                    else:
                        # randomize z axis
                        angle = np.random.uniform(-np.pi, np.pi)
                        RT = rotZ(angle)
                        quat = mat2quat(RT[:3, :3])
                    # scalar-last (x, y, z, w) format in optas
                    orientation = [quat[1], quat[2], quat[3], quat[0]]
                    poses[i, 3:] = quat
                meta['poses'] = poses
                # save data
                print('save data to', filename_meta)
                scipy.io.savemat(filename_meta, meta)

        # setup scene
        meta_obj_names = meta["object_names"]
        names = []
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
        # other objects
        for i, name in enumerate(self.ycb_object_names):
            if name not in names:
                position, orientation = self.cache_object_poses[i]
                self.set_object_pose(name, position, orientation)
        # start simulation
        self.start()
        time.sleep(2.0)

        # cache object pose
        meta_poses = {}
        for i, obj in enumerate(meta_obj_names):
            obj = obj.strip()
            position, orientation = self.get_object_pose(obj)
            meta_poses[obj] = [position, orientation]
        self.meta_poses = meta_poses  
        return meta
    
    
    def reset_scene(self, set_objects):
        # reset scene
        for obj in set_objects:
            pos, orn = self.meta_poses[obj]
            self.set_object_pose(obj, pos, orn)
        for _ in range(100):
            p.stepSimulation()    


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
        lightDirection = self.table_or_shelf_pos - light_position
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


    def execute_plan(self, plan, video_writer=None):
        '''
        @ param plan: shape (ndof, T)
        '''        
        for t in range(plan.shape[1]):
            self.robot.cmd(plan[:, t])
            if t >= plan.shape[1] - 5:
                num = 500
            else:
                num = 200
            for _ in range(num):
                p.stepSimulation()

            if video_writer is not None:
                _, _, rgba, depth, mask = p.getCameraImage(
                    width=self._window_width,
                    height=self._window_height,
                    viewMatrix=self._view_matrix,
                    projectionMatrix=self._proj_matrix,
                    physicsClientId=self.cid,
                )
                video_writer.write(rgba[:, :, [2, 1, 0]].astype(np.uint8))
                # fig = plt.figure()
                # plt.imshow(rgba[:, :, :3])
                # plt.show()


    def compute_reward(self, object_name):
        """Calculates the reward for the episode.

        The reward is 1 if one of the objects is above height
        """
        reward = 0
        pos_prev, _ = self.meta_poses[object_name]
        pos_gripper_prev = self.recorded_gripper_position
        dis_prev = np.linalg.norm(np.array(pos_prev) - np.array(pos_gripper_prev))

        pos, _ = self.get_object_pose(object_name)
        pos_gripper, _ = p.getLinkState(self.robot._id, self.robot.ee_index)[:2]
        dis = np.linalg.norm(np.array(pos) - np.array(pos_gripper))
        if np.absolute(dis_prev - dis) < 0.1:
            reward = 1
        return reward


    def record_gripper_position(self):
        pos, orn = p.getLinkState(self.robot._id, self.robot.ee_index)[:2]
        self.recorded_gripper_position = pos


    def retract(self, retract_distance=0.3, video_writer=None):
        """Retract step."""
        qc = self.robot.q()
        # keep gripper closed
        for idx in self.robot.finger_index:
            qc[idx] = 0        

        self.step(qc)  # grasp
        pos, orn = p.getLinkState(self.robot._id, self.robot.ee_index)[:2]
        num = 10
        offset = retract_distance / num
        for i in range(10):
            pos = (pos[0], pos[1], pos[2] + offset)
            jointPoses = np.array(p.calculateInverseKinematics(self.robot._id, self.robot.ee_index, pos))
            for idx in self.robot.finger_index:
                jointPoses[idx] = 0.0
            self.step(jointPoses.tolist())

            if video_writer is not None:
                _, _, rgba, depth, mask = p.getCameraImage(
                    width=self._window_width,
                    height=self._window_height,
                    viewMatrix=self._view_matrix,
                    projectionMatrix=self._proj_matrix,
                    physicsClientId=self.cid,
                )
                video_writer.write(rgba[:, :, [2, 1, 0]].astype(np.uint8))            
    

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
        default="panda",
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
        "-t",
        "--scene_type",
        type=str,
        default="tabletop",
        help="tabletop or shelf",
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
    scene_type = args.scene_type

    # create the table environment
    env = SceneReplicaEnv(data_dir, robot_name, scene_type)
    for scene_id in [36, 84, 68, 10, 77, 148, 48, 25, 104, 38, 27, 122, 141, 65, 39, 83, 130, 161, 33, 56]:
        env.setup_scene(scene_id)
        rgba, depth, mask, cam_pose, intrinsic_matrix = env.get_observation()
        input('next scene?')
