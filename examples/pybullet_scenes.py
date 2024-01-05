"""
CS 6384 Homework 2 Programming
Implement the look_at_box_front() function in this python script
to capture an image of the box from its front side
"""

import os
import time
import numpy as np
import pybullet as p
import argparse
import scipy
import matplotlib.pyplot as plt
from pybullet_api import Fetch
from utils import *


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
        q = self.default_pose()
        self.robot.cmd(q)
        for _ in range(1000):
            p.stepSimulation()

        # Set table and plane
        self.object_uids = []
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


    def place_objects(self, urdf_filename, position, orientation):
        """
        Place of an object to the scene
        """
        uid = self._add_mesh(urdf_filename, position, orientation)  # xyzw
        self.object_uids.append(uid)
        p.resetBaseVelocity(uid, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for _ in range(100):
            p.stepSimulation()


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
        head_near = 0.01
        head_far = 100
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
                                                   
        # visualization
        fig = plt.figure()
        
        # show RGB image
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(rgba[:, :, :3])
        ax.set_title('RGB image')
        
        # show depth image
        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(depth)
        ax.set_title('depth image')
        
        # show segmentation mask
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(mask)
        ax.set_title('segmentation mask')                  
        plt.show()
    

    def default_pose(self):
        # set robot pose
        # [b'r_wheel_joint', b'l_wheel_joint', b'torso_lift_joint', b'head_pan_joint', b'head_tilt_joint', b'shoulder_pan_joint', 
        # b'shoulder_lift_joint', b'upperarm_roll_joint', b'elbow_flex_joint', b'forearm_roll_joint', b'wrist_flex_joint', 
        # b'wrist_roll_joint', b'r_gripper_finger_joint', b'l_gripper_finger_joint', b'bellows_joint']

        joint_command = np.zeros((self.robot.ndof, ), dtype=np.float32)
        # arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
        #              "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        # arm_joint_positions  = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]

        # raise torso
        joint_command[2] = 0.4
        # move head
        joint_command[3] = 0.009195
        joint_command[4] = 0.908270
        # move arm
        index = [5, 6, 7, 8, 9, 10, 11]
        joint_command[index] = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        return joint_command
    

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
        env.place_objects(filename, position, orientation)
        print(obj, position, orientation)    
    
    # place the cracker box to the table
    # name = '003_cracker_box'
    # env.place_objects(name)

    # render image before looking at the box
    env.get_observation()
