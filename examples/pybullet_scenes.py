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
import matplotlib.pyplot as plt
from pybullet_api import Fetch

class SceneReplicaEnv():

    def __init__(self):

        self._renders = True
        self._egl_render = False
        self.connected = False

        self._window_width = 640
        self._window_height = 480
        self.object_uid = None
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
            p.resetDebugVisualizerCamera(0.6, 180.0, -41.0, [1.0, 2.0, 2.0])
        else:
            self.cid = p.connect(p.DIRECT)

        if self._egl_render:
            import pkgutil
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.connected = True


    def reset(self):

        # Set the camera.
        look = [0.1, 0.2, 0]
        distance = 2.5
        pitch = -56
        yaw = 245
        roll = 0.
        fov = 20.
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
        self._light_position = np.array([-1.0, 0, 2.5])

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
        plane_file = os.path.join(self.root_dir, 'objects/floor/model_normalized.urdf') # _white
        table_file = os.path.join(self.root_dir, 'objects/cafe_table/cafe_table.urdf')

        self.obj_path = [plane_file, table_file]
        self.plane_id = p.loadURDF(plane_file, [0, 0, 0])
        z_offset = -0.03  # difference between Real World and table CAD model
        self.table_pos = np.array([0.8, 0, z_offset])
        self.table_id = p.loadURDF(table_file, self.table_pos)


    def _add_mesh(self, obj_file, trans, quat, scale=1):
        """
        Add a mesh with URDF file.
        """
        bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        return bid


    def place_objects(self, name):
        """
        Place of an object onto the table
        """
        
        # 3D translation is [tx, ty, tz]
        tx = 0
        ty = 0
        tz = 0.4

        # euler angles: roll, pitch, yaw
        # then convert roll, pitch, yaw to quaternion using function p.getQuaternionFromEuler()        
        roll = 0
        pitch = 0
        yaw = 0
        quaternion = p.getQuaternionFromEuler([roll, pitch, yaw])
        
        # put the box using the 3D translation and the 3D rotation
        urdf = os.path.join(self.root_dir, 'data', name, 'model_normalized.urdf')
        uid = self._add_mesh(urdf, [tx, ty, tz], [quaternion[0], quaternion[1], quaternion[2], quaternion[3]])  # xyzw
        self.object_uid = uid
        p.resetBaseVelocity(uid, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        time.sleep(3.0)
        for _ in range(2000):
            p.stepSimulation()



    def get_observation(self, view_matrix):
        """
        Get observation and visualize
        """

        _, _, rgba, depth, mask = p.getCameraImage(width=self._window_width,
                                                   height=self._window_height,
                                                   viewMatrix=view_matrix,
                                                   projectionMatrix=self._proj_matrix,
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
                                                   
                                                   
    def look_at_box_front(self):
    
        # query the pose of the box
        pos, orn = p.getBasePositionAndOrientation(self.object_uid)
        print(pos, orn)
        
        # https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html
        # The X,Y,Z Euler angles are in radians, accumulating 3 rotations expressing the roll around the X, pitch around Y and yaw around the Z axis.
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)  # roll, pitch, yaw
        print(roll, pitch, yaw)
        
        # rotate the camera view
        yaw += np.pi/2
        print(yaw)
        
        # define the view matrix accordingly
        look = pos
        distance = 2.5
        view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw*180 /np.pi, pitch*180 /np.pi, roll*180 /np.pi, 2)       
        
        return view_matrix
    

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
    
    # place the cracker box to the table
    # name = '003_cracker_box'
    # env.place_objects(name)

    # render image before looking at the box
    env.get_observation(env._view_matrix)
        
    # look at the box
    view_matrix = env.look_at_box_front()
    
    # render image again
    env.get_observation(view_matrix)
