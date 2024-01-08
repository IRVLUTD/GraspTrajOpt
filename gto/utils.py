import json
import numpy as np
from transforms3d.quaternions import quat2mat


def default_pose(robot_model):
    # set robot pose
    # ['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 'shoulder_pan_joint', 
    # 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 
    # 'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']

    joint_command = np.zeros((robot_model.ndof, ), dtype=np.float32)

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