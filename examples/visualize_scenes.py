import os
import sys
import argparse
import pathlib
import scipy
import numpy as np

cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
sys.path.append(os.path.join(cwd, ".."))
import optas
from optas.visualize import Visualizer


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


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="003_cracker_box",
        help="object model name",
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


if __name__ == "__main__":
    args = make_args()
    model = args.model
    data_dir = args.data_dir
    scene_id = args.scene_id
    model_dir = os.path.join(args.data_dir, "models")
    grasp_dir = os.path.join(args.data_dir, "grasp_data", "refined_grasps")
    scenes_path = os.path.join(args.data_dir, "final_scenes", "scene_data")

    vis = Visualizer(camera_position=[4, 4, 4])
    vis.grid_floor()

    # table model
    filename = os.path.join(cwd, 'objects', 'cafe_table', 'cafe_table.obj')
    z_offset = -0.03  # difference between Real World and table CAD model
    table_position = [0.8, 0, z_offset]    
    vis.obj(
        filename,
        png_texture_filename=None,
        position=table_position,
        orientation=[0, 0, 0],
        euler_degrees=True,
    )

    # load robot model
    urdf_filename = os.path.join(cwd, "robots", "fetch", "fetch.urdf")
    print(urdf_filename)
    robot_model = optas.RobotModel(urdf_filename=urdf_filename) 

    print(robot_model.actuated_joint_names)
    q = default_pose(robot_model)
    vis.robot(
        robot_model,
        base_position=[0.0, 0, 0],
        base_orientation=[0, 0, 0],
        euler_degrees=True,
        q=q,
    )

    # add objects
    print(f"-----------Scene: {scene_id}---------------")
    meta_f = "meta-%06d.mat" % scene_id
    meta = scipy.io.loadmat(os.path.join(data_dir, "final_scenes", "metadata", meta_f))
    meta_obj_names = meta["object_names"]
    meta_poses = {}
    for i, obj in enumerate(meta_obj_names):
        obj = obj.strip()
        filename = os.path.join(model_dir, obj, 'textured_simple.obj')
        texture_filename = os.path.join(model_dir, obj, 'texture_map.png')
        position = meta["poses"][i][:3]
        quat = meta["poses"][i][3:]
        # scalar-last (x, y, z, w) format in optas
        orientation = [quat[1], quat[2], quat[3], quat[0]]
        vis.obj(
            filename,
            png_texture_filename=texture_filename,
            position=position,
            orientation=orientation,
            euler_degrees=False,
        )
        print(obj, position, orientation)

    save = False
    if save:
        vis.save('vis.png')
    else:
        vis.start()