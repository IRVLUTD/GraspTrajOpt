import os
import sys
import argparse
import pathlib
import numpy as np

cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
import _init_paths
import optas
from optas.visualize import Visualizer
from utils import parse_grasps, rotZ, ycb_special_case
from transforms3d.quaternions import mat2quat


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_args()
    robot_name = args.robot
    model = args.model
    data_dir = args.data_dir

    model_dir = os.path.join(args.data_dir, "models")
    filename = os.path.join(model_dir, model, 'textured_simple.obj')
    texture_filename = os.path.join(model_dir, model, 'texture_map.png')    

    if robot_name == 'fetch':
        # parse grasps
        grasp_dir = os.path.join(args.data_dir, "grasp_data", "refined_grasps")
        grasp_file = os.path.join(grasp_dir, f"fetch_gripper-{model}.json")
        RT_grasps = parse_grasps(grasp_file)
    elif robot_name == 'panda':
        grasp_dir = os.path.join(args.data_dir, "grasp_data", "panda_simulated")
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
            RT_grasps = ycb_special_case(RT_grasps, model)
        offset_pose = np.array(rotZ(np.pi / 2))  # and
        RT_grasps = np.matmul(RT_grasps, offset_pose)  # flip x, y

    # load robot gripper model
    urdf_filename = os.path.join(cwd, "robots", robot_name, f"{robot_name}_gripper.urdf")
    print(urdf_filename)
    robot_model = optas.RobotModel(urdf_filename=urdf_filename)

    vis = Visualizer(camera_position=[1, 1, 1])
    vis.grid_floor()
    vis.obj(
        filename,
        png_texture_filename=texture_filename,
        position=[0, 0, 0],
        orientation=[0, 0, 0],
        euler_degrees=True,
    )

    print(robot_model)
    print(robot_model.link_names)

    # visualize grasps
    q = [0.05, 0.05]
    n = RT_grasps.shape[0]
    for i in np.random.permutation(n)[:5]:
        RT_g = RT_grasps[i]
        print(RT_g)
        position = RT_g[:3, 3]
        # scalar-last (x, y, z, w) format in optas
        quat = mat2quat(RT_g[:3, :3])
        orientation = [quat[1], quat[2], quat[3], quat[0]]
        
        vis.robot(
            robot_model,
            base_position=position,
            base_orientation=orientation,
            q=q
        )

    save = False
    if save:
        vis.save('vis.png')
    else:
        vis.start()