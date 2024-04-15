import os
import sys
import argparse
import pathlib
import scipy
import numpy as np
from utils import *

cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
sys.path.append(os.path.join(cwd, ".."))
import optas
from optas.visualize import Visualizer
from gto.gto_models import GTORobotModel
from gto.utils import *
from pybullet_scenereplica import load_grasps
from transforms3d.quaternions import quat2mat, mat2quat


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


if __name__ == "__main__":
    args = make_args()
    data_dir = args.data_dir
    scene_id = args.scene_id
    robot_name = args.robot
    model_dir = os.path.join(args.data_dir, "objects")
    grasp_dir = os.path.join(args.data_dir, "grasp_data", "refined_grasps")
    scenes_path = os.path.join(args.data_dir, "final_scenes", "scene_data")

    vis = Visualizer(camera_position=[4, 4, 4])
    vis.grid_floor()

    # table model
    filename = os.path.join(cwd, '../data/objects', 'cafe_table', 'cafe_table.obj')
    z_offset = -0.03  # difference between Real World and table CAD model
    table_position = [0.8, 0, z_offset]    
    vis.obj(
        filename,
        png_texture_filename=None,
        position=table_position,
        orientation=[0, 0, 0],
        euler_degrees=True,
    )

    # load config file
    root_dir = get_root_dir()
    config_file = os.path.join(root_dir, 'data', 'configs', f'{robot_name}.yaml')
    if not os.path.exists(config_file):
        print(f'robot {robot_name} not supported', config_file)
        sys.exit(1) 
    cfg = load_yaml(config_file)['robot_cfg']
    print(cfg)    
    
    # load robot model
    robot_model_dir = os.path.join(root_dir, 'data', 'robots', cfg['robot_name'])
    urdf_filename = os.path.join(root_dir, cfg['urdf_robot_path']) 
    robot_model = GTORobotModel(robot_model_dir,
                          urdf_filename=urdf_filename, 
                          time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
                          param_joints=cfg['param_joints'],
                          collision_link_names=cfg['collision_link_names'])
    
    # load robot gripper model
    urdf_filename_gripper = os.path.join(root_dir, cfg['urdf_gripper_path'])
    gripper_model = GTORobotModel(robot_model_dir, urdf_filename=urdf_filename_gripper)    

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

        # show grasp
        RT_grasps = load_grasps(data_dir, robot_name, obj)
        RT_obj = np.eye(4)
        RT_obj[:3, :3] = quat2mat(quat)
        RT_obj[:3, 3] = position
        RT_grasps_world = np.matmul(RT_obj, RT_grasps)
        n = RT_grasps_world.shape[0]
        index = np.random.choice(n, 3)

        q = [0.05, 0.05]
        for j in index:
            RT = RT_grasps_world[j]
            position = RT[:3, 3]
            # scalar-last (x, y, z, w) format in optas
            quat = mat2quat(RT[:3, :3])
            orientation = [quat[1], quat[2], quat[3], quat[0]]
            vis.robot(
                gripper_model,
                base_position=position,
                base_orientation=orientation,
                q=q,
                alpha = 0.3,
            )          

    save = False
    if save:
        vis.save('vis.png')
    else:
        vis.start()