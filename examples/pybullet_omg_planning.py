import os, sys
import time
import numpy as np
import pybullet as p
import argparse
import scipy
import matplotlib.pyplot as plt
from pybullet_api import Fetch, Panda
from pybullet_scenereplica import SceneReplicaEnv
from utils import *
import _init_paths
from omg.core import *
from omg.util import *
from omg.config import cfg
from transforms3d.quaternions import mat2quat

import pathlib
cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
    

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
    parser = argparse.ArgumentParser()
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
        "-s",
        "--scene_id",
        type=int,
        default=10,
        help="SceneReplica scene id",
    )
    parser.add_argument("-v", "--vis", help="renders", action="store_true")
    parser.add_argument("-w", "--write_video", help="write video", action="store_true")
    parser.add_argument("-exp", "--experiment", help="loop through the 100 scenes", action="store_true")
    parser.add_argument("--egl", help="use egl render", action="store_true")      
    args = parser.parse_args()
    return args
        

# main function
if __name__ == "__main__":
    args = make_args()
    robot_name = args.robot
    data_dir = args.data_dir
    scene_id = args.scene_id    

    # setup planner
    cfg.traj_init = "grasp"
    cfg.vis = False
    cfg.scene_file = ""
    cfg.timesteps = 50
    cfg.get_global_param(cfg.timesteps)
    cfg.vis = args.vis
    scene = PlanningScene(cfg)

    # create the table environment
    env = SceneReplicaEnv(data_dir, robot_name)    

    # load all objects
    for i, name in enumerate(env.ycb_object_names):
        trans, orn = env.cache_object_poses[i]
        scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)   

    scene.reset()
    scene.fast_debug_vis(interact=int(args.vis), traj_type=0)
    input('next?')
    sys.exit(1)         


    meta = env.setup_scene(scene_id)

    for name in env.object_names:
        trans, orn = env.meta_poses[name]
        scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)



    scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
    scene.env.combine_sdfs()
    if args.experiment:   
        scene_files = ['scene_{}'.format(i) for i in range(100)]
    else:
        scene_files = [scene_file]

    cnts, rews = 0, 0
    for scene_file in scene_files:
        config.cfg.output_video_name = "output_videos/bullet_" + scene_file + ".avi"
        cfg.scene_file = scene_file
        video_writer = None
        if args.write_video:
            video_writer = cv2.VideoWriter(
                config.cfg.output_video_name,
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (640, 480),
            )
        full_name = os.path.join('data/scenes', scene_file + ".mat")
        env.cache_reset(scene_file=full_name)
        obj_names, obj_poses = env.get_env_info()
        object_lists = [name.split("/")[-1].strip() for name in obj_names]
        object_poses = [pack_pose(pose) for pose in obj_poses]

        exists_ids, placed_poses = [], []
        for i, name in enumerate(object_lists[:-2]):  # update planning scene
            scene.env.update_pose(name, object_poses[i])
            obj_idx = env.obj_path[:-2].index("data/objects/" + name)
            exists_ids.append(obj_idx)
            trans, orn = env.cache_object_poses[obj_idx]
            placed_poses.append(np.hstack([trans, ros_quat(orn)]))

        cfg.disable_collision_set = [
            name.split("/")[-2]
            for obj_idx, name in enumerate(env.obj_path[:-2])
            if obj_idx not in exists_ids
        ]
        scene.env.set_target(env.obj_path[env.target_idx].split("/")[-1])
        scene.reset(lazy=True)
        info = scene.step()
        plan = scene.planner.history_trajectories[-1]

        rew = bullet_execute_plan(env, plan, args.write_video, video_writer)
        for i, name in enumerate(object_lists[:-2]):  # reset planner
            scene.env.update_pose(name, placed_poses[i])
        cnts += 1
        rews += rew
        print('rewards: {} counts: {}'.format(rews, cnts))

    env.disconnect()
