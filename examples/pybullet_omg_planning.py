import os, sys
import time
import numpy as np
import pybullet as p
import argparse
import json 
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
    cfg.cam_V = np.array(
        [
            [-0.9351, 0.3518, 0.0428, 0.3037],
            [0.2065, 0.639, -0.741, 0.132],
            [-0.2881, -0.684, -0.6702, 1.8803],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cfg.remove_flip_grasp = False
    cfg.remove_base_rotate_grasp = False
    cfg.remove_camera_downward_grasp = False
    cfg.augment_flip_grasp = False
    cfg.y_upsample = False
    cfg.z_upsample = False
    cfg.ik_parallel = False
    scene = PlanningScene(cfg)

    # create the table environment
    env = SceneReplicaEnv(data_dir, robot_name)

    # load all objects
    for i, name in enumerate(env.ycb_object_names):
        trans, orn = env.cache_object_poses[i]
        scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)   
    scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
    scene.env.combine_sdfs()        
    scene.reset()

    total_success = 0
    results_scene = {}
    for scene_id in env.all_scene_ids:
        print(f'=====================Scene {scene_id}========================')
        meta = env.setup_scene(scene_id)    

        # two orderings
        results_ordering = {}
        for ordering in ["nearest_first", "random"]:
            object_order = meta[ordering][0].split(",")
            print(ordering, object_order)
            
            # for each object
            results = {}
            set_objects = set(object_order)
            for object_name in object_order:
                print(object_name)
                # reset scene
                env.reset_scene(set_objects)

                # set scene
                for name in env.ycb_object_names:
                    trans, orn = env.get_object_pose(name)
                    pose = np.zeros((7, ))
                    pose[:3] = trans - env.base_position
                    pose[3:] = tf_quat(orn)
                    scene.env.update_pose(name, pose)                

                cfg.disable_collision_set = [
                    name
                    for name in enumerate(env.ycb_object_names)
                    if name not in set_objects
                ]            

                scene.env.set_target(object_name)
                scene.reset(lazy=True)
                info = scene.step()
                plan = scene.planner.history_trajectories[-1]

                if args.vis:
                    scene.fast_debug_vis(interact=int(args.vis), nonstop=True)

                env.robot.execute_plan(plan.T)
                env.robot.close_gripper()
                time.sleep(1.0)
                env.retract()
                reward = env.compute_reward(object_name)
                print('reward:', reward)
                # retract
                env.reset_objects(object_name)
                env.robot.retract()
                set_objects.remove(object_name)                
                total_success += reward
                results[object_name] = reward
            results_ordering[ordering] = results
        results_scene[f'{scene_id}'] = results_ordering

    print('total success', total_success)

    # write results
    outdir = "results"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filename = 'OMG_scenereplica.json'
    with open(filename, "w") as outfile: 
        json.dump(results_scene, outfile)