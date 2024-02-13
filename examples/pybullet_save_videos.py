import os, sys
import cv2
import numpy as np
import pybullet as p
import argparse
import json
import time
from pybullet_scenereplica import SceneReplicaEnv
from utils import *
import _init_paths
from gto.gto_models import GTORobotModel
from gto.utils import *


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
        "-s",
        "--scene_id",
        type=int,
        default=-1,
        help="SceneReplica scene id",
    )
    parser.add_argument(
        "-t",
        "--scene_type",
        type=str,
        default="tabletop",
        help="tabletop or shelf",
    )      
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="results/GTO_scenereplica_panda_24-01-22_T175059.json",
        help="SceneReplica scene id",
    )
    parser.add_argument("-v", "--vis", help="renders", action="store_true")
    args = parser.parse_args()
    return args
        

# main function
if __name__ == '__main__':
    args = make_args()
    robot_name = args.robot
    data_dir = args.data_dir
    scene_id = args.scene_id
    filename = args.file
    scene_type = args.scene_type
    assert robot_name in filename, f"result file {filename} is not for robot {robot_name}"
    assert scene_type in filename, f"result file {filename} is not for scene type {scene_type}"

    if scene_type == 'tabletop':
        orderings = ["nearest_first", "random"]
        standoff_offset = -10
    elif scene_type == 'shelf':
        orderings = ["random"]
        standoff_offset = -10
    else:
        print('unsupported scene type:', scene_type)
        sys.exit(1)

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
    robot = GTORobotModel(robot_model_dir,
                          urdf_filename=urdf_filename, 
                          time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
                          param_joints=cfg['param_joints'],
                          collision_link_names=cfg['collision_link_names'])

    # create the table environment
    env = SceneReplicaEnv(data_dir, robot_name, scene_type)

    # load results
    with open(filename, "r") as outfile: 
        print(filename)
        results_scene = json.load(outfile)

    outdir = "videos"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # main loop
    all_scene_ids = env.all_scene_ids
    for scene_id in all_scene_ids:
        print(f'=====================Scene {scene_id}========================')
        meta = env.setup_scene(scene_id)

        output_video_name = f"videos/{os.path.basename(filename)}_{scene_id}.mp4"
        print('write video to', output_video_name)
        video_writer = cv2.VideoWriter(
            output_video_name,
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (640, 480),
        )        

        # two orderings
        results_ordering = results_scene[f'{scene_id}']
        for ordering in orderings:
            object_order = meta[ordering][0].split(",")
            print(ordering, object_order)
            
            # for each object
            results = results_ordering[ordering]
            set_objects = set(object_order)
            for object_name in object_order:
                # reset scene
                env.reset_scene(set_objects)
                                            
                plan = results[object_name]['plan']
                if plan is None:
                    continue

                plan = np.array(plan)
                env.execute_plan(plan, video_writer)
                env.robot.close_gripper()
                time.sleep(1.0)

                # retrieve object
                if scene_type == 'tabletop':
                    env.retract(cfg['retract_distance'], video_writer=video_writer)
                else:
                    plan_standoff = plan[:, np.arange(standoff_offset - 10, -1)]
                    plan_reverse = plan_standoff[:, ::-1]
                    plan_reverse[cfg['finger_index'], :] = 0
                    env.robot.execute_plan(plan_reverse, video_writer)
                
                # retract robot
                set_objects.remove(object_name)
                env.reset_objects(object_name)
                env.robot.retract()
        video_writer.release()