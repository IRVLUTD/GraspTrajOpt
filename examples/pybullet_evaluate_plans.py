import os, sys
import numpy as np
import pybullet as p
import argparse
import json
from pybullet_scenereplica import SceneReplicaEnv
from utils import *
import _init_paths
from mesh_to_sdf.depth_point_cloud import DepthPointCloud
from gto.gto_models import GTORobotModel
from gto.utils import load_yaml, get_root_dir
    


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
    elif scene_type == 'shelf':
        orderings = ["random"]
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
    
    # intialization
    total_success = 0
    object_success = {}
    object_count = {}
    for obj in env.ycb_object_names:
        object_success[obj] = 0
        object_count[obj] = 0
    total_collision = 0

    # main loop
    all_scene_ids = env.all_scene_ids
    for scene_id in all_scene_ids:
        print(f'=====================Scene {scene_id}========================')
        meta = env.setup_scene(scene_id)

        # two orderings
        results_ordering = results_scene[f'{scene_id}']
        for ordering in orderings:
            object_order = meta[ordering][0].split(",")
            print(ordering, object_order)
            
            # for each object
            results = results_ordering[ordering]
            set_objects = set(object_order)
            for object_name in object_order:
                reward = results[object_name]['reward']
                plan = results[object_name]['plan']
                total_success += reward
                object_success[object_name] += reward
                object_count[object_name] += 1
                
                # reset scene
                env.reset_scene(set_objects)
                                            
                # render image and compute sdf cost field
                rgba, depth, mask, cam_pose, intrinsic_matrix = env.get_observation()
                idx = env.object_uids[env.object_names.index(object_name)]
                target_mask = mask == idx
                depth[target_mask] = 2.0
                depth_pc = DepthPointCloud(depth, intrinsic_matrix, cam_pose, target_mask)

                # check if the robot plan is in collision
                in_collision = False
                if plan is not None:
                    plan = np.array(plan)
                    for i in range(plan.shape[1]):
                        q = plan[:, i]
                        points_base, _ = robot.compute_fk_surface_points(q)
                        points_world = points_base + env.base_position.reshape((1, 3))
                        sdf = depth_pc.get_sdf(points_world)
                        # at least 5 body points in collision
                        if np.sum(sdf < 0) > 5:
                            in_collision = True
                            break
                total_collision += int(in_collision)
                print('--------------------------------')
                print(f'{object_name}, success {reward}, collision {int(in_collision)}')
                set_objects.remove(object_name)
                env.reset_objects(object_name)

    # print output
    print(f'-----------------{scene_type} scenes------------------')
    count = 0
    for obj in env.ycb_object_names:
        print(f'{obj}, success {object_success[obj]}, total {object_count[obj]}')
        count += object_count[obj]
    print('-----------------------------------')
    print('total success', total_success)
    print('total collision', total_collision)
    print('total trial', count)