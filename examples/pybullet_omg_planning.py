import os
import time, datetime
import numpy as np
import argparse
import json
import _init_paths
from pybullet_scenereplica import SceneReplicaEnv
from mesh_to_sdf.depth_point_cloud import DepthPointCloud
from utils import *
from omg.core import *
from omg.util import *
from omg.config import cfg


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
    scene_type = args.scene_type

    # setup planner
    cfg.traj_init = "grasp"
    cfg.vis = False
    cfg.scene_file = ""
    cfg.timesteps = 50
    cfg.get_global_param(cfg.timesteps)
    cfg.vis = args.vis
    cfg.depth_threshold = 1.5
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
    if scene_type == 'shelf':
        cfg.use_point_sdf = True
        cfg.standoff_dist = 0.2
    else:
        cfg.use_point_sdf = False
    scene = PlanningScene(cfg)

    # create the table environment
    env = SceneReplicaEnv(data_dir, robot_name, scene_type)

    # load all objects
    for i, name in enumerate(env.ycb_object_names):
        trans, orn = env.cache_object_poses[i]
        scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)   
    scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    if scene_type == 'tabletop':
        scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
        scene.env.combine_sdfs()
        orderings = ["nearest_first", "random"]
    else:
        # use point cloud of the shelf
        rgba, depth, mask, cam_pose, intrinsic_matrix = env.get_observation()
        depth_pc = DepthPointCloud(depth, intrinsic_matrix, cam_pose, target_mask=None, threshold=cfg.depth_threshold)
        points = depth_pc.points
        scene.env.compute_sdf_from_points(points)
        orderings = ["random"]
    scene.reset()

    total_success = 0
    count = 0
    results_scene = {}
    if scene_id == -1:
        all_scene_ids = env.all_scene_ids
    else:
        all_scene_ids = [scene_id]
    for scene_id in all_scene_ids:
        print(f'=====================Scene {scene_id}========================')
        meta = env.setup_scene(scene_id)    

        # two orderings
        results_ordering = {}
        for ordering in orderings:
            object_order = meta[ordering][0].split(",")
            print(ordering, object_order)
            
            # for each object
            results = {}
            set_objects = set(object_order)
            for object_name in object_order:
                count += 1
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
                print('start IK')
                start = time.time()
                scene.reset(lazy=True)
                ik_time = time.time() - start
                print('IK time', ik_time)
                
                print('start plannnig')
                start = time.time()
                qc = env.robot.q()
                scene.traj.start = np.array(qc).astype(np.float32)
                info = scene.step()
                planning_time = time.time() - start
                print('plannnig time', planning_time)
                plan = scene.planner.history_trajectories[-1]
                plan = plan.T

                if args.vis:
                    scene.fast_debug_vis(interact=int(args.vis), nonstop=True)

                env.robot.execute_plan(plan)
                env.robot.close_gripper()
                env.record_gripper_position()
                time.sleep(1.0)

                # retrieve object
                if scene_type == 'tabletop':
                    env.retract()
                else:
                    standoff_offset = -10
                    plan_standoff = plan[:, np.arange(standoff_offset - 10, -1)]
                    plan_reverse = plan_standoff[:, ::-1]
                    plan_reverse[[7, 8], :] = 0
                    env.robot.execute_plan(plan_reverse)

                reward = env.compute_reward(object_name)
                print(f'scene: {scene_id}, order: {ordering}, object: {object_name}, reward: {reward}')
                # retract
                env.reset_objects(object_name)
                env.robot.retract()
                set_objects.remove(object_name)
                total_success += reward
                print(f'total reward: {total_success}/{count}')
                results[object_name] = {'reward': reward, 'plan': plan.tolist(), 'ik_time': ik_time, 'planning_time': planning_time}
            results_ordering[ordering] = results
        results_scene[f'{scene_id}'] = results_ordering

    print('total success', total_success)

    # write results
    outdir = "results"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    curr_time = datetime.datetime.now()
    exp_time = "{:%y-%m-%d_T%H%M%S}".format(curr_time)        
    filename = os.path.join(outdir, f'OMG_scenereplica_{robot_name}_{scene_type}_{exp_time}.json')
    with open(filename, "w") as outfile: 
        json.dump(results_scene, outfile)