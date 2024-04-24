import os, sys
import cv2
import numpy as np
import pybullet as p
import argparse
import json
import time
from pybullet_scenereplica import SceneReplicaEnv
from pybullet_api import pybullet_show_frame
from transforms3d.quaternions import mat2quat
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
    parser.add_argument("-m", "--mobile", help="mobile manipulation", action="store_true")
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
    is_mobile = args.mobile
    assert robot_name in filename, f"result file {filename} is not for robot {robot_name}"
    assert scene_type in filename, f"result file {filename} is not for scene type {scene_type}"
    if is_mobile:
        assert 'mobile' in filename, f"result file {filename} is not for mobile manipulation"    

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
    env = SceneReplicaEnv(urdf_filename, data_dir, robot_name, scene_type, mobile=is_mobile)
    dyn = p.getDynamicsInfo(env.robot._id, -1)
    mass = dyn[0]
    print('base link mass', mass)    

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
            print(f'=====================Scene {scene_id}========================')
            meta = env.setup_scene(scene_id)

            object_order = meta[ordering][0].split(",")
            print(ordering, object_order)
            
            # for each object
            results = results_ordering[ordering]

            if is_mobile:
                # update camera view
                if env.scene_type == 'tabletop':
                    env.distance = 4.0
                else:
                    env.distance = 4.5
                env.update_view_matrix()

                # move robot base
                # get robot base pose
                pos, orn = env.get_robot_pose()
                RT_base = np.eye(4)
                quat = [orn[3], orn[0], orn[1], orn[2]]   #w, x, y, z
                RT_base[:3, :3] = quat2mat(quat)
                RT_base[:3, 3] = pos

                # new base
                RT_base_new = np.array(results['RT_base_new']).reshape((4, 4))

                # RT_base_new = RT_base @ RT_base_delta
                RT_base_delta = np.linalg.inv(RT_base) @ RT_base_new
                pybullet_show_frame(RT_base_new)

                # move base
                x_delta = RT_base_delta[0, 3]
                y_delta = RT_base_delta[1, 3]
                env.robot.move_to_xy(x_delta, y_delta, env, video_writer)

                # get new robot base pose
                pos, orn = env.get_robot_pose()
                RT_base = np.eye(4)
                quat = [orn[3], orn[0], orn[1], orn[2]]   #w, x, y, z
                RT_base[:3, :3] = quat2mat(quat)
                RT_base[:3, 3] = pos
                delta = np.linalg.inv(RT_base) @ RT_base_new
                quat = mat2quat(delta[:3, :3])
                orn = [quat[1], quat[2], quat[3], quat[0]]
                euler = p.getEulerFromQuaternion(orn)
                yaw = euler[2]            

                # rotate robot
                env.robot.move_to_theta(yaw, env, video_writer)

                # fix base
                pos, orn = env.get_robot_pose()
                RT_base = np.eye(4)
                quat = [orn[3], orn[0], orn[1], orn[2]]   #w, x, y, z
                RT_base[:3, :3] = quat2mat(quat)
                RT_base[:3, 3] = pos            
                RT_base_new = RT_base.copy()
                env.set_robot_pose(pos, orn)
                p.changeDynamics(env.robot._id, -1, mass=0)

                # update camera view
                if env.scene_type == 'tabletop':
                    env.distance = 1.8
                else:
                    env.distance = 1.6
                env.update_view_matrix()

            set_objects = set(object_order)
            for object_name in object_order:
                # reset scene
                env.reset_scene(set_objects)

                if is_mobile:
                    # query object pose in world
                    pos, orn = env.get_object_pose(object_name)
                    obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                    RT_obj = unpack_pose(obj_pose)      
                    # camera look at object
                    env.robot.look_at_point(RT_obj[:3, 3])                
                                            
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
                    env.execute_plan(plan_reverse, video_writer)
                
                # retract robot
                set_objects.remove(object_name)
                env.reset_objects(object_name)
                env.robot.retract()

            if is_mobile:
                p.changeDynamics(env.robot._id, -1, mass=mass)

        video_writer.release()