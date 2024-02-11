import os, sys
import time, datetime
import numpy as np
import pybullet as p
import argparse
import json
import matplotlib.pyplot as plt
from pybullet_api import Fetch, Panda
from pybullet_scenereplica import SceneReplicaEnv
from utils import *
import _init_paths
from mesh_to_sdf.depth_point_cloud import DepthPointCloud
from optas.visualize import Visualizer
from gto.gto_models import GTORobotModel
from gto.gto_planner import GTOPlanner
from gto.ik_solver import IKSolver
from gto.utils import *
from transforms3d.quaternions import mat2quat
    

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
    parser.add_argument("-v", "--vis", help="renders", action="store_true")
    args = parser.parse_args()
    return args
        

# main function
if __name__ == '__main__':
    args = make_args()
    robot_name = args.robot
    data_dir = args.data_dir
    scene_id = args.scene_id
    scene_type = args.scene_type
    if scene_type == 'tabletop':
        standoff_distance = -0.1
        standoff_offset = -10
        ik_collision_avoidance = False
        ik_collision_threshold = 5
        interpolate = True
        orderings = ["nearest_first", "random"]
    elif scene_type == 'shelf':
        standoff_distance = -0.2
        standoff_offset = -10
        ik_collision_avoidance = False
        ik_collision_threshold = 0.001
        interpolate = False
        orderings = ["random"]
    else:
        print('unsupported scene type:', scene_type)
        sys.exit(1)
    # define the standoff pose for collision checking
    offset = -0.01    

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

    # load robot gripper model
    urdf_filename = os.path.join(robot_model_dir, f"{robot_name}_gripper.urdf")
    gripper_model = GTORobotModel(robot_model_dir, urdf_filename=urdf_filename)

    # create the table environment
    env = SceneReplicaEnv(data_dir, robot_name, scene_type)

    # load all grasps
    RT_grasps_all = {}
    for name in env.ycb_object_names:
        print(f'loading grasps for {name}')
        RT_grasps_all[name] = load_grasps(data_dir, robot_name, name)

    # Initialize IK solver
    ik_solver = IKSolver(robot, cfg['link_ee'], cfg['link_gripper'], collision_avoidance=ik_collision_avoidance)   
    
    total_success = 0
    count = 0
    if scene_id == -1:
        all_scene_ids = env.all_scene_ids
    else:
        all_scene_ids = [scene_id]
    for scene_id in all_scene_ids:
        print(f'=====================Scene {scene_id}========================')
        meta = env.setup_scene(scene_id)

        # two orderings
        for ordering in orderings:
            object_order = meta[ordering][0].split(",")
            print(ordering, object_order)
            
            # for each object
            set_objects = set(object_order)
            for object_name in object_order:
                print(object_name)
                # reset scene
                env.reset_scene(set_objects)

                # load grasps
                RT_grasps = RT_grasps_all[object_name]

                # query object pose
                pos, orn = env.get_object_pose(object_name)
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                RT_obj = unpack_pose(obj_pose)
                print(object_name, RT_obj)

                # transform grasps to robot base
                print('start checking collision of grasps')
                start = time.time()
                n = RT_grasps.shape[0]
                RT_grasps_world = np.zeros_like(RT_grasps)
                in_collision = np.zeros((n, ), dtype=np.int32)
                for i in range(n):
                    RT_g = RT_grasps[i]
                    RT = RT_obj @ RT_g
                    RT_grasps_world[i] = RT

                # test IK
                print('start IK')
                start = time.time()
                ik_solver.setup_optimization()
                n = RT_grasps_world.shape[0]
                RT_grasps_base = RT_grasps_world.copy()
                found_ik = np.zeros((n, ), dtype=np.int32)
                q0 = np.array(env.robot.q()).reshape((env.robot.ndof, 1))
                q_solutions = np.zeros((robot.ndof, n), dtype=np.float32)
                for i in range(n):
                    RT = RT_grasps_world[i].copy()
                    # change world to robot base
                    RT[:3, 3] -= env.base_position
                    RT_grasps_base[i] = RT.copy()
                    if scene_type == 'shelf':
                        RT_off = RT @ env.robot.get_standoff_pose(standoff_distance, cfg['axis_standoff'])
                    else:
                        RT_off = RT
                    q_solution, err_pos, err_rot, cost_collision = ik_solver.solve_ik(q0, RT_off, None, env.base_position)
                    q_solutions[:, i] = q_solution
                    if err_pos < 0.01 and err_rot < 5 and cost_collision < ik_collision_threshold:
                        found_ik[i] = 1
                        
                total_success += np.sum(found_ik)
                count += n

    print(scene_type)
    print(robot_name)
    print(f'all trials {count}, total success {total_success}')