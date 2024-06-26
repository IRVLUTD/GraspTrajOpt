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

    if 'fetch' in robot_name:
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
    urdf_filename_gripper = os.path.join(root_dir, cfg['urdf_gripper_path'])
    gripper_model = GTORobotModel(robot_model_dir, urdf_filename=urdf_filename_gripper)

    # create the table environment
    env = SceneReplicaEnv(urdf_filename, data_dir, robot_name, scene_type)

    # Initialize planner
    print('Initialize planner')
    planner = GTOPlanner(robot, cfg['link_ee'], cfg['link_gripper'], 
                         standoff_distance=standoff_distance,
                         standoff_offset=standoff_offset)
    ik_solver = IKSolver(robot, cfg['link_ee'], cfg['link_gripper'], collision_avoidance=ik_collision_avoidance)   
    
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
                # if object_name != '040_large_marker':
                #     env.reset_objects(object_name)
                #     set_objects.remove(object_name)
                #     continue
                count += 1
                print(object_name)
                # reset scene
                env.reset_scene(set_objects)
                                            
                # render image and compute sdf cost field
                rgba, depth, mask, cam_pose, intrinsic_matrix = env.get_observation()
                idx = env.object_uids[env.object_names.index(object_name)]
                target_mask = mask == idx
                depth_pc = DepthPointCloud(depth, intrinsic_matrix, cam_pose, target_mask=None, threshold=cfg['depth_threshold'])
                robot.setup_points_field(depth_pc.points)
                world_points = robot.workspace_points
                sdf_cost_all = depth_pc.get_sdf_cost(world_points)

                # q0 = np.array(env.robot.q()).reshape((env.robot.ndof, 1))
                # visualize_pose(robot, env.base_position, q0, depth_pc)

                # compute sdf cost obstacle
                depth_obstacle = depth.copy()
                depth_obstacle[target_mask] = cfg['depth_threshold']
                depth_pc_obstacle = DepthPointCloud(depth_obstacle, intrinsic_matrix, cam_pose, target_mask, threshold=cfg['depth_threshold'])
                sdf_cost_obstacle = depth_pc_obstacle.get_sdf_cost(world_points)

                # load grasps
                RT_grasps = load_grasps(data_dir, robot_name, object_name)

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

                    # check if the grasp is in collision
                    RT_off = RT @ env.robot.get_standoff_pose(offset, cfg['axis_standoff'])
                    gripper_points, normals = gripper_model.compute_fk_surface_points(cfg['gripper_open_offsets'], tf_base=RT_off)
                    sdf = depth_pc_obstacle.get_sdf(gripper_points)

                    ratio = np.sum(sdf < 0) / len(sdf)
                    print(f'grasp {i}, collision ratio {ratio}')
                    if ratio > 0.01:
                        in_collision[i] = 1

                    # visualization
                    # colors = np.zeros(gripper_points.shape)
                    # colors[sdf < 0, 2] = 1
                    # colors[sdf > 0, 0] = 1
                    # cloud = pyrender.Mesh.from_points(gripper_points, colors=colors)
                    # scene = pyrender.Scene()
                    # scene.add(cloud)
                    # scene.add(pyrender.Mesh.from_points(depth_pc.points))
                    # pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
                
                RT_grasps_world = RT_grasps_world[in_collision == 0]
                checking_time = time.time() - start
                print('Checking grasp collision time', checking_time)
                print('Among %d grasps, %d in collision, %d collision-free' % (n, np.sum(in_collision), RT_grasps_world.shape[0]))
                if RT_grasps_world.shape[0] == 0:
                    set_objects.remove(object_name)
                    results[object_name] = {'reward': 0, 'plan': None, 'checking_time': checking_time,
                                         'ik_time': None, 'planning_time': None}
                    continue           

                # test IK for remaining grasps
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
                    q_solution, err_pos, err_rot, cost_collision = ik_solver.solve_ik(q0, RT_off, sdf_cost_obstacle, env.base_position)
                    q_solutions[:, i] = q_solution
                    if err_pos < 0.01 and err_rot < 5 and cost_collision < ik_collision_threshold:
                        found_ik[i] = 1
                        # if args.vis:
                        #     visualize_pose(robot, env.base_position, q_solution, depth_pc)

                RT_grasps_world = RT_grasps_world[found_ik == 1] 
                RT_grasps_base = RT_grasps_base[found_ik == 1]
                q_solutions = q_solutions[:, found_ik == 1]
                ik_time = time.time() - start
                print('IK time', ik_time)
                print('Among %d grasps, %d found IK' % (n, np.sum(found_ik)))
                print('IK solutions with shape', q_solutions.shape)
                if RT_grasps_world.shape[0] == 0:
                    set_objects.remove(object_name)
                    results[object_name] = {'reward': 0, 'plan': None, 'checking_time': checking_time,
                                         'ik_time': ik_time, 'planning_time': None}
                    continue

                # visualize grasp
                # model_dir = os.path.join(args.data_dir, "models")
                # RT = RT_grasps_world[0]
                # RT_off = RT @ env.robot.get_standoff_pose(standoff_distance, cfg['axis_standoff'])
                # visualize_standoff(cfg, model_dir, object_name, gripper_model, RT, RT_off, RT_obj)
                # visualize_grasp(cfg, robot, gripper_model, env.base_position, q0, depth_pc, RT_grasps_world[0])

                # plan to a grasp set
                qc = env.robot.q()
                print('start planning')
                start = time.time()
                plan, dQ, cost = planner.plan_goalset(qc, RT_grasps_base, sdf_cost_all, sdf_cost_obstacle, 
                                                      env.base_position, q_solutions, use_standoff=True, 
                                                      axis_standoff=cfg['axis_standoff'], interpolate=interpolate)
                planning_time = time.time() - start
                print('plannnig time', planning_time, 'cost', cost)
                
                if args.vis:
                    visualize_plan(robot, gripper_model, env.base_position, plan, depth_pc, RT_grasps_world)
                    debug_plan(robot, gripper_model, env.base_position, plan, depth_pc, sdf_cost_obstacle, RT_grasps_world, show_grasp=False)

                env.robot.execute_plan(plan)
                env.robot.close_gripper()
                env.record_gripper_position()
                time.sleep(1.0)

                # retrieve object
                if scene_type == 'tabletop':
                    env.retract(cfg['retract_distance'])
                else:
                    plan_standoff = plan[:, np.arange(standoff_offset - 10, -1)]
                    plan_reverse = plan_standoff[:, ::-1]
                    plan_reverse[cfg['finger_index'], :] = 0
                    env.robot.execute_plan(plan_reverse)
                reward = env.compute_reward(object_name)
                print(f'scene: {scene_id}, order: {ordering}, object: {object_name}, reward: {reward}')

                # retract robot
                env.reset_objects(object_name)
                env.robot.retract()
                set_objects.remove(object_name)
                total_success += reward
                print(f'total reward: {total_success}/{count}')
                results[object_name] = {'reward': reward, 'plan': plan.tolist(), 'checking_time': checking_time,
                                         'ik_time': ik_time, 'planning_time': planning_time}

            results_ordering[ordering] = results
        results_scene[f'{scene_id}'] = results_ordering                

    print('total success', total_success)
    # write results
    outdir = "results"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    curr_time = datetime.datetime.now()
    exp_time = "{:%y-%m-%d_T%H%M%S}".format(curr_time)        
    filename = os.path.join(outdir, f'GTO_scenereplica_{robot_name}_{scene_type}_{exp_time}.json')
    with open(filename, "w") as outfile: 
        json.dump(results_scene, outfile)