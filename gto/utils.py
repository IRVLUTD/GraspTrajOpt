import os
import json
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
from scipy import interpolate
import yaml
from optas.visualize import Visualizer


def get_root_dir():
    this_dir = os.path.dirname(__file__)
    return os.path.join(this_dir, '..')


def load_yaml(file_path):
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=yaml.Loader)
    else:
        yaml_params = file_path
    return yaml_params


def default_pose(robot_model):
    # set robot pose
    # ['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 'shoulder_pan_joint', 
    # 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 
    # 'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']

    joint_command = np.zeros((robot_model.ndof, ), dtype=np.float32)
    if robot_model.name == 'fetch':

        # arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
        #              "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        # arm_joint_positions  = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]

        # raise torso
        joint_command[2] = 0.4
        # move head
        joint_command[3] = 0.009195
        joint_command[4] = 0.908270
        # move arm
        index = [5, 6, 7, 8, 9, 10, 11]
        joint_command[index] = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
    elif robot_model.name == 'panda':
        joint_command = np.array([0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0])

    return joint_command


def interpolate_waypoints(waypoints, n, m, mode="cubic"):  # linear
    """
    Interpolate the waypoints using interpolation.
    """
    data = np.zeros([n, m])
    x = np.linspace(0, 1, waypoints.shape[0])
    for i in range(waypoints.shape[1]):
        y = waypoints[:, i]

        t = np.linspace(0, 1, n + 2)
        if mode == "linear":  # clamped cubic spline
            f = interpolate.interp1d(x, y, "linear")
        if mode == "cubic":  # clamped cubic spline
            f = interpolate.CubicSpline(x, y, bc_type="clamped")
        elif mode == "quintic":  # seems overkill to me
            pass
        data[:, i] = f(t[1:-1])  #
        # plt.plot(x, y, 'o', t[1:-1], data[:, i], '-') #  f(np.linspace(0, 1, 5 * n+2))
        # plt.show()
    return data


def debug_plan(robot, gripper_model, base_position, plan, depth_pc, sdf_cost, RT_grasps_world, show_grasp=True):
    T = plan.shape[1]
    for i in range(30, T):
        q = plan[:, i]
        points_base, _ = robot.compute_fk_surface_points(q)
        points_world = points_base + base_position.reshape(1, 3)
        offset = robot.points_to_offsets_numpy(points_world).astype(int)
        cost = np.sum(sdf_cost[offset])
        print(f'time step {i}, sdf cost {cost}')

        workspace_points = robot.workspace_points
        vis = Visualizer(camera_position=[3, 0, 3])
        vis.grid_floor()
        vis.points(depth_pc.points, rgb=[1, 1, 1])
        index = sdf_cost[offset] > 0
        vis.points(points_world[~index], rgb=[0, 1, 1], size=5)
        vis.points(points_world[index], rgb=[1, 0, 0], size=5)
        index = sdf_cost > 0
        vis.points(workspace_points[index], rgb=[1, 1, 0], size=3)
        # vis.points(workspace_points[index], rgb=[0, 1, 0], size=10)        
        vis.robot(
            robot,
            base_position=base_position,
            q=q,
            alpha = 1,
        )
        if show_grasp:
            q = [0, 0]
            for i in range(RT_grasps_world.shape[0]):
                RT = RT_grasps_world[i]
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
        vis.start()  


def visualize_plan(robot, gripper_model, base_position, plan, depth_pc, depth_pc_obstacle, RT_grasps_world):
    # visualize grasps
    vis = Visualizer(camera_position=[-1, 3.0, 5.0])
    vis.grid_floor()
    vis.points(
        depth_pc_obstacle.points,
        rgb=[0, 1, 0],
        size=5,
    )
    # q = [0, 0]
    # for i in range(RT_grasps_world.shape[0]):
    #     RT = RT_grasps_world[i]
    #     position = RT[:3, 3]
    #     # scalar-last (x, y, z, w) format in optas
    #     quat = mat2quat(RT[:3, :3])
    #     orientation = [quat[1], quat[2], quat[3], quat[0]]
    #     vis.robot(
    #         gripper_model,
    #         base_position=position,
    #         base_orientation=orientation,
    #         q=q,
    #         alpha = 0.1,
    #     )

    points_base_all = np.zeros((0, 3), dtype=np.float32)
    for i in range(plan.shape[1]):
        q = plan[:, i]
        points_base, _ = robot.compute_fk_surface_points(q)
        sdf = depth_pc_obstacle.get_sdf(points_base)
        index = np.where(sdf < 0)[0]
        if len(index) > 0:
            points_base = points_base[index, :]
            points_base_all = np.concatenate((points_base_all, points_base), axis=0)

    vis.points(
        points_base_all,
        rgb = [1, 0, 0],
        size=3,
        )

    # robot trajectory
    # sample plan
    n = plan.shape[1]
    index = list(range(0, n, 10))
    if index[-1] != n - 1:
        index += [n - 1]
    vis.robot_traj(robot, plan[:, index], base_position, alpha_spec={'style': 'D'})

    vis.start()


def visualize_pose(robot, base_position, q, depth_pc):
    # visualize grasps
    vis = Visualizer(camera_position=[-2, 3.0, 6.0])
    # vis.grid_floor()
    vis.points(
        depth_pc.points,
        rgb=[0, 1, 0],
    )
    workspace_points = robot.workspace_points
    sdf_cost = depth_pc.get_sdf(workspace_points)
    index = sdf_cost > 0
    vis.points(workspace_points[~index], rgb=[0, 1, 1])
    vis.points(workspace_points[index], rgb=[1, 1, 0]) # robot
    vis.robot(
        robot,
        base_position=base_position,
        q=q,
    )
    vis.start()


def visualize_grasp(cfg, robot, gripper_model, base_position, q, depth_pc, RT):
    # visualize grasps
    vis = Visualizer(camera_position=[-2, 3.0, 6.0])
    # vis.grid_floor()
    vis.points(
        depth_pc.points,
        rgb=[0, 1, 0],
    )
    # grasp
    position = RT[:3, 3]
    # scalar-last (x, y, z, w) format in optas
    quat = mat2quat(RT[:3, :3])
    orientation = [quat[1], quat[2], quat[3], quat[0]]
    vis.robot(
        gripper_model,
        base_position=position,
        base_orientation=orientation,
        q=cfg['gripper_open_offsets'],
    )
    # show points on the gripper
    points_goal = gripper_model.compute_fk_link_surface_points(cfg['gripper_open_offsets'], cfg['link_gripper'], RT)
    vis.points(points_goal, rgb=[1, 0, 0], size=5)
    vis.robot(
        robot,
        base_position=base_position,
        q=q,
    )
    tf_base = np.eye(4, dtype=np.float32)
    tf_base[:3, 3] = base_position
    points_gripper = robot.compute_fk_link_surface_points(q, cfg['link_gripper'], tf_base)
    vis.points(points_gripper, rgb=[1, 0, 0], size=5)
    vis.start()


def visualize_standoff(cfg, model_dir, object_name, gripper_model, RT_grasp, RT_off, RT_obj):
    # visualize grasps
    vis = Visualizer(camera_position=[-2, 3.0, 6.0])
    # vis.grid_floor()

    filename = os.path.join(model_dir, object_name, 'textured_simple.obj')
    texture_filename = os.path.join(model_dir, object_name, 'texture_map.png')
    position = RT_obj[:3, 3]
    quat = mat2quat(RT_obj[:3, :3])
    # scalar-last (x, y, z, w) format in optas
    orientation = [quat[1], quat[2], quat[3], quat[0]]
    vis.obj(
        filename,
        png_texture_filename=texture_filename,
        position=position,
        orientation=orientation,
        euler_degrees=False,
    )
    print(object_name, position, orientation)

    # grasp
    position = RT_grasp[:3, 3]
    # scalar-last (x, y, z, w) format in optas
    quat = mat2quat(RT_grasp[:3, :3])
    orientation = [quat[1], quat[2], quat[3], quat[0]]
    vis.robot(
        gripper_model,
        base_position=position,
        base_orientation=orientation,
        q=cfg['gripper_open_offsets'],
        alpha=0.5,
    )

    position = RT_off[:3, 3]
    # scalar-last (x, y, z, w) format in optas
    quat = mat2quat(RT_off[:3, :3])
    orientation = [quat[1], quat[2], quat[3], quat[0]]
    vis.robot(
        gripper_model,
        base_position=position,
        base_orientation=orientation,
        q=cfg['gripper_open_offsets'],
        alpha=0.5,
    )    
    vis.start()


def convert_plan_to_trajectory_toppra(robot, plan, is_show=False):

    import toppra as ta
    import toppra.constraint as constraint
    import toppra.algorithm as algo

    ndof = plan.shape[0]
    T = plan.shape[1]
    ss = np.linspace(0, 1, T)
    way_pts = plan.T
    vlims = robot.velocity_optimized_joint_limits.toarray().flatten()
    alims = np.ones(ndof) * 0.5
    
    path = ta.SplineInterpolator(ss, way_pts)
    pc_vel = constraint.JointVelocityConstraint(vlims)
    pc_acc = constraint.JointAccelerationConstraint(alims)

    instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
    jnt_traj = instance.compute_trajectory()

    ################################################################################
    # The output trajectory is an instance of
    # :class:`toppra.interpolator.AbstractGeometricPath`.
    ts_sample = np.linspace(0, jnt_traj.duration, 100)
    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)
    qdds_sample = jnt_traj(ts_sample, 2)
    
    if is_show:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, sharex=True)
        for i in range(path.dof):
            # plot the i-th joint trajectory
            axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
            axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
            axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
        axs[2].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position (rad)")
        axs[1].set_ylabel("Velocity (rad/s)")
        axs[2].set_ylabel("Acceleration (rad/s2)")
        plt.show()
    return qs_sample, qds_sample, qdds_sample, ts_sample