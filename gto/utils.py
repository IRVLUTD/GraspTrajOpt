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


def visualize_plan(robot, gripper_model, base_position, plan, depth_pc, RT_grasps_world):
    # visualize grasps
    vis = Visualizer(camera_position=[-1, 3.0, 5.0])
    vis.grid_floor()
    vis.points(
        depth_pc.points,
    )
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
            alpha = 0.1,
        )
    # robot trajectory
    # sample plan
    n = plan.shape[1]
    index = list(range(0, n, 10))
    if index[-1] != n - 1:
        index += [n - 1]
    vis.robot_traj(robot, plan[:, index], base_position, alpha_spec={'style': 'A'})
    vis.start()


def visualize_pose(robot, base_position, q, depth_pc):
    # visualize grasps
    vis = Visualizer(camera_position=[-2, 3.0, 6.0])
    vis.grid_floor()
    vis.points(
        depth_pc.points,
    )
    # robot
    vis.robot(
        robot,
        base_position=base_position,
        q=q,
    )
    vis.start()    