# Python standard lib
import os, sys
import pathlib
from xml.parsers.expat import model
import numpy as np
import _init_paths

# OpTaS
import optas
import argparse
import casadi as cs
from optas.visualize import Visualizer
from gto.gto_models import GTORobotModel
from transforms3d.quaternions import mat2quat


class IKSolver:

    def __init__(self, robot, link_ee, link_gripper):
        self.robot = robot
        self.link_ee = link_ee
        self.link_gripper = link_gripper
        self.robot_name = robot.get_name()
        self.gripper_points = robot.surface_pc_map[link_gripper].points
        self.gripper_tf = robot.get_link_transform_function(link=link_gripper, base_link=link_ee)
        self.setup_optimization()


    def setup_optimization(self):
        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=1, robots=[self.robot])

        # setup parameters
        # tf goal is for link_ee
        tf_goal = builder.add_parameter("tf_goal", 4, 4)

        # get robot state variables
        q_T = builder.get_robot_states_and_parameters(self.robot_name)

        # forward kinematics for link_ee
        self.fk = self.robot.get_global_link_transform_function(link=self.link_ee)

        # Setting optimization - cost term and constraints
        tf = self.fk(q_T) @ self.gripper_tf(q_T)
        points_tf = tf[:3, :3] @ self.gripper_points.T + tf[:3, 3].reshape((3, 1))

        tf_gripper_goal = tf_goal @ self.gripper_tf(q_T)
        points_tf_goal = tf_gripper_goal[:3, :3] @ self.gripper_points.T + tf_gripper_goal[:3, 3].reshape((3, 1))
        builder.add_cost_term("cost_pos", optas.sumsqr(points_tf - points_tf_goal))

        # Constraint: joint position limits
        builder.enforce_model_limits(self.robot_name)  # joint limits extracted from URDF

        # setup solver
        self.solver = optas.CasADiSolver(builder.build()).setup("ipopt")    


    def solve_ik(self, q_0, RT):
        self.solver.reset_initial_seed({f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(q_0)})
        self.solver.reset_parameters({f"{self.robot_name}/q/p": self.robot.extract_parameter_dimensions(q_0), 
                                        "tf_goal": RT}) 
        solution = self.solver.solve()
        q = solution[f"{self.robot_name}/q"]

        # compute errors
        tf = self.fk(solution[f"{self.robot_name}/q"]).toarray()
        err_pos = np.linalg.norm(RT[:3, 3] - tf[:3, 3])
        quat1 = mat2quat(RT[:3, :3])
        quat2 = mat2quat(tf[:3, :3])
        err_rot = np.arccos(np.clip(2 * np.square(np.dot(quat1, quat2)) - 1, -1, 1)) * 180 / np.pi

        print("***********************************") 
        print("Casadi IK solution:")
        print(q)
        print('position error', err_pos)
        print('rotation error in degree', err_rot)
        return q, err_pos, err_rot


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_args()
    robot_name = args.robot
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

    # Setup robot
    model_dir = os.path.join(cwd, "../examples/robots", robot_name)
    urdf_filename = os.path.join(model_dir, f"{robot_name}.urdf")  

    if robot_name == 'fetch':
        param_joints = ['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 
                        'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']
        collision_link_names = ["shoulder_pan_link", "shoulder_lift_link", "upperarm_roll_link",
                    "elbow_flex_link", "forearm_roll_link", "wrist_flex_link", "wrist_roll_link", "gripper_link",
                    "l_gripper_finger_link", "r_gripper_finger_link"]
        link_ee = "wrist_roll_link"  # end-effector link name
        link_gripper = 'gripper_link'       
        arm_len = 1.1
        arm_height = 1.1

        RT = np.array([[-0.05241979, -0.45344928, -0.88973933,  0.41363978],
            [-0.27383122, -0.8502871,   0.44947574,  0.12551154],
            [-0.96034825,  0.26719978, -0.07959669,  0.97476065],
            [ 0.,          0.,          0.,          1.        ]])
    elif robot_name == 'panda':
        param_joints = ['panda_finger_joint1', 'panda_finger_joint2']
        collision_link_names = None  # all links
        link_ee = "panda_hand"     # end-effector link name
        link_gripper = 'panda_hand'
        arm_len = 1.0
        arm_height = 0

        RT = np.array([[-0.61162336,  0.79089652,  0.01998741,  0.46388378],
            [ 0.7883297,   0.6071185,   0.09971584, -0.15167381],
            [ 0.06673018,  0.07674521, -0.99481508,  0.22877409],
            [ 0.,          0.,          0.,          1.        ]])

    else:
        print(f'robot {robot_name} not supported')
        sys.exit(1)

    robot = GTORobotModel(model_dir, urdf_filename=urdf_filename, param_joints=param_joints) 
    robot_name = robot.get_name()
    print('optimized joint names:', robot.optimized_joint_names)

    # solve problem
    ik_solver = IKSolver(robot, link_ee, link_gripper)
    q_0 = np.zeros((robot.ndof, 1), dtype=np.float32)
    if robot_name == 'fetch':
        q_0[2, 0] = 0.38
        q_0[3, 0] = 0.009195
        q_0[4, 0] = 0.908270
        q_0[12, 0] = 0.05
        q_0[13, 0] = 0.05
    q_solution, err_pos, err_rot = ik_solver.solve_ik(q_0, RT)
    lo = robot.lower_actuated_joint_limits.toarray()
    hi = robot.upper_actuated_joint_limits.toarray()
    for i in range(robot.ndof):
        print(f'joint {i} {robot.actuated_joint_names[i]}: {lo[i]} <= {q_solution[i]} <= {hi[i]}')

    # visualize grasps
    vis = Visualizer(camera_position=[3, 2, 4])
    vis.grid_floor()      

    q = [0.05, 0.05]
    position = RT[:3, 3]
    # scalar-last (x, y, z, w) format in optas
    quat = mat2quat(RT[:3, :3])
    orientation = [quat[1], quat[2], quat[3], quat[0]]
    # gripper
    urdf_filename = os.path.join(model_dir, f"{robot_name}_gripper.urdf")
    gripper_model = GTORobotModel(model_dir, urdf_filename=urdf_filename)    
    vis.robot(
        gripper_model,
        base_position=position,
        base_orientation=orientation,
        q=q
    )
    # robot
    vis.robot(
        robot,
        base_position=[0.0, 0, 0],
        base_orientation=[0, 0, 0],
        euler_degrees=True,
        q=q_solution,
        alpha=0.1,
    )
    vis.start()