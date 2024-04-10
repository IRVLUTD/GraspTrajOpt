import os
import sys
import pathlib
import numpy as np
import argparse
import _init_paths

# OpTaS
import optas
import casadi as cs
from optas.visualize import Visualizer
from optas.spatialmath import rotz, rt2tr
from optas.models import TaskModel
from gto.gto_models import GTORobotModel
from gto.utils import load_yaml, get_root_dir, rotZ
from transforms3d.quaternions import mat2quat


class BasePlanner:
    def __init__(self, robot, link_ee, link_gripper, collision_avoidance=True):

        # setup task model (x, y, theta)
        self.task = TaskModel('base_pose_estimator', dim=3)
        self.task_name = self.task.name

        # Setup robot
        self.robot = robot
        self.robot_name = robot.get_name()
        self.link_ee = link_ee
        self.link_gripper = link_gripper
        self.gripper_points = robot.surface_pc_map[link_gripper].points
        self.gripper_tf = robot.get_link_transform_function(link=link_gripper, base_link=link_ee)
        self.collision_avoidance = collision_avoidance


    def setup_optimization(self, goal_size=1):
        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=goal_size, robots=[self.robot], tasks=[self.task])

        # Setup parameters
        # tf goals are grasps in the current robot base pose
        tf_goal = builder.add_parameter("tf_goal", 16, goal_size)

        # get optimized robot base
        q = builder.get_model_states(self.task_name)
        x = q[0]
        y = q[1]
        theta = q[2]
        R = rotz(theta)
        t = cs.horzcat(x, y, 0.0)
        tf_base = rt2tr(R, t)

        # penalize movement
        builder.add_cost_term("cost_effort", 0.01 * optas.sumsqr(q))

        # Get joint trajectory
        Q = builder.get_robot_states_and_parameters(
            self.robot_name
        )  # ndof-by-T symbolic array for robot trajectory

        # forward kinematics for link gripper
        self.fk = self.robot.get_global_link_transform_function(self.link_gripper, n=goal_size)
        # trajectory for end-effector (FK) in robot base
        tf_gripper = self.fk(Q)

        # Cost: reach goal pose
        if goal_size == 1:
            points_tf = tf_gripper[:3, :3] @ self.gripper_points.T + tf_gripper[:3, 3].reshape((3, 1))
            tf_g = tf_goal[:, 0].reshape((4, 4)).T
            tf_gripper_goal = tf_base @ tf_g @ self.gripper_tf(Q[:, 0])
            points_tf_goal = tf_gripper_goal[:3, :3] @ self.gripper_points.T + tf_gripper_goal[:3, 3].reshape((3, 1))
            builder.add_cost_term("cost_pos", optas.sumsqr(points_tf - points_tf_goal))
        else:
            cost = cs.MX.zeros(goal_size)
            for i in range(goal_size):
                tf = tf_gripper[i]
                points_tf = tf[:3, :3] @ self.gripper_points.T + tf[:3, 3].reshape((3, 1))
                tf_g = tf_goal[:, i].reshape((4, 4)).T
                tf_gripper_goal = tf_base @ tf_g @ self.gripper_tf(Q[:, i])
                points_tf_goal = tf_gripper_goal[:3, :3] @ self.gripper_points.T + tf_gripper_goal[:3, 3].reshape((3, 1))
                cost[i] = optas.sumsqr(points_tf - points_tf_goal)
            builder.add_cost_term("cost_pos", optas.sum1(cost))

        # Constraint: joint position limits
        builder.enforce_model_limits(self.robot_name)  # joint limits extracted from URDF        

        # Setup solver
        solver_options = {'ipopt': {'max_iter': 100, 'tol': 1e-15}}
        self.solver = optas.CasADiSolver(builder.build()).setup("ipopt", solver_options=solver_options)
    

    def plan_goalset(self, qc, RTs):
        n = RTs.shape[0]
        self.setup_optimization(goal_size=n)
        tf_goal = np.zeros((16, n))
        for i in range(n):
            RT = RTs[i]
            tf_goal[:, i] = RT.flatten()

        Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, n)
        x0 = np.zeros((3, ), dtype=np.float32)

        # Set initial seed
        self.solver.reset_initial_seed(
            {
                f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(Q0),
                f"{self.task_name}/y/x": x0
            }
        )

        # Set parameters
        self.solver.reset_parameters(
            {
                "tf_goal": optas.DM(tf_goal),
                f"{self.robot_name}/q/p": self.robot.extract_parameter_dimensions(Q0),
            }
        )

        # Solve problem
        solution = self.solver.solve()

        # Get robot configuration
        Q = solution[f"{self.robot_name}/q"]
        y = solution[f"{self.task_name}/y"]
        print("***********************************") 
        print("Casadi robot base pose solution:")
        print(Q, Q.shape)
        print(y, y.shape)
        return Q.toarray(), y.toarray().flatten()


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-r",
        "--robot",
        type=str,
        default="fetch",
        help="Robot name",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_args()
    robot_name = args.robot
    
    # load config file
    root_dir = get_root_dir()
    config_file = os.path.join(root_dir, 'data', 'configs', f'{robot_name}.yaml')
    if not os.path.exists(config_file):
        print(f'robot {robot_name} not supported', config_file)
        sys.exit(1) 
    cfg = load_yaml(config_file)['robot_cfg']
    print(cfg)

    if 'fetch' in robot_name:
        RT = np.array([[-0.05241979, -0.45344928, -0.88973933,  0.41363978],
            [-0.27383122, -0.8502871,   0.44947574,  0.12551154],
            [-0.96034825,  0.26719978, -0.07959669,  0.97476065],
            [ 0.,          0.,          0.,          1.        ]])
    elif robot_name == 'panda':
        RT = np.array([[-0.61162336,  0.79089652,  0.01998741,  0.46388378],
            [ 0.7883297,   0.6071185,   0.09971584, -0.15167381],
            [ 0.06673018,  0.07674521, -0.99481508,  0.22877409],
            [ 0.,          0.,          0.,          1.        ]])
    else:
        print(f'robot {robot_name} not supported')
        sys.exit(1)    

    # Setup robot
    model_dir = os.path.join(root_dir, 'data', 'robots', cfg['robot_name'])
    urdf_filename = os.path.join(root_dir, cfg['urdf_robot_path'])

    robot = GTORobotModel(model_dir, urdf_filename=urdf_filename, 
                        time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
                        param_joints=cfg['param_joints'],
                        collision_link_names=cfg['collision_link_names'])
    robot.setup_workspace_field(arm_len=cfg['arm_len'], arm_height=cfg['arm_height'])
    print('optimized joint names:', robot.optimized_joint_names)

    # Initialize planner
    planner = BasePlanner(robot, cfg['link_ee'], cfg['link_gripper'])
    qc = np.array(cfg['default_pose'])

    # Plan base location
    RT[:2, 3] += 2 * np.random.randn(2)
    RT = np.expand_dims(RT, axis=0)
    plan, y = planner.plan_goalset(qc, RT)
    RT_base = rotZ(y[2])
    RT_base[0, 3] = y[0]
    RT_base[1, 3] = y[1]
    RT_base = np.linalg.inv(RT_base)
    print(RT_base)

    # visualization
    vis = Visualizer(camera_position=[3, 0, 3])
    vis.grid_floor()      

    q = [0.05, 0.05]
    position = RT[0, :3, 3]
    # scalar-last (x, y, z, w) format in optas
    quat = mat2quat(RT[0, :3, :3])
    orientation = [quat[1], quat[2], quat[3], quat[0]]
    # gripper
    urdf_filename = os.path.join(root_dir, cfg['urdf_gripper_path'])
    gripper = GTORobotModel(model_dir, urdf_filename=urdf_filename)    
    vis.robot(
        gripper,
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
        q=qc,
        alpha=0.4,
    )
    # new base
    position = RT_base[:3, 3]
    # scalar-last (x, y, z, w) format in optas
    quat = mat2quat(RT_base[:3, :3])
    orientation = [quat[1], quat[2], quat[3], quat[0]]    
    vis.robot(
        robot,
        base_position=position,
        base_orientation=orientation,
        q=plan,
        alpha=0.4,
    )    
    vis.start()