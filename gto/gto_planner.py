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
from optas.spatialmath import standoff
from gto.gto_models import GTORobotModel
from gto.ik_solver import IKSolver
from gto.utils import load_yaml, get_root_dir
from transforms3d.quaternions import mat2quat
from optas.spatialmath import standoff
from gto.utils import interpolate_waypoints


class GTOPlanner:
    def __init__(self, robot, link_ee, link_gripper, collision_avoidance=True, standoff_distance=-0.1, standoff_offset=-10):
        
        # trajectory parameters
        self.T = 50  # no. time steps in trajectory
        self.Tmax = 10.0  # trajectory of 5 secs
        t = optas.linspace(0, self.Tmax, self.T)
        self.dt = float((t[1] - t[0]).toarray()[0, 0])  # time step
        self.standoff_offset = standoff_offset
        self.standoff_distance = standoff_distance     

        # Setup robot
        self.robot = robot
        self.robot_name = robot.get_name()
        self.link_ee = link_ee
        self.link_gripper = link_gripper
        self.gripper_points = robot.surface_pc_map[link_gripper].points
        self.gripper_tf = robot.get_link_transform_function(link=link_gripper, base_link=link_ee)
        self.collision_avoidance = collision_avoidance


    def setup_optimization(self, goal_size=1, use_standoff=False, axis_standoff='x'):
        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, robots=[self.robot])

        # Setup parameters
        qc = builder.add_parameter(
            "qc", self.robot.ndof
        )  # current robot joint configuration
        # goal pose of the gripper link_ee
        tf_goal = builder.add_parameter("tf_goal", 16, goal_size)
        # sdf field
        sdf_cost_obstacle = builder.add_parameter("sdf_cost_obstacle", self.robot.field_size)
        # base position
        base_position = builder.add_parameter("base_position", 3)        

        # Constraint: initial configuration
        builder.initial_configuration(
            self.robot_name,
            self.robot.extract_optimized_dimensions(qc),
        )
        builder.initial_configuration(
            self.robot_name, time_deriv=1
        )  # initial joint vel is zero

        # Constraint: dynamics
        builder.integrate_model_states(
            self.robot_name,
            time_deriv=1,  # i.e. integrate velocities to positions
            dt=self.dt,
        )

        # Get joint trajectory
        Q = builder.get_robot_states_and_parameters(
            self.robot_name
        )  # ndof-by-T symbolic array for robot trajectory

        # forward kinematics for link gripper
        self.fk = self.robot.get_global_link_transform_function(self.link_gripper, n=self.T)
        # trajectory for end-effector (FK)
        tf_gripper = self.fk(Q)

        # Cost: reach goal pose
        # tf0 = tf_gripper[0]    # first time step pose
        tf = tf_gripper[self.T - 1]    # last time step pose
        points_tf = tf[:3, :3] @ self.gripper_points.T + tf[:3, 3].reshape((3, 1))
        if use_standoff:
            tf_standoff = tf_gripper[self.T + self.standoff_offset]
            points_standoff = tf_standoff[:3, :3] @ self.gripper_points.T + tf_standoff[:3, 3].reshape((3, 1))
        cost = cs.MX.zeros(goal_size)
        for i in range(goal_size):
            tf_g = tf_goal[:, i].reshape((4, 4)).T
            tf_gripper_goal = tf_g @ self.gripper_tf(Q[:, self.T - 1])
            points_tf_goal = tf_gripper_goal[:3, :3] @ self.gripper_points.T + tf_gripper_goal[:3, 3].reshape((3, 1))
            # cost_prior = optas.sumsqr(tf0[:3, 3] - tf_gripper_goal[:3, 3])
            # standoff
            if use_standoff:
                self.pose_standoff = standoff(self.standoff_distance, axis_standoff) 
                tf_gripper_standoff = tf_g @ self.pose_standoff @ self.gripper_tf(Q[:, self.T + self.standoff_offset])
                points_standoff_goal = tf_gripper_standoff[:3, :3] @ self.gripper_points.T + tf_gripper_standoff[:3, 3].reshape((3, 1))
                cost[i] = optas.sumsqr(points_tf - points_tf_goal) + optas.sumsqr(points_standoff - points_standoff_goal)
            else:
                cost[i] = optas.sumsqr(points_tf - points_tf_goal) 
        builder.add_cost_term("cost_pos", optas.mmin(cost))

        # obstacle avoidance
        if self.collision_avoidance:
            points_world_all = None
            for i in range(self.T):
                q = Q[:, i]
                for name in self.robot.surface_pc_map.keys():
                    tf = self.robot.visual_tf[name](q)
                    points = self.robot.surface_pc_map[name].points
                    points_world = tf[:3, :3] @ points.T + tf[:3, 3].reshape((3, 1)) + base_position.reshape((3, 1))
                    if points_world_all is None:
                        points_world_all = points_world
                    else:
                        points_world_all = optas.horzcat(points_world_all, points_world)
            points_world_all = points_world_all.T
            offsets = self.robot.points_to_offsets(points_world_all)
            builder.add_cost_term("cost_obstacle", 10 * optas.sumsqr(sdf_cost_obstacle[offsets]))

        # Cost: minimize joint velocity
        dQ = builder.get_robot_states_and_parameters(self.robot_name, time_deriv=1)
        builder.add_cost_term("min_join_vel", 0.01 * optas.sumsqr(dQ))

        # Constraint: joint position limits
        builder.enforce_model_limits(self.robot_name)  # joint limits extracted from URDF        

        # Setup solver
        solver_options = {'ipopt': {'max_iter': 100, 'tol': 1e-15}}
        self.solver = optas.CasADiSolver(builder.build()).setup("ipopt", solver_options=solver_options)


    def plan(self, qc, RT, sdf_cost_obstacle, base_position, q_solution=None, use_standoff=True, axis_standoff='x'):
        self.setup_optimization(goal_size=1, use_standoff=use_standoff, axis_standoff=axis_standoff)
        tf_goal = np.zeros((16, 1))
        tf_goal[:, 0] = RT.flatten()

        # Set initial seed, note joint velocity will be set to zero
        if q_solution is None:
            Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)
        else:
            # interpolate waypoints
            data = interpolate_waypoints(np.stack([qc, q_solution]), self.T, self.robot.ndof)
            index = np.array(self.robot.parameter_joint_indexes).astype(np.int32)
            data[:, index] = np.array(qc)[index]
            Q0 = optas.DM(data.T)

        self.solver.reset_initial_seed(
            {f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(Q0)}
        )

        # Set parameters
        self.solver.reset_parameters(
            {
                "qc": optas.DM(qc),
                "tf_goal": optas.DM(tf_goal),
                "sdf_cost_obstacle": optas.DM(sdf_cost_obstacle),
                "base_position": optas.DM(base_position),
                f"{self.robot_name}/q/p": self.robot.extract_parameter_dimensions(Q0),
            }
        )

        # Solve problem
        solution = self.solver.solve()

        # Get robot configuration
        Q = solution[f"{self.robot_name}/q"]
        dQ = solution[f"{self.robot_name}/dq"]
        cost = solution["f"]
        return Q.toarray(), dQ.toarray(), cost.toarray().flatten()
    

    def plan_goalset(self, qc, RTs, sdf_cost_all, sdf_cost_obstacle, base_position, q_solutions=None, use_standoff=True, axis_standoff='x', interpolate=True):
        n = RTs.shape[0]
        self.setup_optimization(goal_size=n, use_standoff=use_standoff, axis_standoff=axis_standoff)
        tf_goal = np.zeros((16, n))
        for i in range(n):
            RT = RTs[i]
            tf_goal[:, i] = RT.flatten()

        if q_solutions is None:
            Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)
        else:
            # intialize the goal
            cost_all = []
            dist_all = []
            plan_all = []
            for i in range(q_solutions.shape[1]):
                q_solution = q_solutions[:, i]
                # interpolate waypoints
                data = interpolate_waypoints(np.stack([qc, q_solution]), self.T, self.robot.ndof)
                index = np.array(self.robot.parameter_joint_indexes).astype(np.int32)
                data[:, index] = np.array(qc)[index]
                plan = data.T
                plan_all.append(plan.copy())
                cost, dist = self.robot.compute_plan_cost(plan, sdf_cost_obstacle, base_position)
                print(f'plan {i}, cost {cost:.2f}, dist {dist:.2f}')
                cost_all.append(cost)
                dist_all.append(dist)    # large cost reach is better
            ind = np.lexsort((dist_all, cost_all))   # sort by cost, then by distance
            print('intialize with solution', ind[0])
            if interpolate:
                Q0 = optas.DM(plan_all[ind[0]])
            else:
                Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)
                for i in range(self.T + self.standoff_offset, self.T):
                    Q0[:, i] = plan_all[ind[0]][:, self.T - 1]

        # Set initial seed, note joint velocity will be set to zero
        self.solver.reset_initial_seed(
            {f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(Q0)}
        )

        # Set parameters
        self.solver.reset_parameters(
            {
                "qc": optas.DM(qc),
                "tf_goal": optas.DM(tf_goal),
                "sdf_cost_obstacle": optas.DM(sdf_cost_all),
                "base_position": optas.DM(base_position),
                f"{self.robot_name}/q/p": self.robot.extract_parameter_dimensions(Q0),
            }
        )

        # Solve problem
        solution = self.solver.solve()

        # Get robot configuration
        Q = solution[f"{self.robot_name}/q"]
        dQ = solution[f"{self.robot_name}/dq"]
        cost = solution["f"]
        return Q.toarray(), dQ.toarray(), cost.toarray().flatten()


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
    
    # load config file
    root_dir = get_root_dir()
    config_file = os.path.join(root_dir, 'data', 'configs', f'{robot_name}.yaml')
    if not os.path.exists(config_file):
        print(f'robot {robot_name} not supported', config_file)
        sys.exit(1) 
    cfg = load_yaml(config_file)['robot_cfg']
    print(cfg)

    if robot_name == 'fetch':
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
    planner = GTOPlanner(robot, cfg['link_ee'], cfg['link_gripper'])
    sdf_cost_obstacle = np.zeros(robot.field_size)
    sdf_cost_target = np.zeros(robot.field_size)
    qc = np.array(cfg['default_pose'])
    base_position = [0, 0, 0]

    # solve IK first
    ik_solver = IKSolver(robot, cfg['link_ee'], cfg['link_gripper'])
    ik_solver.setup_optimization()
    q_solution, err_pos, err_rot, cost_ik = ik_solver.solve_ik(qc.reshape(-1, 1), RT, sdf_cost_obstacle, base_position)

    # Plan trajectory 
    plan, dQ, cost = planner.plan(qc, RT, sdf_cost_obstacle, sdf_cost_target, base_position, q_solution, use_standoff=True, axis_standoff=cfg['axis_standoff'])
    print(plan.shape)

    lo = robot.lower_actuated_joint_limits.toarray()
    hi = robot.upper_actuated_joint_limits.toarray()
    print('last step')
    for i in range(robot.ndof):
        print(f'joint {i} {robot.actuated_joint_names[i]}: {lo[i]} <= {plan[i, -1]} <= {hi[i]}')

    # visualization
    vis = Visualizer(camera_position=[3, 0, 3])
    vis.grid_floor()      

    q = [0.05, 0.05]
    position = RT[:3, 3]
    # scalar-last (x, y, z, w) format in optas
    quat = mat2quat(RT[:3, :3])
    orientation = [quat[1], quat[2], quat[3], quat[0]]
    # gripper
    urdf_filename = os.path.join(model_dir, f"{robot_name}_gripper.urdf")
    gripper = GTORobotModel(model_dir, urdf_filename=urdf_filename)    
    vis.robot(
        gripper,
        base_position=position,
        base_orientation=orientation,
        q=q
    )
    # robot trajectory
    # sample plan
    n = plan.shape[1]
    index = list(range(0, n, 10))
    if index[-1] != n - 1:
        index += [n - 1]
    vis.robot_traj(robot, plan[:, index], alpha_spec={'style': 'A'})
    vis.start()
