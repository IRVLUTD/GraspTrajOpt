import os
import sys
import pathlib
import numpy as np
import _init_paths
import scipy

# OpTaS
import optas
import tf_conversions
import casadi as cs
from optas.visualize import Visualizer
from gto.gto_models import GTORobotModel
from transforms3d.quaternions import mat2quat
from optas.spatialmath import Quaternion
from utils import *


class GTOPlanner:
    def __init__(self, robot, link_ee, link_gripper, collision_avoidance=True):
        
        # trajectory parameters
        self.T = 50  # no. time steps in trajectory
        self.Tmax = 10.0  # trajectory of 5 secs
        t = optas.linspace(0, self.Tmax, self.T)
        self.dt = float((t[1] - t[0]).toarray()[0, 0])  # time step

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
        builder = optas.OptimizationBuilder(T=self.T, robots=[self.robot])

        # Setup parameters
        qc = builder.add_parameter(
            "qc", self.robot.ndof
        )  # current robot joint configuration
        # goal pose of the gripper link_ee
        tf_goal = builder.add_parameter("tf_goal", 7, goal_size)
        # sdf field
        sdf_cost = builder.add_parameter("sdf_cost", self.robot.field_size)

        # Constraint: initial configuration
        builder.initial_configuration(
            self.robot_name,
            self.robot.extract_optimized_dimensions(qc),
        )
        builder.initial_configuration(
            self.robot_name, time_deriv=1
        )  # initial joint vel is zero
        builder.fix_configuration(
            self.robot_name, time_deriv=1, t=-1,
        )  # end joint vel is zero

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
        tf = tf_gripper[self.T - 1]    # last time step pose
        points_tf = tf[:3, :3] @ self.gripper_points.T + tf[:3, 3].reshape((3, 1))
        cost = cs.MX.zeros(goal_size)
        for i in range(goal_size):
            quat = Quaternion(tf_goal[3, i], tf_goal[4, i], tf_goal[5, i], tf_goal[6, i])
            tf_g = quat.getT(tf_goal[0, i], tf_goal[1, i], tf_goal[2, i])
            tf_gripper_goal = tf_g @ self.gripper_tf(Q[:, self.T - 1])
            points_tf_goal = tf_gripper_goal[:3, :3] @ self.gripper_points.T + tf_gripper_goal[:3, 3].reshape((3, 1))
            cost[i] = optas.sumsqr(points_tf - points_tf_goal)
        builder.add_cost_term("cost_pos", optas.mmin(cost))

        # Cost: obstacle avoidance
        if self.collision_avoidance:
            points_base_all = None            
            for i in range(self.T):
                q = Q[:, i]
                for name in self.robot.surface_pc_map.keys():
                    tf = self.robot.visual_tf[name](q)
                    points = self.robot.surface_pc_map[name].points
                    point_base = tf[:3, :3] @ points.T + tf[:3, 3].reshape((3, 1))
                    if points_base_all == None:
                        points_base_all = point_base
                    else:
                        points_base_all = optas.horzcat(points_base_all, point_base)
            points_base_all = points_base_all.T
            offsets = self.robot.points_to_offsets(points_base_all)
            builder.add_cost_term("cost_sdf", optas.sumsqr(sdf_cost[offsets]))

        # Cost: minimize joint velocity
        dQ = builder.get_robot_states_and_parameters(self.robot_name, time_deriv=1)
        builder.add_cost_term("min_join_vel", 0.01 * optas.sumsqr(dQ))

        # Constraint: joint position limits
        builder.enforce_model_limits(self.robot_name)  # joint limits extracted from URDF        

        # Setup solver
        solver_options = {'ipopt': {'max_iter': 200}}
        self.solver = optas.CasADiSolver(builder.build()).setup("ipopt", solver_options=solver_options)


    def plan(self, qc, RT, sdf_cost):
        self.setup_optimization()
        tf_goal = np.zeros((7, 1))
        quat = mat2quat(RT[:3, :3])
        orientation = [quat[1], quat[2], quat[3], quat[0]]
        tf_goal[:3, 0] = RT[:3, 3]
        tf_goal[3:, 0] = orientation

        # Set initial seed, note joint velocity will be set to zero
        Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)

        self.solver.reset_initial_seed(
            {f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(Q0)}
        )

        # Set parameters
        self.solver.reset_parameters(
            {
                "qc": optas.DM(qc),
                "tf_goal": optas.DM(tf_goal),
                "sdf_cost": optas.DM(sdf_cost),
                f"{self.robot_name}/q/p": self.robot.extract_parameter_dimensions(Q0),
            }
        )

        # Solve problem
        solution = self.solver.solve()

        # Get robot configuration
        Q = solution[f"{self.robot_name}/q"]
        return Q.toarray()
    

    def plan_goalset(self, qc, RTs, sdf_cost):

        n = RTs.shape[0]
        self.setup_optimization(goal_size=n)
        tf_goal = np.zeros((7, n))
        for i in range(n):
            RT = RTs[i]
            quat = mat2quat(RT[:3, :3])
            orientation = [quat[1], quat[2], quat[3], quat[0]]
            tf_goal[:3, i] = RT[:3, 3]
            tf_goal[3:, i] = orientation        

        Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)

        # Set initial seed, note joint velocity will be set to zero
        self.solver.reset_initial_seed(
            {f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(Q0)}
        )

        # Set parameters
        self.solver.reset_parameters(
            {
                "qc": optas.DM(qc),
                "tf_goal": optas.DM(tf_goal),
                "sdf_cost": optas.DM(sdf_cost),
                f"{self.robot_name}/q/p": self.robot.extract_parameter_dimensions(Q0),
            }
        )

        # Solve problem
        solution = self.solver.solve()

        # Get robot configuration
        Q = solution[f"{self.robot_name}/q"]
        return Q.toarray()


def main():

    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

    RT = np.array([[-0.05241979, -0.45344928, -0.88973933,  0.41363978],
        [-0.27383122, -0.8502871,   0.44947574,  0.12551154],
        [-0.96034825,  0.26719978, -0.07959669,  0.97476065],
        [ 0.,          0.,          0.,          1.        ]])

    # Setup robot
    model_dir = os.path.join(cwd, "../examples/robots", "fetch")
    urdf_filename = os.path.join(model_dir, "fetch.urdf")  

    # ['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 'shoulder_pan_joint', 
    # 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 
    # 'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']
    param_joints = ['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 
                    'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']
    
    collision_link_names = ["shoulder_pan_link", "shoulder_lift_link", "upperarm_roll_link",
                  "elbow_flex_link", "forearm_roll_link", "wrist_flex_link", "wrist_roll_link", "gripper_link",
                  "l_gripper_finger_link", "r_gripper_finger_link"]    

    robot = GTORobotModel(model_dir, urdf_filename=urdf_filename, 
                        time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
                        param_joints=param_joints,
                        collision_link_names=collision_link_names)
    robot.setup_workspace_field(arm_len=1.1, arm_height=1.1)    
    link_ee = "wrist_roll_link"  # end-effector link name
    link_gripper = 'gripper_link'
    print('optimized joint names:', robot.optimized_joint_names)

    # Initialize planner
    planner = GTOPlanner(robot, link_ee, link_gripper)

    # Plan trajectory
    sdf_cost = np.zeros(robot.field_size)
    qc = default_pose(robot)
    plan = planner.plan(qc, RT, sdf_cost)
    print(plan.shape)

    # visualization
    vis = Visualizer(camera_position=[3, 0, 3])
    vis.grid_floor()      

    q = [0.05, 0.05]
    position = RT[:3, 3]
    # scalar-last (x, y, z, w) format in optas
    quat = mat2quat(RT[:3, :3])
    orientation = [quat[1], quat[2], quat[3], quat[0]]
    # gripper
    urdf_filename = os.path.join(model_dir, "fetch_gripper.urdf")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
