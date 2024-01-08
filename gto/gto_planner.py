import os
import sys
import pathlib
import numpy as np
import _init_paths

# OpTaS
import optas
import tf_conversions
import casadi as cs
from optas.visualize import Visualizer
from gto.gto_models import GTORobotModel
from transforms3d.quaternions import mat2quat
from utils import *


class Planner:
    def __init__(self, robot, link_ee, points, gripper_tf):
        
        # trajectory parameters
        self.T = 50  # no. time steps in trajectory
        self.Tmax = 10.0  # trajectory of 5 secs
        t = optas.linspace(0, self.Tmax, self.T)
        self.dt = float((t[1] - t[0]).toarray()[0, 0])  # time step

        # Setup robot
        self.robot = robot
        self.robot_name = robot.get_name()
        self.link_ee = link_ee
        self.points = cs.DM(points)
        self.gripper_tf = gripper_tf
        self.gripper_q = [0.05, 0.05]

        self.setup_optimization()


    def setup_optimization(self):
        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, robots=[self.robot])

        # Setup parameters
        qc = builder.add_parameter(
            "qc", self.robot.ndof
        )  # current robot joint configuration
        # goal pose of the gripper
        tf_goal = builder.add_parameter("tf_goal", 4, 4)

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

        # forward kinematics
        self.fk = self.robot.get_global_link_transform_function(self.link_ee, n=self.T)
        # trajectory for end-effector (FK)
        tf_ee = self.fk(Q)

        # Cost: reach goal pose
        tf_T = tf_ee[self.T - 1]    # last time step pose
        tf = tf_T @ self.gripper_tf(self.gripper_q)
        points_tf = tf[:3, :3] @ self.points.T + tf[:3, 3].reshape((3, 1))
        tf_gripper_goal = tf_goal @ self.gripper_tf(self.gripper_q)
        points_tf_goal = tf_gripper_goal[:3, :3] @ self.points.T + tf_gripper_goal[:3, 3].reshape((3, 1))
        builder.add_cost_term("cost_pos", optas.sumsqr(points_tf - points_tf_goal))

        # Cost: minimize joint velocity
        dQ = builder.get_robot_states_and_parameters(self.robot_name, time_deriv=1)
        builder.add_cost_term("min_join_vel", 0.01 * optas.sumsqr(dQ))

        # Constraint: joint position limits
        builder.enforce_model_limits(self.robot_name)  # joint limits extracted from URDF        

        # Setup solver
        self.solver = optas.CasADiSolver(builder.build()).setup("ipopt")


    def plan(self, qc, RT):
        # Set initial seed, note joint velocity will be set to zero
        Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)

        self.solver.reset_initial_seed(
            {f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(Q0)}
        )

        # Set parameters
        self.solver.reset_parameters(
            {
                "qc": optas.DM(qc),
                "tf_goal": optas.DM(RT),
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

    robot = optas.RobotModel(urdf_filename=urdf_filename, 
                             time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
                             param_joints=param_joints)
    link_ee = "wrist_roll_link"  # end-effector link name
    print('optimized joint names:', robot.optimized_joint_names)

    # load robot gripper model
    urdf_filename = os.path.join(model_dir, "fetch_gripper.urdf")
    gripper_model = optas.RobotModel(urdf_filename=urdf_filename)
    gto_robot_model = GTORobotModel(model_dir, gripper_model)
    link_gripper = 'gripper_link'
    gripper_pc = gto_robot_model.surface_pc_map[link_gripper].points
    gripper_tf = gto_robot_model.visual_tf[link_gripper]

    # Initialize planner
    planner = Planner(robot, link_ee, gripper_pc, gripper_tf)

    # Plan trajectory
    qc = default_pose(robot)
    plan = planner.plan(qc, RT)
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
    vis.robot(
        gripper_model,
        base_position=position,
        base_orientation=orientation,
        q=q
    )
    # robot
    vis.robot_traj(robot, plan, alpha_spec={'style': 'A'})
    vis.start()       

    return 0


if __name__ == "__main__":
    sys.exit(main())
