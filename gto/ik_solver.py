# Python standard lib
import os
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


class IKSolver:

    def __init__(self, robot, eff_link, points, gripper_tf):
        self.robot = robot
        self.eff_link = eff_link
        self.robot_name = robot.get_name()
        self.points = cs.DM(points)
        self.gripper_tf = gripper_tf
        self.gripper_q = [0.05, 0.05]
        self.setup_optimization()


    def setup_optimization(self):
        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=1, robots=[self.robot])

        # setup parameters
        tf_goal = builder.add_parameter("tf_goal", 4, 4)

        # get robot state variables
        q_T = builder.get_robot_states_and_parameters(self.robot_name)

        # forward kinematics
        self.fk = self.robot.get_global_link_transform_function(link=self.eff_link)

        # Setting optimization - cost term and constraints
        tf = self.fk(q_T) @ self.gripper_tf(self.gripper_q)
        points_tf = tf[:3, :3] @ self.points.T + tf[:3, 3].reshape((3, 1))
        tf_gripper_goal = tf_goal @ self.gripper_tf(self.gripper_q)
        points_tf_goal = tf_gripper_goal[:3, :3] @ self.points.T + tf_gripper_goal[:3, 3].reshape((3, 1))
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

        print("***********************************")
        print(self.solver._p_dict)         
        print("Casadi IK solution:")
        print(solution[f"{robot_name}/q"])
        print(self.fk(solution[f"{robot_name}/q"]))
        return solution[f"{self.robot_name}/q"]



if __name__ == "__main__":
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

    robot = optas.RobotModel(urdf_filename=urdf_filename, param_joints=param_joints) 
    robot_name = robot.get_name()
    link_ee = "wrist_roll_link"  # end-effector link name
    print('optimized joint names:', robot.optimized_joint_names)

    # load robot gripper model
    urdf_filename = os.path.join(model_dir, "fetch_gripper.urdf")
    gripper_model = optas.RobotModel(urdf_filename=urdf_filename)
    gto_robot_model = GTORobotModel(model_dir, gripper_model)
    link_gripper = 'gripper_link'
    gripper_pc = gto_robot_model.surface_pc_map[link_gripper].points
    gripper_tf = gto_robot_model.visual_tf[link_gripper]

    # solve problem
    ik_solver = IKSolver(robot, link_ee, gripper_pc, gripper_tf)
    q_0 = np.zeros((robot.ndof, 1), dtype=np.float32)
    q_0[2, 0] = 0.38
    q_0[3, 0] = 0.009195
    q_0[4, 0] = 0.908270
    q_0[12, 0] = 0.05
    q_0[13, 0] = 0.05
    q_solution = ik_solver.solve_ik(q_0, RT)
    lo = robot.lower_actuated_joint_limits.toarray()
    hi = robot.upper_actuated_joint_limits.toarray()
    for i in range(robot.ndof):
        print(f'joint {i} {robot.actuated_joint_names[i]}: {lo[i]} <= {q_solution[i]} <= {hi[i]}')

    # visualize grasps
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
    vis.robot(
        robot,
        base_position=[0.0, 0, 0],
        base_orientation=[0, 0, 0],
        euler_degrees=True,
        q=q_solution,
        alpha=0.1,
    )
    vis.start()    