import os
import time
import pathlib
import _init_paths
import optas
import pybullet as p
import pybullet_data
import numpy as np
from utils import *

cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory


def pybullet_show_frame(RT):

    origin = RT[:3, 3]
    frame = np.eye(3)
    frame_new = RT[:3, :3] @ frame + origin.reshape((3, 1))

    x_axis = p.addUserDebugLine(lineFromXYZ = origin,
                            lineToXYZ = frame_new[:, 0],
                            lineColorRGB = [1, 0, 0],
                            lineWidth = 0.1,
                            lifeTime = 0
                            )    

    y_axis = p.addUserDebugLine(lineFromXYZ = origin,
                            lineToXYZ = frame_new[:, 1],
                            lineColorRGB = [0, 1, 0],
                            lineWidth = 0.1,
                            lifeTime = 0
                            )
    
    z_axis = p.addUserDebugLine(lineFromXYZ = origin,
                            lineToXYZ = frame_new[:, 2],
                            lineColorRGB = [0, 0, 1],
                            lineWidth = 0.1,
                            lifeTime = 0
                            )


class PyBullet:
    def __init__(
        self,
        dt,
        add_floor=True,
        camera_distance=1.5,
        camera_yaw=45,
        camera_pitch=-40,
        camera_target_position=[0, 0, 0.5],
        record_video=False,
        gui=True,
    ):
        connect_kwargs = {}
        if record_video:
            stamp = time.time_ns()
            video_filename = pathlib.Path.home() / "Videos" / f"optas_video_{stamp}.mp4"
            connect_kwargs["options"] = f"--mp4={video_filename.absolute()}"

        if gui:
            self.client_id = p.connect(p.GUI, **connect_kwargs)
        else:
            self.client_id = p.connect(p.DIRECT, **connect_kwargs)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(gravX=0.0, gravY=0.0, gravZ=-9.81)
        p.setTimeStep(dt)
        p.configureDebugVisualizer(flag=p.COV_ENABLE_GUI, enable=0)
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target_position,
        )
        if add_floor:
            self.add_floor()

    def add_floor(self, base_position=[0.0] * 3):
        colid = p.createCollisionShape(p.GEOM_PLANE)
        visid = p.createVisualShape(
            p.GEOM_PLANE, rgbaColor=[0, 1, 0, 1.0], planeNormal=[0, 0, 1]
        )
        p.createMultiBody(
            baseMass=0.0,
            basePosition=base_position,
            baseCollisionShapeIndex=colid,
            baseVisualShapeIndex=visid,
        )

    def start(self):
        p.setRealTimeSimulation(1)

    def stop(self):
        p.setRealTimeSimulation(0)

    def close(self):
        p.disconnect(self.client_id)


class DynamicBox:
    def __init__(self, base_position, half_extents, base_mass=0.5):
        colid = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visid = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=[0, 1, 0, 1.0], halfExtents=half_extents
        )
        self._id = p.createMultiBody(
            baseMass=base_mass,
            basePosition=base_position,
            baseCollisionShapeIndex=colid,
            baseVisualShapeIndex=visid,
        )
        p.changeDynamics(
            self._id,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.0,
            rollingFriction=0.0,
            restitution=0.0,
            linearDamping=0.04,
            angularDamping=0.04,
            contactStiffness=2000.0,
            contactDamping=0.7,
        )

    def get_pose(self):
        pos, ori = p.getBasePositionAndOrientation(self._id)
        eul = p.getEulerFromQuaternion(ori)
        return pos, eul


class VisualBox:
    def __init__(
        self,
        base_position,
        half_extents,
        rgba_color=[0, 1, 0, 1.0],
        base_orientation=[0, 0, 0, 1],
    ):
        visid = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=rgba_color, halfExtents=half_extents
        )
        self._id = p.createMultiBody(
            baseMass=0.0,
            basePosition=base_position,
            baseOrientation=base_orientation,
            baseVisualShapeIndex=visid,
        )

    def reset(self, base_position, base_orientation=[0, 0, 0, 1]):
        p.resetBasePositionAndOrientation(
            self._id,
            base_position,
            base_orientation,
        )

class FixedBaseRobot:
    def __init__(self, urdf_filename, base_position=[0.0] * 3, fix_base=1):
        self._id = p.loadURDF(
            fileName=urdf_filename, useFixedBase=fix_base, basePosition=base_position
        )
        self.num_joints = p.getNumJoints(self._id)
        self._actuated_joints = []
        self._actuated_joint_names = []
        for j in range(self.num_joints):
            info = p.getJointInfo(self._id, j)
            print(f'joint {j}:', info[1])
            if info[2] in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}:
                self._actuated_joints.append(j)
                self._actuated_joint_names.append(info[1])
        self.ndof = len(self._actuated_joints)
        print('robot ndof', self.ndof)
        print(self._actuated_joint_names)
        self.robot = optas.RobotModel(urdf_filename, time_derivs=[0])

        self.position_control_gain_p = [0.01] * self.ndof
        self.position_control_gain_d = [1.0] * self.ndof
        self.max_torque = [1000] * self.ndof

    def reset(self, q):
        for j, idx in enumerate(self._actuated_joints):
            qj = q[j]
            p.resetJointState(self._id, idx, qj)           

    def cmd(self, q):
        p.setJointMotorControlArray(
            self._id,
            self._actuated_joints,
            p.POSITION_CONTROL,
            targetPositions=np.asarray(q).tolist(),
            forces=self.max_torque,
            positionGains=self.position_control_gain_p,
            velocityGains=self.position_control_gain_d,
        )     

    def cmd_torque(self, taus):
        for index in range(len(self._actuated_joints)):
            p.setJointMotorControl2(self._id, index, p.VELOCITY_CONTROL, force=0)

        p.setJointMotorControlArray(
            self._id,
            self._actuated_joints,
            p.TORQUE_CONTROL,
            forces = np.asarray(taus).tolist(),
        )

    def q(self):
        return [state[0] for state in p.getJointStates(self._id, self._actuated_joints)]

    def default_pose(self):
        return np.zeros((self.ndof, ))
    
    def execute_plan(self, plan):
        '''
        @ param plan: shape (ndof, T)
        '''        
        for t in range(plan.shape[1]):
            self.cmd(plan[:, t])
            if t >= plan.shape[1] - 5:
                num = 500
            else:
                num = 200
            for _ in range(num):
                p.stepSimulation()

    def open_gripper(self):
        pass

    def close_gripper(self):
        pass

    def retract(self):
        q = self.default_pose()
        self.cmd(q)
        for _ in range(1000):
            p.stepSimulation()
        self.open_gripper()

    def get_standoff_pose(self, offset, axis):
        pose_standoff = np.eye(4, dtype=np.float32)
        if axis == 'x':
            pose_standoff[0, 3] = offset
        elif axis == 'y':
            pose_standoff[1, 3] = offset
        elif axis == 'z':
            pose_standoff[2, 3] = offset
        else:
            print('unknow standoff axis', axis)
        return pose_standoff                   


class Panda(FixedBaseRobot):
    def __init__(self, base_position=[0.0] * 3, scene_type='tabletop'):
        f = os.path.join(cwd, "../data/robots", "panda", "panda.urdf")
        self.urdf_filename = f
        super().__init__(f, base_position=base_position)
        self.ee_index = 7
        self.camera_link_index = 10
        self.gripper_open_offsets = [0.04, 0.04]
        self.finger_index = [7, 8]
        self.scene_type = scene_type

    def default_pose(self):
        # no panda joint 8
        if self.scene_type == 'tabletop':
            return np.array([0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
        else:
            return np.array([0.0, -1.285, 0, -2.356 + 1.4, 0.0, 1.571 - 0.6, 0.785, 0.0, 0.0])
    
    def get_camera_pose(self):
        pos, orn = p.getLinkState(self._id, self.camera_link_index)[:2]
        cam_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        cam_pose_mat = unpack_pose(cam_pose)

        RT = cam_pose_mat.dot(rotX(-np.pi / 2).dot(rotZ(-np.pi)))
        pose = RT.dot(rotX(np.pi))
        # pybullet_show_frame(pose)

        cam_view_matrix = se3_inverse(RT).T.flatten().tolist()  # z backward
        return cam_view_matrix, pose
    
    def close_gripper(self):
        q = self.q()
        q[-2] = 0
        q[-1] = 0
        self.cmd(q)
        for _ in range(1000):
            p.stepSimulation()

    def open_gripper(self):
        q = self.q()
        q[-2] = 0.04
        q[-1] = 0.04
        self.cmd(q)
        for _ in range(100):
            p.stepSimulation()        
    

class Fetch(FixedBaseRobot):
    def __init__(self, base_position=[0.0] * 3, scene_type='tabletop', fix_base=1):
        f = os.path.join(cwd, "../data/robots", "fetch_igibson", "fetch_gripper.urdf")
        self.urdf_filename = f
        super().__init__(f, base_position=base_position, fix_base=fix_base)
        self.ee_index = 16
        self.camera_link_index = 7
        self.gripper_open_joints = [0.05, 0.05]
        self.finger_index = [12, 13]
        self.wheels = [1, 2]
        self.scene_type = scene_type               
        

    def default_pose(self):
        # set robot pose
        # [b'r_wheel_joint', b'l_wheel_joint', b'torso_lift_joint', b'head_pan_joint', b'head_tilt_joint', b'shoulder_pan_joint', 
        # b'shoulder_lift_joint', b'upperarm_roll_joint', b'elbow_flex_joint', b'forearm_roll_joint', b'wrist_flex_joint', 
        # b'wrist_roll_joint', b'r_gripper_finger_joint', b'l_gripper_finger_joint', b'bellows_joint']

        joint_command = np.zeros((self.robot.ndof, ), dtype=np.float32)
        # arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
        #              "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        # arm_joint_positions  = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]

        # raise torso
        joint_command[2] = 0.4
        # move head
        if self.scene_type == 'tabletop':
            joint_command[3] = 0.009195
            joint_command[4] = 0.908270
        elif self.scene_type == 'shelf':
            joint_command[3] = 0.009195
            joint_command[4] = 0.348270
        # move arm
        index = [5, 6, 7, 8, 9, 10, 11]
        joint_command[index] = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        # open gripper
        joint_command[12] = 0.05
        joint_command[13] = 0.05        
        return joint_command
    

    def cmd_wheel_velocities(self, velocities):
        for i, wheel in enumerate(self.wheels):
            p.setJointMotorControl2(self._id,
                            wheel,
                            p.VELOCITY_CONTROL,
                            targetVelocity=velocities[i],
                            force=5)

    
    def get_camera_pose(self):
        pos, orn = p.getLinkState(self._id, self.camera_link_index)[:2]
        cam_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        cam_pose_mat = unpack_pose(cam_pose)
        # z backward
        RT = cam_pose_mat.dot(rotX(-np.pi))
        # show frame for debugging
        # pybullet_show_frame(cam_pose_mat)
        cam_view_matrix = se3_inverse(RT).T.flatten().tolist()
        return cam_view_matrix, cam_pose_mat


    def close_gripper(self):
        q = self.q()
        q[12] = 0
        q[13] = 0
        self.cmd(q)
        for _ in range(100):
            p.stepSimulation()


    def open_gripper(self):
        q = self.q()
        q[12] = 0.05
        q[13] = 0.05
        self.cmd(q)
        for _ in range(100):
            p.stepSimulation()             


class R2D2(FixedBaseRobot):
    def __init__(self, base_position=[0.0] * 3):
        f = os.path.join(cwd, "robots", "r2d2", "r2d2.urdf")
        self.urdf_filename = f
        super().__init__(f, base_position=base_position)

class Nextage(FixedBaseRobot):
    def __init__(self, base_position=[0.0, 0.0, 0.85]):
        f = os.path.join(cwd, "robots", "nextage", "nextage.urdf")
        self.urdf_file_name = f
        super().__init__(f, base_position=base_position)

class KukaLWR(FixedBaseRobot):
    def __init__(self, base_position=[0.0] * 3):
        f = os.path.join(cwd, "robots", "kuka_lwr", "kuka_lwr.urdf")
        self.urdf_filename = f
        super().__init__(f, base_position=base_position)


class KukaLBR(FixedBaseRobot):
    def __init__(self, base_position=[0.0] * 3):
        # Process xacro
        xacro_filename = os.path.join(cwd, "robots", "kuka_lbr", "med7.urdf.xacro")
        import xacro
        from io import StringIO

        try:
            urdf_string = xacro.process(xacro_filename)
        except AttributeError:
            xml = xacro.process_file(xacro_filename)
            out = StringIO()
            xml.writexml(out)
            urdf_string = out.getvalue()

        self.urdf_string = urdf_string
        urdf_filename = os.path.join(cwd, "robots", "kuka_lbr", "kuka_lbr.urdf")
        with open(urdf_filename, "w") as f:
            f.write(urdf_string)

        # Load Kuka LBR
        super().__init__(urdf_filename, base_position=base_position)

        # Remove urdf file
        os.remove(urdf_filename)


def main(gui=True):
    hz = 250
    dt = 1.0 / float(hz)
    pb = PyBullet(dt, gui=gui)
    # robot = KukaLWR()
    # robot = KukaLBR()
    # robot = R2D2([0, 0, 0.5])
    # robot = Nextage()
    robot = Fetch(fix_base=False)
    robot.retract()

    q0 = np.zeros(robot.ndof)
    qF = np.random.uniform(-np.pi, np.pi, size=(robot.ndof,))

    alpha = 0.0

    pb.start()
    while 1:
        robot.cmd_wheel_velocities(velocities=[1, 1])
        time.sleep(0.01)

    while alpha < 1.0:
        q = (1.0 - alpha) * q0 + alpha * qF
        robot.cmd(q)
        time.sleep(dt * float(gui))
        alpha += 0.05 * dt

    pb.stop()
    pb.close()

    return 0


if __name__ == "__main__":
    main()
