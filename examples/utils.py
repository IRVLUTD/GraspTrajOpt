import json
import numpy as np
from transforms3d.quaternions import quat2mat


def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


def parse_grasps(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    grasps = data["grasps"]

    n = len(grasps)
    poses_grasp = np.zeros((n, 4, 4), dtype=np.float32)
    for i in range(n):
        pose = grasps[i]["pose"]
        rot = pose[3:]
        trans = pose[:3]
        RT = ros_qt_to_rt(rot, trans)
        poses_grasp[i, :, :] = RT
    return poses_grasp
