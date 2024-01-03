import json
import numpy as np
from transforms3d.quaternions import quat2mat


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
        # RT = ros_qt_to_rt(rot, trans)
        RT = np.eye(4)
        RT[:3, :3] = quat2mat(rot)
        RT[:3, 3] = trans
        poses_grasp[i, :, :] = RT
    return poses_grasp
