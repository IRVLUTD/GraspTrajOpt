import os
import sys
import argparse
import pathlib

cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
sys.path.append(os.path.join(cwd, ".."))
import optas
from optas.visualize import Visualizer


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="fetch",
        help="robot model name",
    )   
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_args()
    model = args.model

    if model == "lwr":
        urdf_filename = os.path.join(cwd, "robots", "kuka_lwr", "kuka_lwr.urdf")
        robot_model = optas.RobotModel(urdf_filename=urdf_filename)

    elif model == "lbr":
        xacro_filename = os.path.join(cwd, "robots", "kuka_lbr", "med7.urdf.xacro")
        robot_model = optas.RobotModel(xacro_filename=xacro_filename)

    elif model == "fetch":
        urdf_filename = os.path.join(cwd, "robots", "fetch", "fetch_gripper.urdf")
        print(urdf_filename)
        robot_model = optas.RobotModel(urdf_filename=urdf_filename)

    vis = Visualizer(camera_position=[3, 3, 3])
    vis.grid_floor()
    q = robot_model.get_random_joint_positions()
    print(robot_model.actuated_joint_names)
    print(q)
    
    vis.robot(
        robot_model,
        base_position=[0.0, 0, 0],
        base_orientation=[0, 0, 0],
        euler_degrees=True,
        q=q,
        show_links=True,
        display_link_names=True,
        link_names_alpha=0.4,
    )

    save = False
    if save:
        vis.save('vis.png')
    else:
        vis.start()