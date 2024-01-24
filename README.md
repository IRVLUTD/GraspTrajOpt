# GraspTrajOpt
Trajectory optimization for grasping


# Note
- If joint limits are not presented in the urdf file, urdf_parser_py.urdf will set the joint limits to 0s. Make sure every joint has limits in the urdf file.

## Example Usage
  
1. Run GTO planning with SceneReplica in PyBullet:
   ```Shell
   cd examples/
   python pybullet_gto_planning.py
   ```

![](./pics/example.png)
