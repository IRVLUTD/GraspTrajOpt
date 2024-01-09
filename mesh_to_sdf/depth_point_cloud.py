import numpy as np
from sklearn.neighbors import KDTree
import math
import pyrender
import time
import _init_paths
import optas
import casadi as cs


class SignedDensityField(object):
    """Data is stored in the following way
        data[x, y, z]
    update to integrate torch
    """

    def __init__(self, data, origin, delta):
        self.nx, self.ny, self.nz = data.shape
        self.data = cs.DM(data.flatten().tolist())
        self.origin = origin.reshape((1, 3))
        self.delta = delta
        self.min_coords = origin
        self.max_coords = self.origin + delta * np.array(data.shape)

    def _rel_pos_to_idxes(self, rel_pos):
        n = rel_pos.shape[0]
        origin = np.repeat(self.origin, n, axis=0)
        idx = optas.floor((rel_pos - origin) / self.delta)
        return idx

    def get_distance(self, rel_pos):
        idxes = self._rel_pos_to_idxes(rel_pos)
        assert idxes.shape[0] == rel_pos.shape[0]
        # offset = n_3 + N_3 * (n_2 + N_2 * n_1)
        # https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays
        offset = idxes[:, 2] + self.nz * (idxes[:, 1] + self.ny + idxes[:, 0])
        return self.data[offset]


class DepthPointCloud:
    def __init__(self, depth, intrinsic_matrix, camera_pose, mask=None, threshold=1.5):
        self.depth = depth
        self.intrinsic_matrix = intrinsic_matrix
        self.camera_pose = camera_pose
        self.mask = mask
        self.width = depth.shape[1]
        self.height = depth.shape[0]
        self.threshold = threshold
        self.grid_resolution = 0.02

        # backproject to camera
        pc = self.backproject_camera(depth, intrinsic_matrix)

        # transform points to world
        pc_base = camera_pose[:3, :3] @ pc + camera_pose[:3, 3].reshape((3, 1))
        self.points = pc_base.T
        self.compute_sdf_from_points(self.points)


    def compute_sdf_from_points(self, points):
        """
        Environment perceived points as N x 3 in the robot base coordinates
        we use nearest point distance to represent SDF for simplicity
        """
        
        self.workspace_bounds = np.stack((points.min(0), points.max(0)), axis=1)
        margin = 0.24
        self.workspace_points = np.array(np.meshgrid(
                                np.arange(self.workspace_bounds[0][0] - margin, self.workspace_bounds[0][1] + margin, self.grid_resolution),
                                np.arange(self.workspace_bounds[1][0] - margin, self.workspace_bounds[1][1] + margin, self.grid_resolution),
                                np.arange(self.workspace_bounds[2][0] - margin, self.workspace_bounds[2][1] + margin, self.grid_resolution),
                                indexing='ij'))
        print('computing SDF from point cloud ...', self.workspace_bounds, self.workspace_points.shape)

        # set sdf manually
        workspace_points_flattened = self.workspace_points.reshape((3, -1)).T
        s = time.time()
        self.kd_tree = KDTree(self.points)
        dists, _ = self.kd_tree.query(workspace_points_flattened)
        self.sdf = SignedDensityField(dists.reshape(self.workspace_points.shape[1:]), self.workspace_bounds[:,0] - margin, self.grid_resolution)      
        print('process point SDF time:', time.time() - s)
        

    def get_random_surface_points(self, count):
        indices = np.random.choice(self.points.shape[0], count)
        return self.points[indices, :]


    def backproject_camera(self, im_depth, K):  
        Kinv = np.linalg.inv(K)

        width = im_depth.shape[1]
        height = im_depth.shape[0]
        depth = im_depth.astype(np.float32, copy=True).flatten()
        mask = (depth != 0) & (depth < self.threshold)

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

        # backprojection
        R = Kinv.dot(x2d.transpose())
        X = np.multiply(
            np.tile(depth.reshape(1, width * height), (3, 1)), R
        )
        return X[:, mask]
    

    # query points are in world frame
    def get_sdf(self, query_points):
        distances, indices = self.kd_tree.query(query_points)
        distances = distances.astype(np.float32).reshape(-1)
        inside = ~self.is_outside(query_points)
        distances[inside] *= -1
        return distances


    def get_sdf_in_batches(self, query_points, batch_size=1000000):
        if query_points.shape[0] <= batch_size:
            return self.get_sdf(query_points)

        n_batches = int(math.ceil(query_points.shape[0] / batch_size))
        batches = [
            self.get_sdf(points)
            for points in np.array_split(query_points, n_batches)
        ]
        return np.concatenate(batches)


    def show(self):
        # compute sdf for sampled points
        query_points = []
        surface_sample_count = 10000
        surface_points = self.get_random_surface_points(surface_sample_count)
        query_points.append(surface_points + np.random.normal(scale=0.025, size=(surface_sample_count, 3)))
        query_points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))        
        query_points = np.concatenate(query_points).astype(np.float32)
        sdf = self.get_sdf(query_points)

        # visualization
        colors = np.zeros(query_points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1
        cloud = pyrender.Mesh.from_points(query_points, colors=colors)
        scene = pyrender.Scene()
        scene.add(cloud)
        scene.add(pyrender.Mesh.from_points(self.points))
        pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


    def is_outside(self, points):
        # project points to camera view
        RT = np.linalg.inv(self.camera_pose)
        pc = points.T
        pc_camera = RT[:3, :3] @ pc + RT[:3, 3].reshape((3, 1))
        x2d = self.intrinsic_matrix @ pc_camera
        x2d[0, :] /= x2d[2, :]
        x2d[1, :] /= x2d[2, :]
        pixels = x2d[:2].T.astype(int)

        # This only has an effect if the camera is inside the model
        in_viewport = (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0) & (pixels[:, 0] < self.width) & (pixels[:, 1] < self.height)
        pc_camera = pc_camera.T
        result = np.zeros(points.shape[0], dtype=bool)
        result[in_viewport] = pc_camera[in_viewport, 2] < self.depth[pixels[in_viewport, 1], pixels[in_viewport, 0]]
        return result