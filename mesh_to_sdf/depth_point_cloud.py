import numpy as np
from sklearn.neighbors import KDTree
import math
import pyrender
import time
import _init_paths


class DepthPointCloud:
    def __init__(self, depth, intrinsic_matrix, camera_pose, target_mask=None, threshold=1.5):
        self.depth = depth
        self.intrinsic_matrix = intrinsic_matrix
        self.camera_pose = camera_pose
        self.target_mask = target_mask
        self.width = depth.shape[1]
        self.height = depth.shape[0]
        self.threshold = threshold

        # backproject to camera
        pc = self.backproject_camera(depth, intrinsic_matrix)

        # transform points to world
        pc_base = camera_pose[:3, :3] @ pc + camera_pose[:3, 3].reshape((3, 1))
        self.points = pc_base.T
        self.kd_tree = KDTree(self.points)
        

    def get_random_surface_points(self, count):
        indices = np.random.choice(self.points.shape[0], count)
        return self.points[indices, :]


    def backproject_camera(self, im_depth, K):  
        Kinv = np.linalg.inv(K)

        width = im_depth.shape[1]
        height = im_depth.shape[0]
        depth = im_depth.astype(np.float32, copy=True).flatten()
        if self.target_mask is not None:
            mask = (depth > 0) & (depth < self.threshold) & (self.target_mask.flatten() == 0)
        else:
            mask = (depth > 0) & (depth < self.threshold)

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
    
    
    def get_sdf_cost(self, query_points, epsilon=0.02, w_inside=1, vis=False):
        print('computing sdf cost...')
        distances, indices = self.kd_tree.query(query_points)
        distances = distances.astype(np.float32).reshape(-1)
        inside = ~self.is_outside(query_points)
        distances[inside] *= -1

        # visualization
        if vis:
            index = np.absolute(distances) < 0.03
            points_show = query_points[index]
            colors = np.zeros(points_show.shape)
            colors[distances[index] < 0, 2] = 1
            colors[distances[index] > 0, 0] = 1
            cloud = pyrender.Mesh.from_points(points_show, colors=colors)
            scene = pyrender.Scene()
            scene.add(cloud)
            scene.add(pyrender.Mesh.from_points(self.points[::100]))
            pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)

        # cost
        cost = np.zeros_like(distances)
        cost[inside] = w_inside * (-distances[inside] + epsilon / 2)
        index = (distances > 0) & (distances < epsilon)
        cost[index] = np.square(distances[index] - epsilon) / (2 * epsilon)
        print('done')
        return cost


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
        result = np.ones(points.shape[0], dtype=bool)
        result[in_viewport] = pc_camera[in_viewport, 2] < self.depth[pixels[in_viewport, 1], pixels[in_viewport, 0]]
        return result