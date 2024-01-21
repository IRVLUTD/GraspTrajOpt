from casadi import *
import numpy as np

class SDFCallback(Callback):
    def __init__(self, name, data, origin, grid_resolution, field_shape, opts={}):
        Callback.__init__(self)
        self.name = name
        self.data = data
        self.origin = origin
        self.grid_resolution = grid_resolution
        self.field_shape = field_shape
        self.nx = field_shape[0]
        self.ny = field_shape[1]
        self.nz = field_shape[2]
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return Sparsity.dense(3, 1)
    
    def get_sparsity_out(self, i):
        return Sparsity.dense(1, 1)

    # Initialize the object
    def init(self):
        print('initializing SDF callback')

    def rel_pos_to_idxes(self, rel_pos):
        i_min = np.array([0, 0, 0], dtype=np.int32)
        i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int32)
        idx = ((rel_pos - self.origin) / self.grid_resolution).astype(int)
        return np.clip(idx, i_min, i_max)        

    # Evaluate numerically
    def eval(self, arg):
        rel_pos = np.array(arg[0]).reshape(1, 3)
        print('in F', rel_pos)
        idxes = self.rel_pos_to_idxes(rel_pos)
        assert idxes.shape[0] == rel_pos.shape[0]
        f = self.data[idxes[..., 0], idxes[..., 1], idxes[..., 2]]
        return [f]

    def has_jacobian(self): return True
    def get_jacobian(self, name, inames, onames, opts):
        class JacFun(Callback):
            def __init__(self, data, origin, grid_resolution, field_shape, opts={}):
                Callback.__init__(self)
                self.data = data
                self.origin = origin
                self.grid_resolution = grid_resolution
                self.nx = field_shape[0]
                self.ny = field_shape[1]
                self.nz = field_shape[2]                
                self.construct(name, opts)

            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self, i):
                if i == 0: # nominal input
                    return Sparsity.dense(3, 1)
                elif i == 1: # nominal output
                    return Sparsity.dense(1, 1)

            def get_sparsity_out(self, i):
                return Sparsity.dense(1, 3)

            def rel_pos_to_idxes(self, rel_pos):
                i_min = np.array([0, 0, 0], dtype=np.int32)
                i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int32)
                idx = ((rel_pos - self.origin) / self.grid_resolution).astype(int)
                return np.clip(idx, i_min, i_max)                

            # Evaluate numerically
            def eval(self, arg):
                rel_pos = np.array(arg[0]).reshape(1, 3)
                print('in J', rel_pos)
                idxes = self.rel_pos_to_idxes(rel_pos)
                i_min = np.array([0, 0, 0], dtype=np.int32)
                i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int32)
                neighbor1 = np.clip(idxes + 1, i_min, i_max)
                neighbor2 = np.clip(idxes - 1, i_min, i_max)
                dx = (
                    self.data[neighbor1[..., 0], idxes[..., 1], idxes[..., 2]]
                    - self.data[neighbor2[..., 0], idxes[..., 1], idxes[..., 2]]
                ) / (2 * self.grid_resolution)

                dy = (
                    self.data[idxes[..., 0], neighbor1[..., 1], idxes[..., 2]]
                    - self.data[idxes[..., 0], neighbor2[..., 1], idxes[..., 2]]
                ) / (2 * self.grid_resolution)

                dz = (
                    self.data[idxes[..., 0], idxes[..., 1], neighbor1[..., 2]]
                    - self.data[idxes[..., 0], idxes[..., 1], neighbor2[..., 2]]
                ) / (2 * self.grid_resolution)
                print('in J', np.stack([dx, dy, dz], axis=-1))
                return [np.stack([dx, dy, dz], axis=-1)]

        # You are required to keep a reference alive to the returned Callback object
        self.jac_callback = JacFun(self.data, self.origin, self.grid_resolution, self.field_shape)
        return self.jac_callback  


if __name__ == "__main__":
  
    # Use the function
    origin = np.array([-0.2, -1.3, -0.2])
    print(origin.shape)
    field_shape = (75, 130, 75)
    data = np.random.rand(*field_shape)
    grid_resolution = 0.02

    f = SDFCallback('f', data, origin, grid_resolution, field_shape)
    points = np.random.rand(3, 5)
    print(points)
    res = f(points)
    print(res, res.shape)

    x = MX.sym("x", 3)
    J = Function('J', [x], [jacobian(f(x), x)])
    res_j = J(points)
    print(res_j, res_j.shape)