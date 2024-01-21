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
        self.i_min = np.array([0, 0, 0], dtype=np.int32)
        self.i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int32)   
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
        idx = ((rel_pos - self.origin) / self.grid_resolution).astype(int)
        return np.clip(idx, self.i_min, self.i_max)        

    # Evaluate numerically
    def eval(self, arg):
        rel_pos = np.array(arg[0]).reshape(1, 3)
        idxes = self.rel_pos_to_idxes(rel_pos)
        assert idxes.shape[0] == rel_pos.shape[0]
        offsets = idxes[:, 2] + self.field_shape[2] * (idxes[:, 1] + self.field_shape[1] * idxes[:, 0])
        f = self.data[offsets]
        return [f]

    def has_jacobian(self): return True
    def get_jacobian(self, name, inames, onames, opts):
        # You are required to keep a reference alive to the returned Callback object
        self.jac_callback = JacFun(name, self.data, self.origin, self.grid_resolution, self.field_shape)
        return self.jac_callback
    

class JacFun(Callback):
    def __init__(self, name, data, origin, grid_resolution, field_shape, opts={}):
        Callback.__init__(self)
        self.data = data
        self.origin = origin
        self.grid_resolution = grid_resolution
        self.nx = field_shape[0]
        self.ny = field_shape[1]
        self.nz = field_shape[2]
        self.i_min = np.array([0, 0, 0], dtype=np.int32)
        self.i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int32)           
        self.field_shape = field_shape         
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
        idx = ((rel_pos - self.origin) / self.grid_resolution).astype(int)
        return np.clip(idx, self.i_min, self.i_max)                

    # Evaluate numerically
    def eval(self, arg):
        rel_pos = np.array(arg[0]).reshape(1, 3)
        idxes = self.rel_pos_to_idxes(rel_pos)
        neighbor1 = np.clip(idxes + 1, self.i_min, self.i_max)
        neighbor2 = np.clip(idxes - 1, self.i_min, self.i_max)

        offsets1 = idxes[:, 2] + self.field_shape[2] * (idxes[:, 1] + self.field_shape[1] * neighbor1[:, 0])
        offsets2 = idxes[:, 2] + self.field_shape[2] * (idxes[:, 1] + self.field_shape[1] * neighbor2[:, 0])
        dx = (
            self.data[offsets1] - self.data[offsets2]
        ) / (2 * self.grid_resolution)

        offsets1 = idxes[:, 2] + self.field_shape[2] * (neighbor1[:, 1] + self.field_shape[1] * idxes[:, 0])
        offsets2 = idxes[:, 2] + self.field_shape[2] * (neighbor2[:, 1] + self.field_shape[1] * idxes[:, 0])
        dy = (
            self.data[offsets1] - self.data[offsets2]
        ) / (2 * self.grid_resolution)

        offsets1 = neighbor1[:, 2] + self.field_shape[2] * (idxes[:, 1] + self.field_shape[1] * idxes[:, 0])
        offsets2 = neighbor2[:, 2] + self.field_shape[2] * (idxes[:, 1] + self.field_shape[1] * idxes[:, 0])
        dz = (
            self.data[offsets1] - self.data[offsets2]
        ) / (2 * self.grid_resolution)
        return [np.stack([dx, dy, dz], axis=-1)]
    
    def has_jacobian(self): return True
    def get_jacobian(self, name, inames, onames, opts):
        # You are required to keep a reference alive to the returned Callback object
        self.hes_callback = HesFun(name, self.data, self.origin, self.grid_resolution, self.field_shape)
        return self.hes_callback
    

class HesFun(Callback):
    def __init__(self, name, data, origin, grid_resolution, field_shape, opts={}):
        Callback.__init__(self)
        self.data = data
        self.origin = origin
        self.grid_resolution = grid_resolution
        self.nx = field_shape[0]
        self.ny = field_shape[1]
        self.nz = field_shape[2]
        self.field_shape = field_shape
        self.i_min = np.array([0, 0, 0], dtype=np.int32)
        self.i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int32)           
        self.construct(name, opts)

    def get_n_in(self): return 3
    def get_n_out(self): return 2

    def get_sparsity_in(self, i):
        if i == 0: # nominal input 1
            return Sparsity.dense(3, 1)
        elif i == 1: # nominal output
            return Sparsity.dense(1, 1)
        elif i == 2:
            return Sparsity.dense(1, 3)
        
    def get_sparsity_out(self, i):
        if i == 0:
            return Sparsity.dense(3, 3)
        elif i == 1:
            return Sparsity.dense(3, 1)

    def rel_pos_to_idxes(self, rel_pos):
        idx = ((rel_pos - self.origin) / self.grid_resolution).astype(int)
        return np.clip(idx, self.i_min, self.i_max)
    
    def get_value(self, idxes):
        idxes = np.clip(idxes, self.i_min, self.i_max)
        offsets = idxes[:, 2] + self.field_shape[2] * (idxes[:, 1] + self.field_shape[1] * idxes[:, 0])
        return self.data[offsets]

    # Evaluate numerically
    def eval(self, arg):
        rel_pos = np.array(arg[0]).reshape(1, 3)
        initial = self.rel_pos_to_idxes(rel_pos)
        n = 3
        output = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                ei = np.zeros(n, dtype=np.int32)
                ei[i] = 1
                ej = np.zeros(n, dtype=np.int32)
                ej[j] = 1
                f1 = self.get_value(initial + ei + ej)
                f2 = self.get_value(initial + ei - ej)
                f3 = self.get_value(initial - ei + ej)
                f4 = self.get_value(initial - ei - ej)
                numdiff = (f1-f2-f3+f4) / (4*self.grid_resolution*self.grid_resolution)
                output[i, j] = numdiff
                output[j, i] = numdiff
        return [output, np.zeros((n, 1))]        


if __name__ == "__main__":
  
    # Use the function
    origin = np.array([-0.2, -1.3, -0.2])
    print(origin.shape)
    field_shape = (75, 130, 75)
    data = np.random.rand(*field_shape).flatten()
    grid_resolution = 0.02

    f = SDFCallback('f', data, origin, grid_resolution, field_shape)
    points = np.random.rand(3, 5)
    print(points)
    res = f(points)
    print(res, res.shape)

    x = MX.sym("x", 3)
    jac = jacobian(f(x), x)
    J = Function('J', [x], [jac])
    res_j = J(points)
    print(res_j, res_j.shape)

    hes = jacobian(jac, x)
    H = Function('H', [x], [hes])
    res_h = H(points)
    print(res_h, res_h.shape)    