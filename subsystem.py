import hj_reachability as hj
import hj_reachability.shapes as shp
import jax.numpy as jnp
import numpy as np

class Subsystem(object):
    def __init__(self, dynamics, grid_mins, grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=None, solver_settings="low"):
        self.solver_settings = hj.SolverSettings.with_accuracy(solver_settings)
        self.dynamics = dynamics
        min_bounds = np.array(grid_mins)
        max_bounds = np.array(grid_maxs)
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
                                        hj.sets.Box(min_bounds, max_bounds),
                                        grid_res,
                                        periodic_dims=periodic_dims)
        self.time_step = time_step
        target_sets = [shp.upper_half_space(self.grid, a, b) for a, b in target_mins] + [shp.lower_half_space(self.grid, a, b) for a, b in target_maxs]
        initial_target = shp.intersection(*target_sets)
        self.result_list = [initial_target]

    def set_disturbances(self, min_disturbances, max_disturbances):
        disturbance_space = hj.sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        self.dynamics.disturbance_space = disturbance_space

    def step(self):
        result = self._compute_brs(self.solver_settings, self.dynamics, self.grid, self.result_list[-1], self.time_step)
        result = np.clip(result, -1e5, +1e5)
        self.result_list.append(result)
        return result
    
    def combine(self):
        n = len(self.result_list)
        result = np.copy(self.result_list[0])
        for i in range(1, n):
            result = np.minimum(result, self.result_list[i])
        return result
    
    def find_reach_range(self, dim):
        dim_value = shp.project_onto(self.combine(), dim)
        negative_idxs = np.where(dim_value < 0.)[0]
        range = self.grid.coordinate_vectors[dim][negative_idxs]
        if np.any(np.diff(negative_idxs) != 1):
            print("WARNING: range is not convex")
        if len(range) < 1:
            print("WARNING: no safe range found")
            return (0., 0.)
        return (range[0], range[-1])
    
    def _compute_brs(self, solver_settings, dynamics, grid, target, t):
        values = hj.step(solver_settings, dynamics, grid, 0., target, -t)
        return np.asarray(values)


