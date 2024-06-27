import hj_reachability as hj
import hj_reachability.shapes as shp
import jax.numpy as jnp
import numpy as np
from hj_plot import plot_set_3D

# avoid_dynamics = hj.systems.X_vx_vy_d(min_ddelta=-26,
avoid_dynamics = hj.systems.Y_vx_vy_d(min_ddelta=-26,
                                      max_ddelta=+26,
                                      min_acc=-3.6,
                                      max_acc=+3.6,
                                      min_disturbances=[0., 0.],
                                      max_disturbances=[jnp.pi/2, jnp.pi/3]).with_mode('avoid')

min_bounds = np.array([-4., -10., -3, -np.pi/4])
max_bounds = np.array([+4., +10., +3, +np.pi/4])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (3, 11, 11, 3))#,
                                                            #    periodic_dims=2)

solver_settings = hj.SolverSettings.with_accuracy("low")
horizon = 1.
t_step = 1.
e = 1e-5
time_frame = np.arange(start=0., stop=horizon + e, step=t_step)

def compute_brs(time_frame, solver_settings, dynamics, grid, target):
    # target = shp.make_tube(time_frame, target)
    # values = hj.solve(solver_settings, dynamics, grid,
    #                 time_frame, target)
    values = hj.step(solver_settings, dynamics, grid, 0., values, t_step):
    values = np.asarray(values)
    return values#np.flip(values, axis=0)

# breakpoint()

# target = shp.intersection(shp.lower_half_space(grid, 0, +6.),
#                           shp.upper_half_space(grid, 0, -6.),
target = shp.intersection(shp.lower_half_space(grid, 0, +2.),
                          shp.upper_half_space(grid, 0, -2.),
                          shp.lower_half_space(grid, 1, 0.))

result = compute_brs(time_frame, solver_settings, avoid_dynamics, grid, target)
print(result.shape)
res_proj = shp.project_onto(result[:,:,:,:,:], 0, 1, 2, 3)

grid_proj = grid.states[..., [0,1,2]]
grid_proj = grid_proj[:,:,:,0]
# breakpoint()

jnp.save('hj_data/YVyVxD_grid', grid.states)
jnp.save('hj_data/YVxVyD_result', result)

plot_set_3D(grid_proj[..., 0].ravel(), grid_proj[..., 1].ravel(), grid_proj[..., 2].ravel(), res_proj[1].ravel())

# breakpoint()
