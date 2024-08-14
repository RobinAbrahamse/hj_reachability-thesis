import hj_reachability as hj
import hj_reachability.shapes as shp
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import hj_tools

avoid_dynamics = hj.systems.yaw_w(min_disturbances=[-10., -7., -jnp.pi/2],
                                  max_disturbances=[-1.,  +7., +jnp.pi/2]).with_mode('avoid')

min_bounds = np.array([0.,       -jnp.pi/2])
max_bounds = np.array([2*jnp.pi, +jnp.pi/2])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (21, 21),
                                                               periodic_dims=0)

solver_settings = hj.SolverSettings.with_accuracy("low")
t_step = .25

target = shp.intersection(shp.lower_half_space(grid, 0, 7*jnp.pi/4.),
                          shp.upper_half_space(grid, 0, jnp.pi/4.))

# print(grid.coordinate_vectors[0])
# for idx, i in enumerate(grid.coordinate_vectors[0]):
#     for idy, j in enumerate(grid.coordinate_vectors[1]):
#         state = jnp.array([i,j])
#         grads = grid.grad_values(target)[idx,idy]
#         _, dstrbs = avoid_dynamics.optimal_control_and_disturbance(state, 0., grads)
#         t = avoid_dynamics(state, jnp.array([0.]), dstrbs, 0.)
#         print(f"Disturbances: {dstrbs}, derivs: {t}")
# exit()

result = hj_tools.compute_brs(solver_settings, avoid_dynamics, grid, target, t_step)
print(result)

hj_tools.plot_value_2D(grid.coordinate_vectors[0], grid.coordinate_vectors[1], result[:, :].T, axes_labels=["yaw", "w"])
# breakpoint()