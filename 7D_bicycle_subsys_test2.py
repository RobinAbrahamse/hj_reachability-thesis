import hj_reachability as hj
import hj_reachability.shapes as shp
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from hj_tools import compute_brs

avoid_dynamics = hj.systems.Y_yaw(min_disturbances=[-5., -0., -jnp.pi/3],
                                  max_disturbances=[+5., +0., +jnp.pi/3]).with_mode('avoid')

min_bounds = np.array([-10., 0])
max_bounds = np.array([+10., 2*jnp.pi])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (21, 21),
                                                               periodic_dims=1)

solver_settings = hj.SolverSettings.with_accuracy("low")
t_step = .5

target = shp.intersection(shp.lower_half_space(grid, 0, +6.),
                          shp.upper_half_space(grid, 0, -6.),
                          shp.lower_half_space(grid, 1, 7*jnp.pi/4.),
                          shp.upper_half_space(grid, 1, jnp.pi/4.))

result = compute_brs(solver_settings, avoid_dynamics, grid, target, t_step)
print(result.shape)

plt.figure(figsize=(13, 8))
plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], result[:, :].T)
plt.colorbar()
plt.show()
