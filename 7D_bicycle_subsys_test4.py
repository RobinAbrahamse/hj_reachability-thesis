import hj_reachability as hj
import hj_reachability.shapes as shp
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from hj_tools import compute_brs

avoid_dynamics = hj.systems.yaw_w(min_disturbances=[-5.],
                                  max_disturbances=[+5.]).with_mode('avoid')

min_bounds = np.array([0.,       -jnp.pi/2])
max_bounds = np.array([2*jnp.pi, +jnp.pi/2])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (21, 21),
                                                               periodic_dims=0)

solver_settings = hj.SolverSettings.with_accuracy("low")
t_step = .5

target = shp.intersection(shp.lower_half_space(grid, 0, 7*jnp.pi/4.),
                          shp.upper_half_space(grid, 0, jnp.pi/4.))

result = compute_brs(solver_settings, avoid_dynamics, grid, target, t_step)
print(result.shape)

plt.figure(figsize=(13, 8))
plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], result[:, :].T)
plt.colorbar()
plt.show()
