import hj_reachability as hj
import hj_reachability.shapes as shp
import jax.numpy as jnp
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from hj_tools import compute_brs

avoid_dynamics = hj.systems.Bicycle7D(min_controls=[-26, -3.6],
                                      max_controls=[+26, +3.6]).with_mode('avoid')

min_bounds = np.array([-6., -6., -np.pi/4, -10., -1., -np.pi/3, -np.pi/4])
max_bounds = np.array([+6., +6., +np.pi/4, +10., +1., +np.pi/3, +np.pi/4])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (11, 11, 5, 7, 5, 5, 5))#,
                                                            #    periodic_dims=2)

solver_settings = hj.SolverSettings.with_accuracy("low")
t_step = .1

target = shp.intersection(shp.lower_half_space(grid, 0, +6.),
                          shp.upper_half_space(grid, 0, -6.),
                          shp.lower_half_space(grid, 1, +2.),
                          shp.upper_half_space(grid, 1, -2.),
                          # shp.lower_half_space(grid, 2, np.pi/4. + 0.1),
                          # shp.upper_half_space(grid, 2, -np.pi/4. - 0.1),
                          shp.lower_half_space(grid, 3, 0.))

result = compute_brs(solver_settings, avoid_dynamics, grid, target, t_step)
print(result.shape)
res_proj = -shp.project_onto(-result[:,:,4,:,0,0,:], 0, 1, 2)

grid_proj = grid.states[..., [0,1,3]]
grid_proj = grid_proj[:,:,4,:,0,0,1,:]
# breakpoint()

fig = go.Figure(data=go.Isosurface(x=grid_proj[..., 0].ravel(),
                             y=grid_proj[..., 1].ravel(),
                             z=grid_proj[..., 2].ravel(),
                             value=res_proj.ravel(),
                             colorscale="jet",
                             isomin=0,
                             surface_count=1,
                             isomax=0))

fig.show()
