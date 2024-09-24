import hj_reachability as hj
import hj_reachability.shapes as shp
import jax.numpy as jnp
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from hj_tools import compute_brs

avoid_dynamics = hj.systems.Bicycle7D(min_controls=[-26, -3.6],
                                      max_controls=[+26, +3.6]).with_mode('avoid')

grid_dims = np.array([22, 22, 12, 12, 9, 9, 7])
grid_mins = np.array([-4., -4., 0.,      +0.1, -2., -np.pi, -np.pi/4])
grid_maxs = np.array([+4., +4., 2*np.pi, +3.6, +2., +np.pi, +np.pi/4])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(grid_mins, grid_maxs),
                                                               grid_dims,
                                                               periodic_dims=2)

solver_settings = hj.SolverSettings.with_accuracy("low")
t_step = .1

target = shp.union(
            shp.intersection(shp.lower_half_space(grid, 0, -.8),
               shp.upper_half_space(grid, 0, +.8),
               shp.lower_half_space(grid, 1, -.8),
               shp.upper_half_space(grid, 3, 0.)),
            shp.intersection(shp.lower_half_space(grid, 0, -.8),
               shp.upper_half_space(grid, 0, +.8),
               shp.upper_half_space(grid, 1, +.8),
               shp.upper_half_space(grid, 3, 0.))
         )

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
