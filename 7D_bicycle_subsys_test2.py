import hj_reachability as hj
import hj_reachability.shapes as shp
import jax.numpy as jnp
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go

# avoid_dynamics = hj.systems.X_vx_vy_d(min_ddelta=-26,
avoid_dynamics = hj.systems.X_yaw(min_disturbances=[-5., -3., -jnp.pi/3],
                                  max_disturbances=[+5., +3., +jnp.pi/3]).with_mode('avoid')

min_bounds = np.array([-10., -jnp.pi])
max_bounds = np.array([+10., +jnp.pi])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (21, 21),
                                                               periodic_dims=1)

solver_settings = hj.SolverSettings.with_accuracy("low")
horizon = 1.
t_step = 1.
e = 1e-5
time_frame = np.arange(start=0., stop=horizon + e, step=t_step)

def compute_brs(time_frame, solver_settings, dynamics, grid, target):
    # target = shp.make_tube(time_frame, target)
    values = hj.solve(solver_settings, dynamics, grid,
                    time_frame, target)
    values = np.asarray(values)
    return np.flip(values, axis=0)

# breakpoint()

target = shp.intersection(shp.lower_half_space(grid, 0, +6.),
                          shp.upper_half_space(grid, 0, -6.),
                          shp.lower_half_space(grid, 1, -jnp.pi/4.),
                          shp.upper_half_space(grid, 1, +jnp.pi/4.))

result = compute_brs(time_frame, solver_settings, avoid_dynamics, grid, target)
print(result.shape)
res_proj = result
# res_proj = shp.project_onto(result[:,:,:], 0, 1, 2, 3)

grid_proj = grid.states[..., [0,1,2]]
# grid_proj = grid_proj[:,:,:,0]
# breakpoint()

fig = go.Figure(data=go.Isosurface(x=grid_proj[..., 0].ravel(),
                             y=grid_proj[..., 1].ravel(),
                             z=grid_proj[..., 2].ravel(),
                             value=res_proj[1].ravel(),
                             colorscale="jet",
                             isomin=0,
                             surface_count=1,
                             isomax=0))

fig.show()

# breakpoint()
