import hj_reachability as hj
import hj_reachability.shapes as shp
import jax.numpy as jnp
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go

avoid_dynamics = hj.systems.Bicycle7D(min_ddelta=-26,
                                      max_ddelta=+26,
                                      min_acc=-3.6,
                                      max_acc=+3.6).with_mode('avoid')

min_bounds = np.array([-6., -6., -np.pi/4, -10., -1., -np.pi/3, -np.pi/4])
max_bounds = np.array([+6., +6., +np.pi/4, +10., +1., +np.pi/3, +np.pi/4])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (5, 5, 3, 7, 3, 3, 3))#,
                                                            #    periodic_dims=2)

solver_settings = hj.SolverSettings.with_accuracy("low")
horizon = .2
t_step = .2
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
                          shp.lower_half_space(grid, 1, +2.),
                          shp.upper_half_space(grid, 1, -2.),
                          shp.lower_half_space(grid, 2, np.pi/4. + 0.1),
                          shp.upper_half_space(grid, 2, -np.pi/4. - 0.1),
                          shp.lower_half_space(grid, 3, 0.))

result = compute_brs(time_frame, solver_settings, avoid_dynamics, grid, target)
print(result.shape)
res_proj = shp.project_onto(result[:,:,:,2,:,0,0,:], 0, 1, 2, 3)

grid_proj = grid.states[..., [0,1,3]]
grid_proj = grid_proj[:,:,2,:,0,0,1,:]
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
