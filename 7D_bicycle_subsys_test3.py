import hj_reachability as hj
import hj_reachability.shapes as shp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import hj_tools

avoid_dynamics = hj.systems.vx_vy_w_d(min_controls=[-.2, -16],
                                      max_controls=[+.2, +16]).with_mode('avoid')

min_bounds = np.array([-10., -3., -np.pi/3, -np.pi/4])
max_bounds = np.array([+10., +3., +np.pi/3, +np.pi/4])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (21, 13, 11, 9))

solver_settings = hj.SolverSettings.with_accuracy("low")
t_step = .5

target = shp.intersection(shp.lower_half_space(grid, 0, 0.),
                          shp.lower_half_space(grid, 1, 1),
                          shp.upper_half_space(grid, 1, -1),)

result = hj_tools.compute_brs(solver_settings, avoid_dynamics, grid, target, t_step)

result = np.minimum(target, result)

print(result.shape)

res_proj = -shp.project_onto(-result[:,:,:,:], 0, 1, 2)
# res_proj = shp.project_onto(result[:,:,:,:], 0, 1, 2)

grid_proj = grid.states[..., [0,1,2]]
grid_proj = grid_proj[:,:,:,0,:]

hj_tools.plot_set_3D(grid_proj[..., 0].ravel(), 
            grid_proj[..., 1].ravel(), 
            grid_proj[..., 2].ravel(), 
            res_proj.ravel(),
            ("v_x", "v_y", "w"))

# hj_tools.plot_value_2D(grid.coordinate_vectors[0], 
#                       grid.coordinate_vectors[1], 
#                       res_proj[:, :, 1].T, ("v_x", "v_y"))

# hj_tools.plot_value_2D(grid.coordinate_vectors[0], 
#                       grid.coordinate_vectors[1], 
#                       res_proj[:, :, 4].T, ("v_x", "v_y"))

# hj_tools.plot_value_2D(grid.coordinate_vectors[0], 
#                       grid.coordinate_vectors[1], 
#                       res_proj[:, :, 7].T, ("v_x", "v_y"))
