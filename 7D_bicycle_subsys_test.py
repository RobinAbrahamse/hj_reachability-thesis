import hj_reachability as hj
import hj_reachability.shapes as shp
import numpy as np
import numpy as np
import hj_plot

avoid_dynamics = hj.systems.Y_vx_vy_d(min_controls=[-16, -0.2],
                                      max_controls=[+16, +0.2],
                                      min_disturbances=[-np.pi, -np.pi/3.],
                                      max_disturbances=[+np.pi, +np.pi/3.]).with_mode('avoid')

min_bounds = np.array([-4., -10., -3, -np.pi/4])
max_bounds = np.array([+4., +10., +3, +np.pi/4])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (21, 21, 21, 13))

solver_settings = hj.SolverSettings.with_accuracy("low")
t_step = 0.3

def compute_brs(solver_settings, dynamics, grid, target, t):
    values = hj.step(solver_settings, dynamics, grid, 0., target, -t)
    return np.asarray(values)

target = shp.intersection(shp.lower_half_space(grid, 0, +2.),
                          shp.upper_half_space(grid, 0, -2.),
                          shp.lower_half_space(grid, 1, 0.))

result = compute_brs(solver_settings, avoid_dynamics, grid, target, t_step)
# result = np.minimum(target, result)

print(result.shape)
res_proj = -shp.project_onto(-result[:,:,:,:], 0, 1, 2)

grid_proj = grid.states[..., [0,1,2]]
grid_proj = grid_proj[:,:,:,0]

np.save('hj_data/YVyVxD_grid', grid.states)
np.save('hj_data/YVxVyD_result', result)

hj_plot.plot_set_3D(grid_proj[..., 0].ravel(), 
            grid_proj[..., 1].ravel(), 
            grid_proj[..., 2].ravel(), 
            res_proj.ravel(),
            ("x", "v_x", "v_y"))