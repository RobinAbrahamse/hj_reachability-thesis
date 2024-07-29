import hj_reachability as hj
import hj_reachability.shapes as shp
import numpy as np
import numpy as np
from subsystem import Subsystem
import hj_tools

dynamics = hj.systems.Y_vx_vy_d(min_controls=[-0.2, -16],
                                      max_controls=[+0.2, +16],
                                      min_disturbances=[-np.pi, -np.pi/3.],
                                      max_disturbances=[+np.pi, +np.pi/3.]).with_mode('avoid')

grid_mins = np.array([-4., -10., -3, -np.pi/4])
grid_maxs = np.array([+4., +10., +3, +np.pi/4])
grid_res = (21, 21, 21, 13)
time_step = 0.3
target_mins = [(0, -2.)]
target_maxs = [(0, +2.), (1, 0.)]

Y_vx_vy_d = Subsystem(dynamics, grid_mins, grid_maxs, grid_res, time_step, target_mins, target_maxs)

Y_vx_vy_d.step()
result = Y_vx_vy_d.combine()

print(result.shape)
res_proj = -shp.project_onto(-result[:,:,:,:], 0, 1, 2)
# res_proj = shp.project_onto(result[:,:,:,:], 0, 1, 2)

grid_proj = Y_vx_vy_d.grid.states[..., [0,1,2]]
grid_proj = grid_proj[:,:,:,0]

# np.save('hj_data/YVyVxD_grid', grid.states)
# np.save('hj_data/YVxVyD_result', result)

hj_tools.plot_set_3D(grid_proj[..., 0].ravel(), 
            grid_proj[..., 1].ravel(), 
            grid_proj[..., 2].ravel(), 
            res_proj.ravel(),
            ("x", "v_x", "v_y"))