import hj_reachability as hj
import hj_reachability.shapes as shp
import numpy as np
import matplotlib.pyplot as plt
from subsystem import Subsystem
import hj_tools
from itertools import product

time_step = .1
n = 20
min_controls = [-.2, -16]
max_controls = [+.2, +16]

grid_dims = np.array([41, 41, 17, 21, 11, 11, 7])
grid_mins = np.array([-20., -20., 0.,      -10., -5., -np.pi/2, -np.pi/4])
grid_maxs = np.array([+20., +20., np.pi/2, +10., +5., +np.pi/2, +np.pi/4])

# VX_VY_W_D
vx_vy_w_d_dynamics = hj.systems.vx_vy_w_d(min_controls=min_controls,
                                          max_controls=max_controls).with_mode('avoid')
vx_vy_w_d_idxs = [3,4,5,6]
subsys_grid_mins = grid_mins[vx_vy_w_d_idxs]
subsys_grid_maxs = grid_maxs[vx_vy_w_d_idxs]
grid_res = tuple(grid_dims[vx_vy_w_d_idxs])
target_mins = []
target_maxs = [(0, 0.)]

vx_vy_w_d = Subsystem(vx_vy_w_d_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs)

# YAW_W
yaw_w_dynamics = hj.systems.yaw_w().with_mode('avoid')

yaw_w_idxs = [2,5]
subsys_grid_mins = grid_mins[yaw_w_idxs]
subsys_grid_maxs = grid_maxs[yaw_w_idxs]
grid_res = tuple(grid_dims[yaw_w_idxs])
periodic_dims = 0
target_mins = [(0, np.pi/4.)]
target_maxs = [(0, 7*np.pi/4.)]

yaw_w = Subsystem(yaw_w_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)

# X_YAW
X_yaw_dynamics = hj.systems.X_yaw().with_mode('avoid')
X_yaw_idxs = [0,2]
subsys_grid_mins = grid_mins[X_yaw_idxs]
subsys_grid_maxs = grid_maxs[X_yaw_idxs]
grid_res = tuple(grid_dims[X_yaw_idxs])
periodic_dims = 1
target_mins = [(0, -6.), (1, np.pi/4.)]
target_maxs = [(0, +6.), (1, 7*np.pi/4.)]

x_yaw = Subsystem(X_yaw_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)

# Y_YAW
Y_yaw_dynamics = hj.systems.Y_yaw().with_mode('avoid')
Y_yaw_idxs = [0,2]
subsys_grid_mins = grid_mins[Y_yaw_idxs]
subsys_grid_maxs = grid_maxs[Y_yaw_idxs]
grid_res = tuple(grid_dims[Y_yaw_idxs])
periodic_dims = 1
target_mins = [(0, -2.), (1, np.pi/4.)]
target_maxs = [(0, +2.), (1, 7*np.pi/4.)]

y_yaw = Subsystem(X_yaw_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)

# X_VX_VY_D
x_vx_vy_d_dynamics = hj.systems.X_vx_vy_d(min_controls=min_controls,
                                          max_controls=max_controls).with_mode('avoid')
x_vx_vy_d_idxs = [0,3,4,6]
subsys_grid_mins = grid_mins[x_vx_vy_d_idxs]
subsys_grid_maxs = grid_maxs[x_vx_vy_d_idxs]
grid_res = tuple(grid_dims[x_vx_vy_d_idxs])
target_mins = [(0, -6.)]
target_maxs = [(0, +6.), (1, 0.)]

x_vx_vy_d = Subsystem(x_vx_vy_d_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs)

# Y_VX_VY_D
y_vx_vy_d_dynamics = hj.systems.Y_vx_vy_d(min_controls=min_controls,
                                          max_controls=max_controls).with_mode('avoid')
y_vx_vy_d_idxs = [1,3,4,6]
subsys_grid_mins = grid_mins[y_vx_vy_d_idxs]
subsys_grid_maxs = grid_maxs[y_vx_vy_d_idxs]
grid_res = tuple(grid_dims[y_vx_vy_d_idxs])
target_mins = [(0, -2.)]
target_maxs = [(0, +2.), (1, 0.)]

y_vx_vy_d = Subsystem(y_vx_vy_d_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs)


# Step 2 to n
print("Starting reachability calculations...")
for i in range(n):
    vx_vy_w_d.step()
    vx_virt_disturbance = vx_vy_w_d.find_reach_range(dim=0)
    vy_virt_disturbance = vx_vy_w_d.find_reach_range(dim=1)
    w_virt_disturbance = vx_vy_w_d.find_reach_range(dim=2)
    delta_virt_disturbance = vx_vy_w_d.find_reach_range(dim=3)
    yaw_w.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0], delta_virt_disturbance[0]],
                           [vx_virt_disturbance[1], vy_virt_disturbance[1], delta_virt_disturbance[1]])
    yaw_w.step()
    yaw_virt_disturbance = yaw_w.find_reach_range(dim=0)
    x_yaw.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0], w_virt_disturbance[0]],
                           [vx_virt_disturbance[1], vy_virt_disturbance[1], w_virt_disturbance[1]])
    x_yaw.step()
    y_yaw.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0], w_virt_disturbance[0]],
                           [vx_virt_disturbance[1], vy_virt_disturbance[1], w_virt_disturbance[1]])
    y_yaw.step()
    x_vx_vy_d.set_disturbances([yaw_virt_disturbance[0], w_virt_disturbance[0]],
                               [yaw_virt_disturbance[1], w_virt_disturbance[1]])
    x_vx_vy_d.step()
    y_vx_vy_d.set_disturbances([yaw_virt_disturbance[0], w_virt_disturbance[0]],
                               [yaw_virt_disturbance[1], w_virt_disturbance[1]])
    y_vx_vy_d.step()
    print(f"Time step {i+1} completed (t={(i+1)*time_step})")
print("Done")


# Combine and project results
vx_vy_w_d_result = vx_vy_w_d.combine()
yaw_w_result = yaw_w.combine()
x_yaw_result = x_yaw.combine()
y_yaw_result = y_yaw.combine()
x_vx_vy_d_result = x_vx_vy_d.combine()
y_vx_vy_d_result = y_vx_vy_d.combine()

def back_project(grid_dims, subsys_value, subsys_idxs):
    idxs = np.array([np.newaxis] * len(grid_dims))
    idxs[subsys_idxs] = slice(None)
    pattern = np.array(grid_dims)
    pattern[subsys_idxs] = 1
    return np.tile(subsys_value[tuple(idxs)], pattern)

### BACK PROJECT FULL
value_function = back_project(grid_dims, vx_vy_w_d_result, vx_vy_w_d_idxs)
value_function = np.maximum(value_function, back_project(grid_dims, yaw_w_result, yaw_w_idxs))
value_function = np.maximum(value_function, back_project(grid_dims, x_yaw_result, X_yaw_idxs))
value_function = np.maximum(value_function, back_project(grid_dims, y_yaw_result, Y_yaw_idxs))
value_function = np.maximum(value_function, back_project(grid_dims, x_vx_vy_d_result, x_vx_vy_d_idxs))
value_function = np.maximum(value_function, back_project(grid_dims, y_vx_vy_d_result, y_vx_vy_d_idxs))

value_function_d = -shp.project_onto(-value_function, 0, 1, 2, 3, 4, 5)

### BACK PROJECT MANUAL
x_yaw_sel = (x_yaw_result[:,2] + x_yaw_result[:,3]*2)/3.
y_yaw_sel = (y_yaw_result[:,2] + y_yaw_result[:,3]*2)/3.
yaw_w_sel = (yaw_w_result[2,3] + yaw_w_result[3,3]*2)/3.

x_yaw_sel = back_project(grid_dims[[0,1]], x_yaw_sel, [0])
y_yaw_sel = back_project(grid_dims[[0,1]], y_yaw_sel, [1])
x_y_yaw_sel = np.maximum(x_yaw_sel, y_yaw_sel)

vx_vy_w_d_res_proj = -shp.project_onto(-vx_vy_w_d_result[:,:,:,:], 0, 1, 2)
x_vx_vy_d_res_proj = -shp.project_onto(-x_vx_vy_d_result[:,:,:,:], 0, 1, 2)
y_vx_vy_d_res_proj = -shp.project_onto(-y_vx_vy_d_result[:,:,:,:], 0, 1, 2)

x_y_yaw_mask = back_project(grid_dims[[0,1,3]], x_y_yaw_sel, [0, 1])
x_vx_sel = back_project(grid_dims[[0,1,3,6]], x_vx_vy_d_result[:,:,10,:], [0, 2, 3])
y_vx_sel = back_project(grid_dims[[0,1,3,6]], y_vx_vy_d_result[:,:,10,:], [1, 2, 3])
x_y_vx_sel = np.maximum(x_vx_sel, y_vx_sel)
x_y_vx_sel = -shp.project_onto(-x_y_vx_sel, 0, 1, 2)
x_y_vx_sel = np.maximum(x_y_vx_sel, x_y_yaw_mask)

x_y_vx_grid = np.array(list(product(x_vx_vy_d.grid.coordinate_vectors[0], y_vx_vy_d.grid.coordinate_vectors[0], x_vx_vy_d.grid.coordinate_vectors[1])))

hj_tools.plot_set_3D(x_y_vx_grid[:,0], 
            x_y_vx_grid[:,1], 
            x_y_vx_grid[:,2], 
            value_function_d[:,:,3,:,10,3].ravel(),
            # x_y_vx_sel.ravel(),
            ("x", "y", "v_x"))
hj_tools.plot_set_3D(x_y_vx_grid[:,0], 
            x_y_vx_grid[:,1], 
            x_y_vx_grid[:,2], 
            x_y_vx_sel.ravel(),
            ("x", "y", "v_x"))

breakpoint()

vx_vy_w_grid = np.array(list(product(vx_vy_w_d.grid.coordinate_vectors[0], vx_vy_w_d.grid.coordinate_vectors[1], vx_vy_w_d.grid.coordinate_vectors[2])))
hj_tools.plot_set_3D(vx_vy_w_grid[..., 0].ravel(), 
            vx_vy_w_grid[..., 1].ravel(), 
            vx_vy_w_grid[..., 2].ravel(), 
            vx_vy_w_d_res_proj.ravel(),
            ("v_x", "v_y", "w"))

x_vx_vy_grid = np.array(list(product(x_vx_vy_d.grid.coordinate_vectors[0], x_vx_vy_d.grid.coordinate_vectors[1], x_vx_vy_d.grid.coordinate_vectors[2])))
hj_tools.plot_set_3D(x_vx_vy_grid[..., 0].ravel(), 
            x_vx_vy_grid[..., 1].ravel(), 
            x_vx_vy_grid[..., 2].ravel(), 
            x_vx_vy_d_res_proj.ravel(),
            ("x", "v_x", "v_y"))

y_vx_vy_grid = np.array(list(product(y_vx_vy_d.grid.coordinate_vectors[0], y_vx_vy_d.grid.coordinate_vectors[1], y_vx_vy_d.grid.coordinate_vectors[2])))
hj_tools.plot_set_3D(y_vx_vy_grid[..., 0].ravel(), 
            y_vx_vy_grid[..., 1].ravel(), 
            y_vx_vy_grid[..., 2].ravel(), 
            y_vx_vy_d_res_proj.ravel(),
            ("y", "v_x", "v_y"))

plt.figure(figsize=(13, 8))
plt.contourf(yaw_w.grid.coordinate_vectors[0], yaw_w.grid.coordinate_vectors[1], yaw_w_result[:, :].T)
plt.colorbar()
plt.xlabel('yaw')
plt.ylabel('yaw rate')
# plt.show()

plt.figure(figsize=(13, 8))
plt.contourf(x_yaw.grid.coordinate_vectors[0], x_yaw.grid.coordinate_vectors[1], x_yaw_result[:, :].T)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('yaw')
# plt.show()

plt.figure(figsize=(13, 8))
plt.contourf(y_yaw.grid.coordinate_vectors[0], y_yaw.grid.coordinate_vectors[1], y_yaw_result[:, :].T)
plt.colorbar()
plt.xlabel('y')
plt.ylabel('yaw')
plt.show()

