import hj_reachability as hj
import hj_reachability.shapes as shp
import numpy as np
import matplotlib.pyplot as plt
from subsystem import Subsystem
import hj_tools
from itertools import product


### PARAMETERS
time_step = .2
n = 7
min_controls = [-2.0, -16]
max_controls = [+1.5, +16]

# grid_dims = np.array([41, 41, 17, 21, 11, 11, 7])
# grid_mins = np.array([-20., -20., 0.,      -10., -5., -np.pi/2, -np.pi/4])
# grid_maxs = np.array([+20., +20., 2*np.pi, +10., +5., +np.pi/2, +np.pi/4])
grid_dims = np.array([22, 22, 12, 12, 9, 9, 7])
grid_mins = np.array([-4., -4., 0.,      +0.1, -2., -np.pi/2, -np.pi/4])
grid_maxs = np.array([+4., +4., 2*np.pi, +3.6, +2., +np.pi/2, +np.pi/4])

### SET UP SUBSYSTEMS
# VX_VY_W_D
subsys_dynamics = hj.systems.vx_vy_w_d(min_controls=min_controls,
                                       max_controls=max_controls).with_mode('avoid')
vx_vy_w_d_idxs = [3,4,5,6]
subsys_grid_mins = grid_mins[vx_vy_w_d_idxs]
subsys_grid_maxs = grid_maxs[vx_vy_w_d_idxs]
grid_res = tuple(grid_dims[vx_vy_w_d_idxs])
target_mins = [(0, .0), (2, -np.pi/1.5)]
target_maxs = [(2, +np.pi/1.5)]

vx_vy_w_d = Subsystem(subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs)

# YAW_W_D
subsys_dynamics = hj.systems.yaw_w_d(min_controls=min_controls,
                                     max_controls=max_controls).with_mode('avoid')
yaw_w_d_idxs = [2,5,6]
subsys_grid_mins = grid_mins[yaw_w_d_idxs]
subsys_grid_maxs = grid_maxs[yaw_w_d_idxs]
grid_res = tuple(grid_dims[yaw_w_d_idxs])
periodic_dims = 0
target_mins = [(1, -np.pi/1.5)]#[(0, np.pi/4.)]
target_maxs = [(1, +np.pi/1.5)]#[(0, 7*np.pi/4.)]

yaw_w_d = Subsystem(subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)

# X_YAW
subsys_dynamics = hj.systems.X_yaw().with_mode('avoid')
x_yaw_idxs = [0,2]
subsys_grid_mins = grid_mins[x_yaw_idxs]
subsys_grid_maxs = grid_maxs[x_yaw_idxs]
grid_res = tuple(grid_dims[x_yaw_idxs])
periodic_dims = 1
target_mins = [(0, -0.8)]
target_maxs = [(0, +0.8)]

x_yaw = Subsystem(subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)

# Y_YAW
subsys_dynamics = hj.systems.Y_yaw().with_mode('avoid')
y_yaw_idxs = [1,2]
subsys_grid_mins = grid_mins[y_yaw_idxs]
subsys_grid_maxs = grid_maxs[y_yaw_idxs]
grid_res = tuple(grid_dims[y_yaw_idxs])
periodic_dims = 1
target_mins = []
target_maxs = [(0, -0.8)]

y_yaw = Subsystem(subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)
target_mins = [(0, +0.8)]
target_maxs = []
y_yaw.add_target(target_mins, target_maxs)

# X_VX_VY_D
subsys_dynamics = hj.systems.X_vx_vy_d(min_controls=min_controls,
                                       max_controls=max_controls).with_mode('avoid')
x_vx_vy_d_idxs = [0,3,4,6]
subsys_grid_mins = grid_mins[x_vx_vy_d_idxs]
subsys_grid_maxs = grid_maxs[x_vx_vy_d_idxs]
grid_res = tuple(grid_dims[x_vx_vy_d_idxs])
target_mins = [(0, -0.8), (1, .0)]
target_maxs = [(0, +0.8)]

x_vx_vy_d = Subsystem(subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs)

# Y_VX_VY_D
subsys_dynamics = hj.systems.Y_vx_vy_d(min_controls=min_controls,
                                       max_controls=max_controls).with_mode('avoid')
y_vx_vy_d_idxs = [1,3,4,6]
subsys_grid_mins = grid_mins[y_vx_vy_d_idxs]
subsys_grid_maxs = grid_maxs[y_vx_vy_d_idxs]
grid_res = tuple(grid_dims[y_vx_vy_d_idxs])
target_mins = [(1, .0)]
target_maxs = [(0, -0.8)]

y_vx_vy_d = Subsystem(subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs)
target_mins = [(0, +0.8), (1, .0)]
target_maxs = []
y_vx_vy_d.add_target(target_mins, target_maxs)
del subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, target_mins, target_maxs

# CALCULATE SUBSYTEM AVOID SETS
print("Starting reachability calculations...")
for i in range(n):
    vx_vy_w_d.step()
    vx_virt_disturbance = vx_vy_w_d.find_reach_range(dim=0)
    vy_virt_disturbance = vx_vy_w_d.find_reach_range(dim=1)
    w_virt_disturbance = vx_vy_w_d.find_reach_range(dim=2)
    yaw_w_d.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0]],
                             [vx_virt_disturbance[1], vy_virt_disturbance[1]])
    yaw_w_d.step()
    x_yaw.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0], w_virt_disturbance[0]],
                           [vx_virt_disturbance[1], vy_virt_disturbance[1], w_virt_disturbance[1]])
    x_yaw.step()
    y_yaw.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0], w_virt_disturbance[0]],
                           [vx_virt_disturbance[1], vy_virt_disturbance[1], w_virt_disturbance[1]])
    y_yaw.step()
    yaw_virt_disturbance = yaw_w_d.find_reach_range(dim=0)
    if yaw_virt_disturbance is None:
        yaw_virt_disturbance = x_yaw.find_reach_range(dim=1)
    if yaw_virt_disturbance is None:
        yaw_virt_disturbance = y_yaw.find_reach_range(dim=1)
    # print(x_yaw.find_reach_range(dim=1), y_yaw.find_reach_range(dim=1))
    x_vx_vy_d.set_disturbances([yaw_virt_disturbance[0], w_virt_disturbance[0]],
                               [yaw_virt_disturbance[1], w_virt_disturbance[1]])
    x_vx_vy_d.step()
    y_vx_vy_d.set_disturbances([yaw_virt_disturbance[0], w_virt_disturbance[0]],
                               [yaw_virt_disturbance[1], w_virt_disturbance[1]])
    y_vx_vy_d.step()
    print(f"Time step {i+1} completed (t={(i+1)*time_step})")
print("Done")


### BACK PROJECTION
def back_project(grid_dims, subsys_value, subsys_idxs):
    idxs = np.array([np.newaxis] * len(grid_dims))
    idxs[subsys_idxs] = slice(None)
    pattern = np.array(grid_dims)
    pattern[subsys_idxs] = 1
    return np.tile(subsys_value[tuple(idxs)], pattern)

print("Projecting and combining results...")
value_function = np.tile([np.inf], grid_dims)
for i in range(n+1):
    vf_t = back_project(grid_dims, vx_vy_w_d.result_list[i], vx_vy_w_d_idxs)
    vf_t = np.maximum(vf_t, back_project(grid_dims, yaw_w_d.result_list[i], yaw_w_d_idxs))
    vf_t = np.maximum(vf_t, back_project(grid_dims, x_yaw.result_list[i], x_yaw_idxs))
    vf_t = np.maximum(vf_t, back_project(grid_dims, y_yaw.result_list[i], y_yaw_idxs))
    vf_t = np.maximum(vf_t, back_project(grid_dims, x_vx_vy_d.result_list[i], x_vx_vy_d_idxs))
    vf_t = np.maximum(vf_t, back_project(grid_dims, y_vx_vy_d.result_list[i], y_vx_vy_d_idxs))
    value_function = np.minimum(value_function, vf_t)

value_function_d = -shp.project_onto(-value_function, 0, 1, 2, 3, 4, 5)

dynamics = hj.systems.Bicycle7D(min_controls=min_controls,
                                max_controls=max_controls).with_mode('avoid')
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
                                hj.sets.Box(grid_mins, grid_maxs),
                                grid_dims,
                                periodic_dims=[2])

print("Done")


### LEAST-RESTRICTIVE CONTROL SET
def plot_combined_control_set(x):
    res = 50
    xs, ys = np.meshgrid(np.linspace(min_controls[0], max_controls[0], res), np.linspace(min_controls[1], max_controls[1], res))
    control_set = np.tile([np.inf], (res, res))

    for i in range(n+1):
        a, b = hj_tools.lrcs(vx_vy_w_d.dynamics, vx_vy_w_d.grid, time_step, vx_vy_w_d.result_list[i], x[vx_vy_w_d_idxs])
        subsys_controls = a + b[0]*xs + b[1]*ys
        cs_t = subsys_controls
        if not np.isfinite(vx_vy_w_d.result_list[i]).all():
            cs_t = -np.ones(cs_t.shape)

        a, b = hj_tools.lrcs(x_vx_vy_d.dynamics, x_vx_vy_d.grid, time_step, x_vx_vy_d.result_list[i], x[x_vx_vy_d_idxs], disturbance=x[[2,5]])
        subsys_controls = a + b[0]*xs + b[1]*ys
        if np.isfinite(x_vx_vy_d.result_list[i]).all():
            cs_t = np.maximum(cs_t, subsys_controls)

        a, b = hj_tools.lrcs(y_vx_vy_d.dynamics, y_vx_vy_d.grid, time_step, y_vx_vy_d.result_list[i], x[y_vx_vy_d_idxs], disturbance=x[[2,5]])
        subsys_controls = a + b[0]*xs + b[1]*ys
        if np.isfinite(y_vx_vy_d.result_list[i]).all():
            cs_t = np.maximum(cs_t, subsys_controls)

        a, b = hj_tools.lrcs(yaw_w_d.dynamics, yaw_w_d.grid, time_step, yaw_w_d.result_list[i], x[yaw_w_d_idxs], disturbance=x[[3,4]])
        subsys_controls = a + b[0]*xs + b[1]*ys
        if np.isfinite(yaw_w_d.result_list[i]).all():
            cs_t = np.maximum(cs_t, subsys_controls)

        a, b = hj_tools.lrcs(x_yaw.dynamics, x_yaw.grid, time_step, x_yaw.result_list[i], x[x_yaw_idxs], disturbance=x[[3,4,5]])
        subsys_controls = a + b[0]*xs + b[1]*ys
        if np.isfinite(x_yaw.result_list[i]).all():
            cs_t = np.maximum(cs_t, subsys_controls)

        a, b = hj_tools.lrcs(y_yaw.dynamics, y_yaw.grid, time_step, y_yaw.result_list[i], x[y_yaw_idxs], disturbance=x[[3,4,5]])
        subsys_controls = a + b[0]*xs + b[1]*ys
        if np.isfinite(y_yaw.result_list[i]).all():
            cs_t = np.maximum(cs_t, subsys_controls)
        control_set = np.minimum(control_set, cs_t)
    # control_set = (control_set >= 0)
    plt.figure(figsize=(13, 8))
    plt.contourf(xs[0,:], ys[:,0], control_set)
    plt.colorbar()
    plt.xlabel('a_x')
    plt.ylabel('delta_dot')
    return control_set

def plot_reconstructed_control_set(x):
    a, b = hj_tools.lrcs(dynamics, grid, time_step, value_function, x)
    res = 50
    xs, ys = np.meshgrid(np.linspace(min_controls[0], max_controls[0], res), np.linspace(min_controls[1], max_controls[1], res))
    control_set = a + b[0]*xs + b[1]*ys
    # control_set = (control_set >= 0)
    plt.figure(figsize=(13, 8))
    plt.contourf(xs[0,:], ys[:,0], control_set)
    plt.colorbar()
    plt.xlabel('a_x')
    plt.ylabel('delta_dot')
    return control_set

x = np.array([-0.8, -0.8, 0., +2.8, 0., +1., +np.pi/4.])
# plot_combined_control_set(x)
# plot_reconstructed_control_set(x)
# plt.show()
def vf(n):
    value_function = np.tile([np.inf], grid_dims)
    for i in range(n+1):
        vf_t = back_project(grid_dims, vx_vy_w_d.result_list[i], vx_vy_w_d_idxs)
        vf_t = np.maximum(vf_t, back_project(grid_dims, yaw_w_d.result_list[i], yaw_w_d_idxs))
        vf_t = np.maximum(vf_t, back_project(grid_dims, x_yaw.result_list[i], x_yaw_idxs))
        vf_t = np.maximum(vf_t, back_project(grid_dims, y_yaw.result_list[i], y_yaw_idxs))
        vf_t = np.maximum(vf_t, back_project(grid_dims, x_vx_vy_d.result_list[i], x_vx_vy_d_idxs))
        vf_t = np.maximum(vf_t, back_project(grid_dims, y_vx_vy_d.result_list[i], y_vx_vy_d_idxs))
        value_function = np.minimum(value_function, vf_t)
    return value_function

breakpoint()

def mem_footprint():
    subsys_footprint = vx_vy_w_d.result_list[0].nbytes*len(vx_vy_w_d.result_list) + x_vx_vy_d.result_list[0].nbytes*len(x_vx_vy_d.result_list) + \
                        y_vx_vy_d.result_list[0].nbytes*len(y_vx_vy_d.result_list) + yaw_w_d.result_list[0].nbytes*len(yaw_w_d.result_list) + \
                        x_yaw.result_list[0].nbytes*len(x_yaw.result_list) + y_yaw.result_list[0].nbytes*len(y_yaw.result_list)
    recon_footprint = value_function.nbytes
    print(f"subsystems: {subsys_footprint/1024} kB\n reconstruction: {recon_footprint/1024} kB\n reduction: {recon_footprint/subsys_footprint}")

### PLOTTING
x_y_vx_grid = np.array(list(product(x_vx_vy_d.grid.coordinate_vectors[0], y_vx_vy_d.grid.coordinate_vectors[0], x_vx_vy_d.grid.coordinate_vectors[1])))
x_y_yaw_grid = np.array(list(product(x_vx_vy_d.grid.coordinate_vectors[0], y_vx_vy_d.grid.coordinate_vectors[0], yaw_w_d.grid.coordinate_vectors[0])))
x_y_vy_grid = np.array(list(product(x_vx_vy_d.grid.coordinate_vectors[0], y_vx_vy_d.grid.coordinate_vectors[0], y_vx_vy_d.grid.coordinate_vectors[2])))
# [22, 22, 12, 12, 9, 9, 7]
[hj_tools.plot_set_3D(x_y_vx_grid[:,0], 
            x_y_vx_grid[:,1], 
            x_y_vx_grid[:,2], 
            value_function_d[:,:,i,:,4,4].ravel(),
            ("x", "y", "v_x")) for i in range(12)]

breakpoint()

# SUBSYSTEM RESULTS
vx_vy_w_d_result = vx_vy_w_d.combine()
yaw_w_d_result = yaw_w_d.combine()
x_yaw_result = x_yaw.combine()
y_yaw_result = y_yaw.combine()
x_vx_vy_d_result = x_vx_vy_d.combine()
y_vx_vy_d_result = y_vx_vy_d.combine()

vx_vy_w_d_res_proj = -shp.project_onto(-vx_vy_w_d_result, 0, 1, 2)
x_vx_vy_d_res_proj = -shp.project_onto(-x_vx_vy_d_result, 0, 1, 2)
y_vx_vy_d_res_proj = -shp.project_onto(-y_vx_vy_d_result, 0, 1, 2)
yaw_w_d_res_proj = -shp.project_onto(-yaw_w_d_result, 0, 1)

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

# g = np.array(list(product(yaw_w_d.grid.coordinate_vectors[0], yaw_w_d.grid.coordinate_vectors[1], yaw_w_d.grid.coordinate_vectors[2])))

# hj_tools.plot_set_3D(g[..., 0].ravel(), 
#             g[..., 1].ravel(), 
#             g[..., 2].ravel(), 
#             t.ravel(),
#             ("yaw", "w", "d"))


y_vx_vy_grid = np.array(list(product(y_vx_vy_d.grid.coordinate_vectors[0], y_vx_vy_d.grid.coordinate_vectors[1], y_vx_vy_d.grid.coordinate_vectors[2])))
hj_tools.plot_set_3D(y_vx_vy_grid[..., 0].ravel(), 
            y_vx_vy_grid[..., 1].ravel(), 
            y_vx_vy_grid[..., 2].ravel(), 
            y_vx_vy_d_res_proj.ravel(),
            ("y", "v_x", "v_y"))

plt.figure(figsize=(13, 8))
plt.contourf(yaw_w_d.grid.coordinate_vectors[0], yaw_w_d.grid.coordinate_vectors[1], yaw_w_d_res_proj[:, :].T)
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

# plt.figure(figsize=(8, 8))
# plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], -value_function[:, :, 3, 0, 0, 0, 0].T, [-1000, 0, 1000])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


breakpoint()