import hj_reachability as hj
import hj_reachability.shapes as shp
import numpy as np
import matplotlib.pyplot as plt
from subsystem import Subsystem
import hj_tools
from time import process_time


### PARAMETERS
time_step = .1
n = 6
min_controls = [-2.0, -16]
max_controls = [+1.5, +16]

# grid_dims = np.array([41, 41, 17, 21, 11, 11, 7])
# grid_mins = np.array([-20., -20., 0.,      -10., -5., -np.pi/2, -np.pi/4])
# grid_maxs = np.array([+20., +20., 2*np.pi, +10., +5., +np.pi/2, +np.pi/4])
# grid_dims = np.array([32, 32, 22, 18, 14, 14, 9])
grid_dims = np.array([22, 22, 12, 12, 9, 9, 7])
grid_mins = np.array([-4., -4., 0.,      +0.1, -2., -np.pi, -np.pi/4])
grid_maxs = np.array([+4., +4., 2*np.pi, +3.6, +2., +np.pi, +np.pi/4])

### SET UP SUBSYSTEMS
# VX_VY_W_D
subsys_dynamics = hj.systems.vx_vy_w_d(min_controls=min_controls,
                                       max_controls=max_controls).with_mode('avoid')
vx_vy_w_d_idxs = [3,4,5,6]
subsys_grid_mins = grid_mins[vx_vy_w_d_idxs]
subsys_grid_maxs = grid_maxs[vx_vy_w_d_idxs]
grid_res = tuple(grid_dims[vx_vy_w_d_idxs])
target_mins = [(0, .0), (2, -4*np.pi)]
target_maxs = [(2, +4*np.pi)]

vx_vy_w_d = Subsystem(subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs)

# X_YAW_W_D
subsys_dynamics = hj.systems.X_yaw_w_d(min_controls=min_controls,
                                       max_controls=max_controls).with_mode('avoid')
x_yaw_w_d_idxs = [0,2,5,6]
subsys_grid_mins = grid_mins[x_yaw_w_d_idxs]
subsys_grid_maxs = grid_maxs[x_yaw_w_d_idxs]
grid_res = tuple(grid_dims[x_yaw_w_d_idxs])
periodic_dims = 1
target_mins = [(0, -0.8), (2, -4*np.pi)]
target_maxs = [(0, +0.8), (2, +4*np.pi)]

x_yaw_w_d = Subsystem(subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)

# Y_YAW_W_D
subsys_dynamics = hj.systems.Y_yaw_w_d(min_controls=min_controls,
                                       max_controls=max_controls).with_mode('avoid')
y_yaw_w_d_idxs = [1,2,5,6]
subsys_grid_mins = grid_mins[y_yaw_w_d_idxs]
subsys_grid_maxs = grid_maxs[y_yaw_w_d_idxs]
grid_res = tuple(grid_dims[y_yaw_w_d_idxs])
periodic_dims = 1
target_mins = [(2, -4*np.pi)]
target_maxs = [(0, -0.8), (2, +4*np.pi)]

y_yaw_w_d = Subsystem(subsys_dynamics, subsys_grid_mins, subsys_grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)
target_mins = [(0, +0.8), (2, -4*np.pi)]
target_maxs = [(2, +4*np.pi)]
y_yaw_w_d.add_target(target_mins, target_maxs)

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
t0 = process_time()
for i in range(n):
    vx_vy_w_d.step()
    vx_virt_disturbance = vx_vy_w_d.find_reach_range(dim=0)
    vy_virt_disturbance = vx_vy_w_d.find_reach_range(dim=1)
    w_virt_disturbance = vx_vy_w_d.find_reach_range(dim=2)
    x_yaw_w_d.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0]],
                               [vx_virt_disturbance[1], vy_virt_disturbance[1]])
    x_yaw_w_d.step()
    y_yaw_w_d.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0]],
                               [vx_virt_disturbance[1], vy_virt_disturbance[1]])
    y_yaw_w_d.step()
    yaw_virt_disturbance = x_yaw_w_d.find_reach_range(dim=1)
    x_vx_vy_d.set_disturbances([yaw_virt_disturbance[0], w_virt_disturbance[0]],
                               [yaw_virt_disturbance[1], w_virt_disturbance[1]])
    x_vx_vy_d.step()
    y_vx_vy_d.set_disturbances([yaw_virt_disturbance[0], w_virt_disturbance[0]],
                               [yaw_virt_disturbance[1], w_virt_disturbance[1]])
    y_vx_vy_d.step()
    print(f"Time step {i+1} completed (t={(i+1)*time_step})")
print(f"Done in {process_time() - t0} seconds")

print("Projecting and combining results...")
t0 = process_time()
value_function = np.tile([np.inf], grid_dims)
for i in range(n+1):
    vf_t = hj_tools.back_project(grid_dims, vx_vy_w_d.result_list[i], vx_vy_w_d_idxs)
    vf_t = np.maximum(vf_t, hj_tools.back_project(grid_dims, x_yaw_w_d.result_list[i], x_yaw_w_d_idxs))
    vf_t = np.maximum(vf_t, hj_tools.back_project(grid_dims, y_yaw_w_d.result_list[i], y_yaw_w_d_idxs))
    vf_t = np.maximum(vf_t, hj_tools.back_project(grid_dims, x_vx_vy_d.result_list[i], x_vx_vy_d_idxs))
    vf_t = np.maximum(vf_t, hj_tools.back_project(grid_dims, y_vx_vy_d.result_list[i], y_vx_vy_d_idxs))
    value_function = np.minimum(value_function, vf_t)
print(f"Done in {process_time() - t0} seconds")

def vf(n):
    value_function = np.tile([np.inf], grid_dims)
    for i in range(n+1):
        vf_t = hj_tools.back_project(grid_dims, vx_vy_w_d.result_list[i], vx_vy_w_d_idxs)
        vf_t = np.maximum(vf_t, hj_tools.back_project(grid_dims, x_yaw_w_d.result_list[i], x_yaw_w_d_idxs))
        vf_t = np.maximum(vf_t, hj_tools.back_project(grid_dims, y_yaw_w_d.result_list[i], y_yaw_w_d_idxs))
        vf_t = np.maximum(vf_t, hj_tools.back_project(grid_dims, x_vx_vy_d.result_list[i], x_vx_vy_d_idxs))
        vf_t = np.maximum(vf_t, hj_tools.back_project(grid_dims, y_vx_vy_d.result_list[i], y_vx_vy_d_idxs))
        value_function = np.minimum(value_function, vf_t)
    return value_function

def mem_footprint():
    subsys_footprint = vx_vy_w_d.result_list[0].nbytes*len(vx_vy_w_d.result_list) + x_vx_vy_d.result_list[0].nbytes*len(x_vx_vy_d.result_list) + \
                       y_vx_vy_d.result_list[0].nbytes*len(y_vx_vy_d.result_list) + x_yaw_w_d.result_list[0].nbytes*len(x_yaw_w_d.result_list) + \
                       y_yaw_w_d.result_list[0].nbytes*len(y_yaw_w_d.result_list)
    recon_footprint = value_function.nbytes
    # recon_footprint = vx_vy_w_d.result_list[0][0,0,0,0].nbytes*np.prod(grid_dims)
    print(f"subsystems: {subsys_footprint/1024} kB\n reconstruction: {recon_footprint/1024} kB\n reduction: {recon_footprint/subsys_footprint}")

### LEAST-RESTRICTIVE CONTROL SET
def plot_combined_control_set(x):
    res = 50
    inputs = [np.linspace(min_controls[0], max_controls[0], res), np.linspace(min_controls[1], max_controls[1], res)]
    vx_vy_w_d_lrcs = hj_tools.LRCS(vx_vy_w_d.dynamics, time_step, *vx_vy_w_d.grid.coordinate_vectors)
    x_vx_vy_d_lrcs = hj_tools.LRCS(x_vx_vy_d.dynamics, time_step, *x_vx_vy_d.grid.coordinate_vectors)
    y_vx_vy_d_lrcs = hj_tools.LRCS(y_vx_vy_d.dynamics, time_step, *y_vx_vy_d.grid.coordinate_vectors)
    x_yaw_w_d_lrcs = hj_tools.LRCS(x_yaw_w_d.dynamics, time_step, *x_yaw_w_d.grid.coordinate_vectors, periodic_dims=[1])
    y_yaw_w_d_lrcs = hj_tools.LRCS(y_yaw_w_d.dynamics, time_step, *y_yaw_w_d.grid.coordinate_vectors, periodic_dims=[1])
    vx_vy_w_d_lrcs.dynamics.open_loop_dynamics(x[vx_vy_w_d_idxs], None)
    x_vx_vy_d_lrcs.dynamics.open_loop_dynamics(x[x_vx_vy_d_idxs], None)
    y_vx_vy_d_lrcs.dynamics.open_loop_dynamics(x[y_vx_vy_d_idxs], None)
    x_yaw_w_d_lrcs.dynamics.open_loop_dynamics(x[x_yaw_w_d_idxs], None)
    y_yaw_w_d_lrcs.dynamics.open_loop_dynamics(x[y_yaw_w_d_idxs], None)
    vx_vy_w_d_lrcs.dynamics.control_jacobian(x[vx_vy_w_d_idxs], None)
    x_vx_vy_d_lrcs.dynamics.control_jacobian(x[x_vx_vy_d_idxs], None)
    y_vx_vy_d_lrcs.dynamics.control_jacobian(x[y_vx_vy_d_idxs], None)
    x_yaw_w_d_lrcs.dynamics.control_jacobian(x[x_yaw_w_d_idxs], None)
    y_yaw_w_d_lrcs.dynamics.control_jacobian(x[y_yaw_w_d_idxs], None)
    x_vx_vy_d_lrcs.dynamics.disturbance_dynamics(x[x_vx_vy_d_idxs], None, np.array([0,0]))
    y_vx_vy_d_lrcs.dynamics.disturbance_dynamics(x[y_vx_vy_d_idxs], None, np.array([0,0]))
    x_yaw_w_d_lrcs.dynamics.disturbance_dynamics(x[x_yaw_w_d_idxs], None, np.array([0,0]))
    y_yaw_w_d_lrcs.dynamics.disturbance_dynamics(x[y_yaw_w_d_idxs], None, np.array([0,0]))

    control_set = np.tile([np.inf], (res, res))
    cs_t = np.tile([-np.inf], (res, res))
    t0 = process_time()
    for i in range(n+1):
        t1 = process_time()
        cs_t = vx_vy_w_d_lrcs.lrcs(inputs, vx_vy_w_d.result_list[i], x[vx_vy_w_d_idxs])
        print(f"subsys 1 LRCS in {process_time() - t1} seconds")

        t1 = process_time()
        subsys_controls = x_vx_vy_d_lrcs.lrcs(inputs, x_vx_vy_d.result_list[i], x[x_vx_vy_d_idxs], disturbance=x[[2,5]])
        print(f"subsys 2 LRCS in {process_time() - t1} seconds")
        cs_t = np.maximum(cs_t, subsys_controls)

        t1 = process_time()
        subsys_controls = y_vx_vy_d_lrcs.lrcs(inputs, y_vx_vy_d.result_list[i], x[y_vx_vy_d_idxs], disturbance=x[[2,5]])
        print(f"subsys 3 LRCS in {process_time() - t1} seconds")
        cs_t = np.maximum(cs_t, subsys_controls)

        t1 = process_time()
        subsys_controls = x_yaw_w_d_lrcs.lrcs(inputs, x_yaw_w_d.result_list[i], x[x_yaw_w_d_idxs], disturbance=x[[3,4]])
        print(f"subsys 4 LRCS in {process_time() - t1} seconds")
        cs_t = np.maximum(cs_t, subsys_controls)

        t1 = process_time()
        subsys_controls = y_yaw_w_d_lrcs.lrcs(inputs, y_yaw_w_d.result_list[i], x[y_yaw_w_d_idxs], disturbance=x[[3,4]])
        print(f"subsys 5 LRCS in {process_time() - t1} seconds")
        cs_t = np.maximum(cs_t, subsys_controls)
        control_set = np.minimum(control_set, cs_t)
    # control_set = (control_set >= 0)
    print(f"Combined LRCS in {process_time() - t0} seconds")

    xs, ys = np.meshgrid(*inputs)
    plt.figure(figsize=(13, 8))
    plt.contourf(xs[0,:], ys[:,0], control_set)
    # plt.colorbar()
    plt.xlabel('a_x')
    plt.ylabel('delta_dot')
    return control_set

def plot_reconstructed_control_set(x):
    dynamics = hj.systems.Bicycle7D(min_controls=min_controls,
                                    max_controls=max_controls).with_mode('avoid')
    full_lrcs = hj_tools.LRCS(dynamics, time_step, 
                              x_yaw_w_d.grid.coordinate_vectors[0],
                              y_yaw_w_d.grid.coordinate_vectors[0],
                              y_yaw_w_d.grid.coordinate_vectors[1],
                              vx_vy_w_d.grid.coordinate_vectors[0],
                              vx_vy_w_d.grid.coordinate_vectors[1],
                              vx_vy_w_d.grid.coordinate_vectors[2],
                              vx_vy_w_d.grid.coordinate_vectors[3], 
                              periodic_dims=[2])
    dynamics.open_loop_dynamics(x, None)
    dynamics.control_jacobian(x, None)
    res = 50
    inputs = [np.linspace(min_controls[0], max_controls[0], res), np.linspace(min_controls[1], max_controls[1], res)]
    t0 = process_time()
    control_set = full_lrcs.lrcs(inputs, value_function, x)
    print(f"reconstructed LRCS in {process_time() - t0} seconds")
    # control_set = (control_set >= 0)

    xs, ys = np.meshgrid(*inputs)
    plt.figure(figsize=(13, 8))
    plt.contourf(xs[0,:], ys[:,0], control_set)
    # plt.colorbar()
    plt.xlabel('a_x')
    plt.ylabel('delta_dot')
    return control_set

mem_footprint()
breakpoint()
x = np.array([-1.2, -2.8, 6., 1.8, 1.4, 0., 2.])
# x = np.array([1.5, 2.1, 2.8, 1.2, 1., 0., 2.])
plot_combined_control_set(x)
plt.colorbar()
plot_reconstructed_control_set(x)
plt.colorbar()
plt.show()
breakpoint()

### ANIMATION
# hj_tools.animation_images(x_vx_vy_d.grid.coordinate_vectors[0], 
#             y_vx_vy_d.grid.coordinate_vectors[0], 
#             vx_vy_w_d.grid.coordinate_vectors[0], 
#             value_function[:,:,11,:,4,4,3],
#             vf(0)[:,:,11,:,4,4,3],
#             ("x", "y", "v_x"), 'anim_0yaw')
# hj_tools.animation_images(x_vx_vy_d.grid.coordinate_vectors[0], 
#             y_vx_vy_d.grid.coordinate_vectors[0], 
#             vx_vy_w_d.grid.coordinate_vectors[0], 
#             value_function[:,:,8,:,4,4,3],
#             vf(0)[:,:,8,:,4,4,3],
#             ("x", "y", "v_x"), 'anim_90yaw')
# breakpoint()

### PLOTTING
# [22, 22, 12, 12, 9, 9, 7]
[hj_tools.plot_set_3D(x_vx_vy_d.grid.coordinate_vectors[0], 
            y_vx_vy_d.grid.coordinate_vectors[0], 
            vx_vy_w_d.grid.coordinate_vectors[0], 
            value_function[:,:,i,:,4,4,3],
            vf(0)[:,:,i,:,4,4,3],
            ("x", "y", "v_x")).show() for i in [8,11]]

hj_tools.plot_set_3D(x_vx_vy_d.grid.coordinate_vectors[0], 
            y_vx_vy_d.grid.coordinate_vectors[0], 
            vx_vy_w_d.grid.coordinate_vectors[0], 
            value_function[:,:,11,:,4,4,3],
            vf(0)[:,:,11,:,4,4,3],
            ("x", "y", "v_x")).show()

hj_tools.plot_set_3D(x_yaw_w_d.grid.coordinate_vectors[1]*180/np.pi, 
            vx_vy_w_d.grid.coordinate_vectors[0], 
            vx_vy_w_d.grid.coordinate_vectors[1], 
            value_function[14,16,:,:,:,1,3],
            vf(0)[14,16,:,:,:,1,3],
#             value_function_d[10,9,:,:,:,1],
            ("yaw", "v_x", "v_y")).show()

hj_tools.plot_set_3D(x_yaw_w_d.grid.coordinate_vectors[1]*180/np.pi, 
            vx_vy_w_d.grid.coordinate_vectors[0], 
            vx_vy_w_d.grid.coordinate_vectors[2], 
            value_function[14,16,:,:,4,:,3],
            vf(0)[14,16,:,:,4,:,3],
            ("yaw", "v_x", "w")).show()



breakpoint()

# SUBSYSTEM RESULTS
vx_vy_w_d_result = vx_vy_w_d.combine()
x_yaw_w_d_result = x_yaw_w_d.combine()
y_yaw_w_d_result = y_yaw_w_d.combine()
x_vx_vy_d_result = x_vx_vy_d.combine()
y_vx_vy_d_result = y_vx_vy_d.combine()

vx_vy_w_d_res_proj = -shp.project_onto(-vx_vy_w_d_result, 0, 1, 2)
x_vx_vy_d_res_proj = -shp.project_onto(-x_vx_vy_d_result, 0, 1, 2)
y_vx_vy_d_res_proj = -shp.project_onto(-y_vx_vy_d_result, 0, 1, 2)
x_yaw_w_d_res_proj = -shp.project_onto(-x_yaw_w_d_result, 0, 1, 2)
y_yaw_w_d_res_proj = -shp.project_onto(-y_yaw_w_d_result, 0, 1, 2)

hj_tools.plot_set_3D(vx_vy_w_d.grid.coordinate_vectors[0], 
            vx_vy_w_d.grid.coordinate_vectors[1], 
            vx_vy_w_d.grid.coordinate_vectors[2], 
            vx_vy_w_d_res_proj,
            vx_vy_w_d.result_list[0][...,0],
            ("v_x", "v_y", "w")).show()

hj_tools.plot_set_3D(x_vx_vy_d.grid.coordinate_vectors[0], 
            x_vx_vy_d.grid.coordinate_vectors[1], 
            x_vx_vy_d.grid.coordinate_vectors[2], 
            x_vx_vy_d_res_proj,
            x_vx_vy_d.result_list[0][...,0],
            ("x", "v_x", "v_y")).show()

hj_tools.plot_set_3D(y_vx_vy_d.grid.coordinate_vectors[0], 
            y_vx_vy_d.grid.coordinate_vectors[1], 
            y_vx_vy_d.grid.coordinate_vectors[2], 
            y_vx_vy_d_res_proj,
            y_vx_vy_d.result_list[0][...,0],
            ("y", "v_x", "v_y")).show()

hj_tools.plot_set_3D(x_yaw_w_d.grid.coordinate_vectors[0], 
            x_yaw_w_d.grid.coordinate_vectors[1], 
            x_yaw_w_d.grid.coordinate_vectors[2], 
            x_yaw_w_d_res_proj,
            x_yaw_w_d.result_list[0][...,0],
            ("x", "yaw", "w")).show()

hj_tools.plot_set_3D(y_yaw_w_d.grid.coordinate_vectors[0], 
            y_yaw_w_d.grid.coordinate_vectors[1], 
            y_yaw_w_d.grid.coordinate_vectors[2], 
            y_yaw_w_d_res_proj,
            y_yaw_w_d.result_list[0][...,0],
            ("y", "yaw", "w")).show()

breakpoint()