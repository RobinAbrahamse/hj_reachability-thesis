import hj_reachability as hj
import hj_reachability.shapes as shp
import numpy as np
import matplotlib.pyplot as plt
from subsystem import Subsystem
import hj_tools
from itertools import product


time_step = .25
n = 8
min_controls = [-.2, -16]
max_controls = [+.2, +16]

# VX_VY_W_D
vx_vy_w_d_dynamics = hj.systems.vx_vy_w_d(min_controls=min_controls,
                                          max_controls=max_controls).with_mode('avoid')

grid_mins = np.array([-10., -7., -np.pi/2, -np.pi/4])
grid_maxs = np.array([+10., +7., +np.pi/2, +np.pi/4])
grid_res = (21, 21, 21, 11)

target_mins = [(1, -1)]
target_maxs = [(0, 0.), (1, 1.)]

vx_vy_w_d = Subsystem(vx_vy_w_d_dynamics, grid_mins, grid_maxs, grid_res, time_step, target_mins, target_maxs)
vx_vy_w_d.step()

# Calculate virtual disturbance ranges
vx_virt_disturbance = vx_vy_w_d.find_reach_range(dim=0)
vy_virt_disturbance = vx_vy_w_d.find_reach_range(dim=1)
w_virt_disturbance = vx_vy_w_d.find_reach_range(dim=2)

# YAW_W
yaw_w_dynamics = hj.systems.yaw_w(min_disturbances=[vy_virt_disturbance[0]],
                                  max_disturbances=[vy_virt_disturbance[1]]).with_mode('avoid')

grid_mins = np.array([0.,      -np.pi/2])
grid_maxs = np.array([2*np.pi, +np.pi/2])
grid_res = (21, 21)
periodic_dims = 0

target_mins = [(0, np.pi/4.)]
target_maxs = [(0, 7*np.pi/4.)]

yaw_w = Subsystem(yaw_w_dynamics, grid_mins, grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)
yaw_w.step()

# X_YAW
X_yaw_dynamics = hj.systems.X_yaw(min_disturbances=[vx_virt_disturbance[0], vy_virt_disturbance[0], w_virt_disturbance[0]],
                                  max_disturbances=[vx_virt_disturbance[1], vy_virt_disturbance[1], w_virt_disturbance[1]]).with_mode('avoid')
grid_mins = np.array([-20., 0.])
grid_maxs = np.array([+20., 2*np.pi])
grid_res = (41, 21)
periodic_dims = 1

target_mins = [(0, -6.), (1, np.pi/4.)]
target_maxs = [(0, +6.), (1, 7*np.pi/4.)]

x_yaw = Subsystem(X_yaw_dynamics, grid_mins, grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)
x_yaw.step()


# Y_YAW
Y_yaw_dynamics = hj.systems.Y_yaw(min_disturbances=[vx_virt_disturbance[0], vy_virt_disturbance[0], w_virt_disturbance[0]],
                                  max_disturbances=[vx_virt_disturbance[1], vy_virt_disturbance[1], w_virt_disturbance[1]]).with_mode('avoid')
grid_mins = np.array([-20., 0.])
grid_maxs = np.array([+20., 2*np.pi])
grid_res = (41, 21)
periodic_dims = 1

target_mins = [(0, -2.), (1, np.pi/4.)]
target_maxs = [(0, +2.), (1, 7*np.pi/4.)]

y_yaw = Subsystem(X_yaw_dynamics, grid_mins, grid_maxs, grid_res, time_step, target_mins, target_maxs, periodic_dims=periodic_dims)
y_yaw.step()

# Calculate virtual disturbance ranges
yaw_virt_disturbance = yaw_w.find_reach_range(dim=0)

# X_VX_VY_D
x_vx_vy_d_dynamics = hj.systems.X_vx_vy_d(min_controls=min_controls,
                                            max_controls=max_controls,
                                            min_disturbances=[yaw_virt_disturbance[0], w_virt_disturbance[0]],
                                            max_disturbances=[yaw_virt_disturbance[1], w_virt_disturbance[1]]).with_mode('avoid')

grid_mins = np.array([-20., -10., -7., -np.pi/4])
grid_maxs = np.array([+20., +10., +7., +np.pi/4])
grid_res = (41, 21, 21, 11)
target_mins = [(0, -6.)]
target_maxs = [(0, +6.), (1, 0.)]

x_vx_vy_d = Subsystem(x_vx_vy_d_dynamics, grid_mins, grid_maxs, grid_res, time_step, target_mins, target_maxs)
x_vx_vy_d.step()

# Y_VX_VY_D
y_vx_vy_d_dynamics = hj.systems.Y_vx_vy_d(min_controls=min_controls,
                                            max_controls=max_controls,
                                            min_disturbances=[yaw_virt_disturbance[0], w_virt_disturbance[0]],
                                            max_disturbances=[yaw_virt_disturbance[1], w_virt_disturbance[1]]).with_mode('avoid')

grid_mins = np.array([-20., -10., -7., -np.pi/4])
grid_maxs = np.array([+20., +10., +7., +np.pi/4])
grid_res = (41, 21, 21, 11)
target_mins = [(0, -2.)]
target_maxs = [(0, +2.), (1, 0.)]

y_vx_vy_d = Subsystem(y_vx_vy_d_dynamics, grid_mins, grid_maxs, grid_res, time_step, target_mins, target_maxs)
y_vx_vy_d.step()


# Step 2 to n
for i in range(n-1):
    print("time step ", i+1, " completed (t=", (i+1)*time_step, ")")
    vx_vy_w_d.step()
    vx_virt_disturbance = vx_vy_w_d.find_reach_range(dim=0)
    vy_virt_disturbance = vx_vy_w_d.find_reach_range(dim=1)
    w_virt_disturbance = vx_vy_w_d.find_reach_range(dim=2)
    yaw_w.set_disturbances([vy_virt_disturbance[0]], [vy_virt_disturbance[1]])
    yaw_w.step()
    x_yaw.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0], w_virt_disturbance[0]],
                        [vx_virt_disturbance[0], vy_virt_disturbance[1], w_virt_disturbance[1]])
    x_yaw.step()
    y_yaw.set_disturbances([vx_virt_disturbance[0], vy_virt_disturbance[0], w_virt_disturbance[0]],
                        [vx_virt_disturbance[0], vy_virt_disturbance[1], w_virt_disturbance[1]])
    y_yaw.step()
    yaw_virt_disturbance = yaw_w.find_reach_range(dim=0)
    x_vx_vy_d.set_disturbances([yaw_virt_disturbance[0], w_virt_disturbance[0]],
                            [yaw_virt_disturbance[1], w_virt_disturbance[1]])
    x_vx_vy_d.step()
    y_vx_vy_d.set_disturbances([yaw_virt_disturbance[0], w_virt_disturbance[0]],
                            [yaw_virt_disturbance[1], w_virt_disturbance[1]])
    y_vx_vy_d.step()
print("Completed all ", n, "time steps (t=", n*time_step, ")")


# Combine and project results
vx_vy_w_d_result = vx_vy_w_d.combine()
yaw_w_result = yaw_w.combine()
x_yaw_result = x_yaw.combine()
y_yaw_result = y_yaw.combine()
x_vx_vy_d_result = x_vx_vy_d.combine()
y_vx_vy_d_result = y_vx_vy_d.combine()

x_yaw_sel = (x_yaw_result[:,2] + x_yaw_result[:,3]*2)/3.
y_yaw_sel = (y_yaw_result[:,2] + y_yaw_result[:,3]*2)/3.
yaw_w_sel = (yaw_w_result[2,3] + yaw_w_result[3,3]*2)/3.

x_yaw_t = np.repeat(x_yaw_sel[:,np.newaxis], y_yaw.grid.shape[0], axis=1)
y_yaw_t = np.repeat(y_yaw_sel[np.newaxis,:], x_yaw.grid.shape[0], axis=0)
x_y_yaw_sel = np.maximum(x_yaw_t, y_yaw_t)

plt.figure(figsize=(13, 8))
plt.contourf(x_yaw.grid.coordinate_vectors[0], y_yaw.grid.coordinate_vectors[0], x_y_yaw_sel[:, :].T, levels=[-1e3, 0., 1e3])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

vx_vy_w_d_res_proj = -shp.project_onto(-vx_vy_w_d_result[:,:,:,:], 0, 1, 2)
vx_vy_w_d_grid_proj = vx_vy_w_d.grid.states[..., [0,1,2]]
vx_vy_w_d_grid_proj = vx_vy_w_d_grid_proj[:,:,:,0,:]
vx_vy_w_d_sel = vx_vy_w_d_result[:,10,3,:]

x_vx_vy_d_res_proj = -shp.project_onto(-x_vx_vy_d_result[:,:,:,:], 0, 1, 2)
x_vx_vy_d_grid_proj = x_vx_vy_d.grid.states[..., [0,1,2]]
x_vx_vy_d_grid_proj = x_vx_vy_d_grid_proj[:,:,:,0,:]
x_vx_vy_d_sel = x_vx_vy_d_result[:,:,10,:]

y_vx_vy_d_res_proj = -shp.project_onto(-y_vx_vy_d_result[:,:,:,:], 0, 1, 2)
y_vx_vy_d_grid_proj = y_vx_vy_d.grid.states[..., [0,1,2]]
y_vx_vy_d_grid_proj = y_vx_vy_d_grid_proj[:,:,:,0,:]
y_vx_vy_d_sel = y_vx_vy_d_result[:,:,10,:]

x_t = np.repeat(x_vx_vy_d_result[:,np.newaxis,:,10,:], y_vx_vy_d.grid.shape[0], axis=1)
y_t = np.repeat(y_vx_vy_d_result[np.newaxis,:,:,10,:], x_vx_vy_d.grid.shape[0], axis=0)
x_y_t = np.maximum(x_t, y_t)
x_y_t = -shp.project_onto(-x_y_t, 0, 1, 2)

x_y_t_grid = np.array(list(product(x_vx_vy_d.grid.coordinate_vectors[0], y_vx_vy_d.grid.coordinate_vectors[0], x_vx_vy_d.grid.coordinate_vectors[1])))

hj_tools.plot_set_3D(x_y_t_grid[:,0], 
            x_y_t_grid[:,1], 
            x_y_t_grid[:,2], 
            x_y_t.ravel(),
            ("x", "y", "v_x"))

exit()

hj_tools.plot_set_3D(vx_vy_w_d_grid_proj[..., 0].ravel(), 
            vx_vy_w_d_grid_proj[..., 1].ravel(), 
            vx_vy_w_d_grid_proj[..., 2].ravel(), 
            vx_vy_w_d_res_proj.ravel(),
            ("v_x", "v_y", "w"))

hj_tools.plot_set_3D(x_vx_vy_d_grid_proj[..., 0].ravel(), 
            x_vx_vy_d_grid_proj[..., 1].ravel(), 
            x_vx_vy_d_grid_proj[..., 2].ravel(), 
            x_vx_vy_d_res_proj.ravel(),
            ("x", "v_x", "v_y"))

hj_tools.plot_set_3D(y_vx_vy_d_grid_proj[..., 0].ravel(), 
            y_vx_vy_d_grid_proj[..., 1].ravel(), 
            y_vx_vy_d_grid_proj[..., 2].ravel(), 
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

