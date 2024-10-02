import hj_reachability as hj
# import hj_reachability.shapes as shp
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from itertools import product

def compute_brs(solver_settings, dynamics, grid, target, t):
    values = hj.step(solver_settings, dynamics, grid, 0., target, -t)
    return np.asarray(values)

def back_project(grid_dims, subsys_value, subsys_idxs):
    idxs = np.array([np.newaxis] * len(grid_dims))
    idxs[subsys_idxs] = slice(None)
    pattern = np.array(grid_dims)
    pattern[subsys_idxs] = 1
    return np.tile(subsys_value[tuple(idxs)], pattern)

def plot_set_3D(x, y, z, value, target, axes_titles=("x", "y", "z")):
    coords = np.array(list(product(x, y, z)));
    layout = go.Layout(
        margin = dict(l=0, r=0, t=0, b=0),
        scene = dict(
                    xaxis = dict(
                        title=axes_titles[0]),
                    yaxis = dict(
                        title=axes_titles[1]),
                    zaxis = dict(
                        title=axes_titles[2]),
                    camera = dict(
                        center=dict(
                            x=0,
                            y=0,
                            z=-0.2
                        ),
                        eye=dict(
                            x=2,
                            y=0,
                            z=1)
                    )
                )
            )
    value_surface = go.Isosurface(x=coords[..., 0],
                                y=coords[..., 1],
                                z=coords[..., 2],
                                value=value.ravel(),
                                colorscale="oranges",
                                opacity=0.75,
                                showscale=False,
                                isomin=0,
                                surface_count=1,
                                isomax=0)
    target_surface = go.Isosurface(x=coords[..., 0],
                                y=coords[..., 1],
                                z=coords[..., 2],
                                value=target.ravel(),
                                colorscale="reds",
                                showscale=False,
                                isomin=0,
                                surface_count=1,
                                isomax=0)
    fig = go.Figure(data = target_surface, layout=layout)
    fig.add_trace(value_surface)
    return fig

def animation_images(x, y, z, val, tar, axes, folder_name='animations'):
    fig = plot_set_3D(x, y, z, val,tar, axes)
    layout = fig.layout
    for i, a in enumerate(np.linspace(0.,2*np.pi,120)):
        layout['scene']['camera']['eye']['x'] = 2*np.cos(a)
        layout['scene']['camera']['eye']['y'] = 2*np.sin(a)
        fig.update_layout(layout)
        fig.write_image(f"{folder_name}/img{i}.png")

def plot_value_2D(X, Y, V, axes_labels=None):
    plt.figure(figsize=(13, 8))
    plt.contourf(X, Y, V)
    plt.colorbar()
    if axes_labels is not None:
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
    plt.show()

class LRCS(object):
    def __init__(self, dynamics, dt, *args, periodic_dims=[]):
        self.dynamics = dynamics
        self.dt = dt
        self.periodic_dims = periodic_dims
        self.grid_coords = []
        self.grid_spacings = []
        for dim in args:
            self.grid_coords.append(dim)
            self.grid_spacings.append(dim[1] - dim[0])
        self.shape = [len(arr) for arr in self.grid_coords]

    def lrcs(self, inputs, vf, x, disturbance=None):
        dvdt, dvdu = self.linear_value_derivs(vf, x, disturbance)
        input_mesh = np.meshgrid(*inputs)
        control_set = dvdt + sum([input_mesh[i]*dvdu[i] for i in range(len(inputs))])
        return control_set

    def linear_value_derivs(self, vf, x, disturbance):
        f = self.dynamics.open_loop_dynamics(x, None)
        g = self.dynamics.control_jacobian(x, None)
        d = 0. if disturbance is None else self.dynamics.disturbance_dynamics(x, None, disturbance)

        ix = self.nearest_index(x)
        dvdx = self.spatial_deriv(vf, ix)
        dvdx_dt = self.dt*dvdx.T

        dvdt = np.array(vf[tuple(ix)] + dvdx_dt @ (f + d))
        dvdu = np.array(dvdx_dt @ g)
        return dvdt, dvdu

    def nearest_index(self, x):
        x = np.array(x)
        # assert (x >= min_bounds).all() and (x <= max_bounds).all(), f'Point {x} is out of bounds'
        pos = np.zeros((len(x)))
        for axis in range(len(x)):
            pos[axis] = (x[axis] - self.grid_coords[axis][0]) / self.grid_spacings[axis]
            pos[axis] = pos[axis] % (self.shape[axis]) if axis in self.periodic_dims else pos[axis]
        ix = np.round(pos).astype(np.int32)
        ix = np.where(ix >= self.shape, np.array(self.shape)-1, ix)
        return tuple(ix)

    def spatial_deriv(self, vf, ix):
        spatial_deriv = []
        for axis in range(len(ix)):
            ix_nxt = list(ix)
            ix_nxt[axis] += 1
            ix_nxt = tuple(ix_nxt)

            ix_prv = list(ix)
            ix_prv[axis] -= 1
            ix_prv = tuple(ix_prv)

            sign = np.sign(vf[ix])
            if ix[axis] == 0:
                leftv = (vf[ix_nxt[:axis] + (-1,) + ix_nxt[axis+1:]] if axis in self.periodic_dims else 
                        vf[ix] + sign*np.abs(vf[ix_nxt] - vf[ix]))
                rightv = vf[ix_nxt]
            elif ix[axis] == self.shape[axis] - 1:
                leftv = vf[ix_prv]
                rightv = (vf[ix_prv[:axis] + (0,) + ix_prv[axis+1:]] if axis in self.periodic_dims else 
                        vf[ix] + sign*np.abs(vf[ix] - vf[ix_prv]))
            else:
                leftv = vf[ix_prv]
                rightv = vf[ix_nxt]

            axis_deriv = (rightv - leftv) / self.grid_spacings[axis]
            spatial_deriv.append(axis_deriv)
        return np.array(spatial_deriv)