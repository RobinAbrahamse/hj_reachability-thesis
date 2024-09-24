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
    coords = np.array(list(product(x, y, z)))
    layout = go.Layout(
        scene = dict(
                    xaxis = dict(
                        title=axes_titles[0]),
                    yaxis = dict(
                        title=axes_titles[1]),
                    zaxis = dict(
                        title=axes_titles[2]))
    )
    value_surface = go.Isosurface(x=coords[..., 0],
                                y=coords[..., 1],
                                z=coords[..., 2],
                                value=value.ravel(),
                                colorscale="jet",
                                isomin=0,
                                surface_count=1,
                                isomax=0)
    target_surface = go.Isosurface(x=coords[..., 0],
                                y=coords[..., 1],
                                z=coords[..., 2],
                                value=target.ravel(),
                                colorscale="reds",
                                isomin=0,
                                surface_count=1,
                                isomax=0)
    fig = go.Figure(data = target_surface, layout=layout)
    fig.add_trace(value_surface)
    fig.show()


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
        n_in = len(inputs)
        input_mesh = np.meshgrid(*inputs)
        a, b = self.linear_value_derivs(vf, x, disturbance)
        control_set = a + sum([input_mesh[i]*b[i] for i in range(n_in)])
        return control_set

    def linear_value_derivs(self, vf, x, disturbance):
        f = self.dynamics.open_loop_dynamics(x, None)
        g = self.dynamics.control_jacobian(x, None)
        d = 0. if disturbance is None else self.dynamics.disturbance_dynamics(x, None, disturbance)

        ix = self.nearest_index(x)
        dvdx = self.spatial_deriv(vf, ix)

        a = np.array(vf[tuple(ix)] + self.dt*(dvdx.T @ (f + d)))
        b = np.array(self.dt*(dvdx.T @ g))
        return a, b

    def nearest_index(self, x):
        x = np.array(x)
        # assert (x >= min_bounds).all() and (x <= max_bounds).all(), f'Point {x} is out of bounds'
        pos = np.zeros((len(x)))
        for axis in range(len(x)):
            pos[axis] = (x[axis] - self.grid_coords[axis][0]) / self.grid_spacings[axis]
            pos[axis] = pos[axis] % (self.shape[axis]) if axis in self.periodic_dims else pos[axis]
        ix = np.round(pos).astype(np.int32)
        ix = np.where(ix >= self.shape, np.array(self.shape)-1, ix)
        print(ix)
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
