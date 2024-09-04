import hj_reachability as hj
# import hj_reachability.shapes as shp
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def compute_brs(solver_settings, dynamics, grid, target, t):
    values = hj.step(solver_settings, dynamics, grid, 0., target, -t)
    return np.asarray(values)

def spatial_deriv(grid, vf, ix):
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
            leftv = (vf[ix_nxt[:axis] + (-1,) + ix_nxt[axis+1:]] if grid._is_periodic_dim[axis] else 
                      vf[ix] + sign*np.abs(vf[ix_nxt] - vf[ix]))
            rightv = vf[ix_nxt]
        elif ix[axis] == grid.shape[axis] - 1:
            leftv = vf[ix_prv]
            rightv = (vf[ix_prv[:axis] + (0,) + ix_prv[axis+1:]] if grid._is_periodic_dim[axis] else 
                      vf[ix] + sign*np.abs(vf[ix] - vf[ix_prv]))
        else:
            leftv = vf[ix_prv]
            rightv = vf[ix_nxt]

        axis_deriv = (rightv - leftv) / (2 * grid.spacings[axis])
        spatial_deriv.append(axis_deriv)
    return np.array(spatial_deriv)

def nearest_index(grid, x):
    x = np.array(x)
    # assert (x >= min_bounds).all() and (x <= max_bounds).all(), f'Point {x} is out of bounds'
    ix = np.array(grid.nearest_index(x), int)
    ix = np.where(ix >= grid.shape, np.array(grid.shape)-1, ix)
    return tuple(ix)

def lrcs(dynamics, grid, dt, vf, x, disturbance=None):
    f = dynamics.open_loop_dynamics(x, None)
    g = dynamics.control_jacobian(x, None)
    d = 0. if disturbance is None else dynamics.disturbance_dynamics(x, None, disturbance)

    ix = nearest_index(grid, x)
    dvdx = spatial_deriv(grid, vf, ix)

    a = np.array(vf[tuple(ix)] + dt*(dvdx.T @ (f + d)))
    b = np.array(dt*(dvdx.T @ g))
    return a, b

def plot_set_3D(x, y, z, val, axes_titles=("x", "y", "z")):
    layout = go.Layout(
        scene = dict(
                    xaxis = dict(
                        title=axes_titles[0]),
                    yaxis = dict(
                        title=axes_titles[1]),
                    zaxis = dict(
                        title=axes_titles[2]))
    )
    fig = go.Figure(data=go.Isosurface(x=x,
                                y=y,
                                z=z,
                                value=val,
                                colorscale="jet",
                                isomin=0,
                                surface_count=1,
                                isomax=0),
                                layout=layout)

    fig.show()


def plot_value_2D(X, Y, V, axes_labels=None):
    plt.figure(figsize=(13, 8))
    plt.contourf(X, Y, V)
    plt.colorbar()
    if axes_labels is not None:
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
    plt.show()
