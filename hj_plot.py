import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
