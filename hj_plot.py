import plotly.graph_objects as go

def plot_set_3D(x, y, z, val):
    fig = go.Figure(data=go.Isosurface(x=x,
                                y=y,
                                z=z,
                                value=val,
                                colorscale="jet",
                                isomin=0,
                                surface_count=1,
                                isomax=0))

    fig.show()