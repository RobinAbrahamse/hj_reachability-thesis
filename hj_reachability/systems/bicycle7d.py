import jax.numpy as jnp
# from scipy.optimize import minimize
from jaxopt import ScipyBoundedMinimize

from hj_reachability import dynamics
from hj_reachability import sets

e = 5e-1

m = 5.0639
I_z = 0.0772
l_r = 0.174
l_f = 0.321 - l_r
C = 24.5
# phi_max = 26.0
# I_t = 0.0004778

def clip_mag(x, e):
    sign = x >= 0
    sign = sign.astype(int) * 2 - 1
    return sign * jnp.maximum(jnp.abs(x), e)

class Bicycle7D(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_controls,
                 max_controls,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0] * 7
        if max_disturbances is None:
            max_disturbances = [0] * 7

        if control_space is None:
            control_space = sets.Box(jnp.array(min_controls),
                                     jnp.array(max_controls))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    ## Dynamics
    # Implements the affine dynamics 
    #   `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`.
    # 
    #   F_cr = -C * (v_y - w_z*l_r)/clip_mag(v_x, e)
    #   F_cf = -C * ((v_y + w_z*l_f)/clip_mag(v_x, e) - delta)
    #
    #   x_dot = v_x * jnp.cos(yaw) - v_y * jnp.sin(yaw)
    #   y_dot = v_x * jnp.sin(yaw) + v_y * jnp.cos(yaw)
    #   yaw_dot = w_z
    #   v_x_dot = w_z * v_y + a_x
    #   v_y_dot = -w_z * v_x + 1./m * (F_cr + F_cf*jnp.cos(delta))
    #   w_z_dot = 1./I_z * (l_f*F_cf - l_r*F_cr)
    #   delta_dot = w_s

    def open_loop_dynamics(self, state, time):
        x, y, yaw, v_x, v_y, w_z, delta = state
        F_cr = -C * (v_y - w_z*l_r)/clip_mag(v_x, e)
        F_cf = -C * ((v_y + w_z*l_f)/clip_mag(v_x, e) - delta)
        return jnp.array([
            v_x * jnp.cos(yaw) - v_y * jnp.sin(yaw),
            v_x * jnp.sin(yaw) + v_y * jnp.cos(yaw),
            w_z,
            w_z * v_y,
            -w_z * v_x + 1./m * (F_cr + F_cf*jnp.cos(delta)),
            1./I_z * (l_f*F_cf - l_r*F_cr),
            0.
        ])

    def control_jacobian(self, state, time):
        x, y, yaw, v_x, v_y, w_z, delta = state
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.]
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.zeros(7)


class X_vx_vy_d(dynamics.Dynamics):

    def __init__(self,
                 min_controls, 
                 max_controls,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0] * 2
        if max_disturbances is None:
            max_disturbances = [0] * 2

        if control_space is None:
            control_space = sets.Box(jnp.array(min_controls),
                                     jnp.array(max_controls))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    def open_loop_dynamics(self, state, time):
        x, v_x, v_y, delta = state
        # F_cr = -C * (v_y - w_z*l_r)/clip_mag(v_x, e)
        # F_cf = -C * ((v_y + w_z*l_f)/clip_mag(v_x, e) - delta)
        return jnp.array([
            0., # v_x * jnp.cos(yaw) - v_y * jnp.sin(yaw),
            0., # w * v_y,
            -C/m*(v_y*(1 + jnp.cos(delta))/clip_mag(v_x, e) - delta*jnp.cos(delta)), # -w * v_x + 1/m * (F_cr + F_cf*jnp.cos(delta)),
            0.
        ])

    def control_jacobian(self, state, time):
        x, v_x, v_y, delta = state
        return jnp.array([
            [0., 0.],
            [1., 0.],
            [0., 0.],
            [0., 1.]
        ])

    def affine_disturbance_jacobian(self, state, time):
        # disturbance: w
        x, v_x, v_y, delta = state
        return jnp.array([
            [0.],
            [v_y],
            [-v_x - C/m * (l_f*jnp.cos(delta) - l_r)/clip_mag(v_x, e)],
            [0.]
        ])
    
    def disturbance_dynamics(self, state, time, disturbance):
        x, v_x, v_y, delta = state
        yaw, w = disturbance
        return jnp.array([
            v_x * jnp.cos(yaw) - v_y * jnp.sin(yaw),
            w * v_y, # w * v_y,
            -w * v_x - C/m * w * (l_f*jnp.cos(delta) - l_r)/clip_mag(v_x, e), # -w * v_x + 1/m * (F_cr + F_cf*jnp.cos(delta)),
            0.
        ])
    
    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction
        disturbance_direction = grad_value @ self.affine_disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction

        w_disturbance = self.disturbance_space.extreme_point(jnp.array([0., disturbance_direction[0]]))[1]
        x, v_x, v_y, delta = state
        def yaw_disturbance_value_function(yaw):
            return grad_value[0] * (v_x * jnp.cos(yaw) - v_y * jnp.sin(yaw))
        
        n = 50
        yaws = jnp.array([self.disturbance_space.lo[0] + i*(self.disturbance_space.hi[0] - self.disturbance_space.lo[0])/n for i in range(n)])
        vals = yaw_disturbance_value_function(yaws)
        yaw_disturbance = 0

        if self.disturbance_mode == "min":
            idx = jnp.argmin(vals)
            yaw_disturbance = yaws[idx]
        else:
            idx = jnp.argmax(vals)
            yaw_disturbance = yaws[idx]
        
        return (self.control_space.extreme_point(control_direction),
                jnp.array([yaw_disturbance, w_disturbance]))
    
    def __call__(self, state, control, disturbance, time):
        """Implements the affine dynamics `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t)`."""
        return (self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control +
                self.disturbance_dynamics(state, time, disturbance))

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del value, grad_value_box  # unused
        # An overestimation; see Eq. (25) from https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.pdf.
        return (jnp.abs(self.open_loop_dynamics(state, time)) +
                jnp.abs(self.control_jacobian(state, time)) @ self.control_space.max_magnitudes +
                jnp.abs(self.disturbance_dynamics(state, time, self.disturbance_space.max_magnitudes)))


class Y_vx_vy_d(dynamics.Dynamics):

    def __init__(self,
                 min_controls, 
                 max_controls,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0] * 2
        if max_disturbances is None:
            max_disturbances = [0] * 2

        if control_space is None:
            control_space = sets.Box(jnp.array(min_controls),
                                     jnp.array(max_controls))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self
        
    def open_loop_dynamics(self, state, time):
        y, v_x, v_y, delta = state
        # F_cr = -C * (v_y - w_z*l_r)/clip_mag(v_x, e)
        # F_cf = -C * ((v_y + w_z*l_f)/clip_mag(v_x, e) - delta)
        return jnp.array([
            0., # v_x * jnp.sin(yaw) + v_y * jnp.cos(yaw),
            0., # w * v_y,
            -C/m*(v_y*(1 + jnp.cos(delta))/clip_mag(v_x, e) - delta*jnp.cos(delta)), # -w * v_x + 1/m * (F_cr + F_cf*jnp.cos(delta)),
            0.
        ])

    def control_jacobian(self, state, time):
        x, v_x, v_y, delta = state
        return jnp.array([
            [0., 0.],
            [1., 0.],
            [0., 0.],
            [0., 1.]
        ])

    def affine_disturbance_jacobian(self, state, time):
        # disturbance: w
        x, v_x, v_y, delta = state
        return jnp.array([
            [0.],
            [v_y],
            [-v_x - C/m * (l_f*jnp.cos(delta) - l_r)/clip_mag(v_x, e)],
            [0.]
        ])
    
    def disturbance_dynamics(self, state, time, disturbance):
        y, v_x, v_y, delta = state
        yaw, w = disturbance
        return jnp.array([
            v_x * jnp.sin(yaw) + v_y * jnp.cos(yaw), # NOT VIRT DISTURBANCE AFFINE
            w * v_y, # w * v_y,
            -w * v_x - C/m * w * (l_f*jnp.cos(delta) - l_r)/clip_mag(v_x, e), # -w * v_x + 1/m * (F_cr + F_cf*jnp.cos(delta)),
            0.
        ])
    
    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction
        disturbance_direction = grad_value @ self.affine_disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction

        w_disturbance = self.disturbance_space.extreme_point(jnp.array([0., disturbance_direction[0]]))[1]
        y, v_x, v_y, delta = state
        def yaw_disturbance_value_function(yaw):
            return grad_value[0] * (v_x * jnp.sin(yaw) + v_y * jnp.cos(yaw))
        
        n = 50
        yaws = jnp.array([self.disturbance_space.lo[0] + i*(self.disturbance_space.hi[0] - self.disturbance_space.lo[0])/n for i in range(n+1)])
        vals = yaw_disturbance_value_function(yaws)
        yaw_disturbance = 0

        if self.disturbance_mode == "min":
            idx = jnp.argmin(vals)
            yaw_disturbance = yaws[idx]
        else:
            idx = jnp.argmax(vals)
            yaw_disturbance = yaws[idx]
        
        return (self.control_space.extreme_point(control_direction),
                jnp.array([yaw_disturbance, w_disturbance]))
    
    def __call__(self, state, control, disturbance, time):
        """Implements the affine dynamics `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t)`."""
        return (self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control +
                self.disturbance_dynamics(state, time, disturbance))

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del value, grad_value_box  # unused
        # An overestimation; see Eq. (25) from https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.pdf.
        return (jnp.abs(self.open_loop_dynamics(state, time)) +
                jnp.abs(self.control_jacobian(state, time)) @ self.control_space.max_magnitudes +
                jnp.abs(self.disturbance_dynamics(state, time, self.disturbance_space.max_magnitudes)))


class X_yaw(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0] * 3
        if max_disturbances is None:
            max_disturbances = [0] * 3

        control_space = sets.Box(jnp.array([0., 0.]),
                                 jnp.array([0., 0.]))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    def open_loop_dynamics(self, state, time):
        x, yaw = state
        return jnp.array([
            0.,
            0.
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
        ])

    def disturbance_jacobian(self, state, time):
        # disturbance: v_x, v_y, w
        x, yaw = state
        return jnp.array([
            [jnp.cos(yaw), -jnp.sin(yaw), 0.],
            [0.,           0.,            1]
        ])
    
    def disturbance_dynamics(self, state, time, disturbance):
        return self.disturbance_jacobian(state, time) @ disturbance


class Y_yaw(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0] * 3
        if max_disturbances is None:
            max_disturbances = [0] * 3

        control_space = sets.Box(jnp.array([0., 0.]),
                                 jnp.array([0., 0.]))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    def open_loop_dynamics(self, state, time):
        y, yaw = state
        return jnp.array([
            0.,
            0.
        ])

    def control_jacobian(self, state, time):
        y, yaw = state
        return jnp.array([
            [0., 0.],
            [0., 0.],
        ])

    def disturbance_jacobian(self, state, time):
        # disturbances: v_x, v_y, w
        y, yaw = state
        return jnp.array([
            [jnp.sin(yaw), jnp.cos(yaw), 0.],
            [0.,           0.,           1.]
        ])

    def disturbance_dynamics(self, state, time, disturbance):
        return self.disturbance_jacobian(state, time) @ disturbance


class vx_vy_w_d(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_controls,
                 max_controls,
                #  min_disturbances=None, 
                #  max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        if control_space is None:
            control_space = sets.Box(jnp.array(min_controls),
                                     jnp.array(max_controls))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([0.] * 4),
                                         jnp.array([0.] * 4))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    def open_loop_dynamics(self, state, time):
        v_x, v_y, w, delta = state
        F_cr = -C * (v_y - w*l_r)/clip_mag(v_x, e)
        F_cf = -C * ((v_y + w*l_f)/clip_mag(v_x, e) - delta)
        return jnp.array([
            w * v_y,
            -w * v_x + 1/m * (F_cr + F_cf*jnp.cos(delta)),
            1/I_z * (l_f*F_cf - l_r*F_cr),
            0.
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.]
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.zeros(4)


class yaw_w_d(dynamics.Dynamics):

    def __init__(self,
                 min_controls,
                 max_controls,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0]
        if max_disturbances is None:
            max_disturbances = [0]

        if control_space is None:
            control_space = sets.Box(jnp.array(min_controls),
                                     jnp.array(max_controls))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    def open_loop_dynamics(self, state, time):
        yaw, w, delta = state
        # F_cr = -C * (v_y - w*l_r)/clip_mag(v_x, e)
        # F_cf = -C * ((v_y + w*l_f)/clip_mag(v_x, e) - delta)
        return jnp.array([
            w,
            0., # 1/I_z * (l_f*F_cf - l_r*F_cr)
            0.
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 1.]
        ])
    
    def disturbance_dynamics(self, state, time, disturbance):
        yaw, w, delta = state
        v_x, v_y = disturbance
        return jnp.array([
            0.,
            -C/I_z * (
                v_y * (l_f-l_r)/clip_mag(v_x, e) + 
                w * (l_f**2 + l_r**2)/clip_mag(v_x, e) + 
                delta*l_f),
            0.
        ])
        
    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction

        yaw, w, delta = state
        def w_v_y_disturbance_value_function(v_xs, v_ys):
            v_x_grid, v_y_grid = jnp.meshgrid(v_xs, v_ys)
            val = -v_y_grid/clip_mag(v_x_grid, e)*(l_f - l_r) - w/clip_mag(v_xs, e)*(l_f**2 + l_r**2) - delta*l_f
            return grad_value[1] * val
        
        n = 50
        v_xs = jnp.array([self.disturbance_space.lo[0] + i*(self.disturbance_space.hi[0] - self.disturbance_space.lo[0])/n for i in range(n)])
        v_ys = jnp.array([self.disturbance_space.lo[1] + i*(self.disturbance_space.hi[1] - self.disturbance_space.lo[1])/n for i in range(n)])
        vals = w_v_y_disturbance_value_function(v_xs, v_ys)
        v_x_disturbance = 0
        v_y_disturbance = 0

        if self.disturbance_mode == "min":
            idx, idy = jnp.unravel_index(vals.argmin(), vals.shape)
            v_x_disturbance = v_xs[idx]
            v_y_disturbance = v_ys[idy]
        else:
            idx, idy = jnp.unravel_index(vals.argmax(), vals.shape)
            v_x_disturbance = v_xs[idx]
            v_y_disturbance = v_ys[idy]
        
        return (self.control_space.extreme_point(control_direction),
                jnp.array([v_x_disturbance, v_y_disturbance]))
    
    def __call__(self, state, control, disturbance, time):
        """Implements the affine dynamics `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t)`."""
        return (self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control +
                self.disturbance_dynamics(state, time, disturbance))

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del value, grad_value_box  # unused
        # An overestimation; see Eq. (25) from https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.pdf.
        return (jnp.abs(self.open_loop_dynamics(state, time)) +
                jnp.abs(self.control_jacobian(state, time)) @ self.control_space.max_magnitudes +
                jnp.abs(self.disturbance_dynamics(state, time, self.disturbance_space.max_magnitudes)))

class X_yaw_w_d(dynamics.Dynamics):

    def __init__(self,
                 min_controls,
                 max_controls,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0]
        if max_disturbances is None:
            max_disturbances = [0]

        if control_space is None:
            control_space = sets.Box(jnp.array(min_controls),
                                     jnp.array(max_controls))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    def open_loop_dynamics(self, state, time):
        x, yaw, w, d = state
        return jnp.array([
            0.,
            w,
            0.,
            0.
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.],            
        ])
    
    def disturbance_dynamics(self, state, time, disturbance):
        x, yaw, w, delta = state
        v_x, v_y = disturbance
        return jnp.array([
            v_x*jnp.cos(yaw) - v_y*jnp.sin(yaw),
            0.,
            -C/I_z * (
                v_y * (l_f-l_r)/clip_mag(v_x, e) + 
                w * (l_f**2 + l_r**2)/clip_mag(v_x, e) + 
                delta*l_f),
            0.
        ])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction

        x, yaw, w, delta = state
        def w_v_y_disturbance_value_function(v_xs, v_ys):
            v_x_grid, v_y_grid = jnp.meshgrid(v_xs, v_ys)
            val0 = v_x_grid*jnp.cos(yaw) - v_y_grid*jnp.sin(yaw)
            val2 = -v_y_grid/clip_mag(v_x_grid, e)*(l_f - l_r) - w/clip_mag(v_xs, e)*(l_f**2 + l_r**2) - delta*l_f
            return grad_value[0] * val0 + grad_value[2] * val2
        
        n = 50
        v_xs = jnp.array([self.disturbance_space.lo[0] + i*(self.disturbance_space.hi[0] - self.disturbance_space.lo[0])/n for i in range(n)])
        v_ys = jnp.array([self.disturbance_space.lo[1] + i*(self.disturbance_space.hi[1] - self.disturbance_space.lo[1])/n for i in range(n)])
        vals = w_v_y_disturbance_value_function(v_xs, v_ys)
        v_x_disturbance = 0
        v_y_disturbance = 0

        if self.disturbance_mode == "min":
            idx, idy = jnp.unravel_index(vals.argmin(), vals.shape)
            v_x_disturbance = v_xs[idx]
            v_y_disturbance = v_ys[idy]
        else:
            idx, idy = jnp.unravel_index(vals.argmax(), vals.shape)
            v_x_disturbance = v_xs[idx]
            v_y_disturbance = v_ys[idy]
        
        return (self.control_space.extreme_point(control_direction),
                jnp.array([v_x_disturbance, v_y_disturbance]))
    
    def __call__(self, state, control, disturbance, time):
        """Implements the affine dynamics `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t)`."""
        return (self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control +
                self.disturbance_dynamics(state, time, disturbance))

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del value, grad_value_box  # unused
        # An overestimation; see Eq. (25) from https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.pdf.
        return (jnp.abs(self.open_loop_dynamics(state, time)) +
                jnp.abs(self.control_jacobian(state, time)) @ self.control_space.max_magnitudes +
                jnp.abs(self.disturbance_dynamics(state, time, self.disturbance_space.max_magnitudes)))


class Y_yaw_w_d(dynamics.Dynamics):

    def __init__(self,
                 min_controls,
                 max_controls,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0]
        if max_disturbances is None:
            max_disturbances = [0]

        if control_space is None:
            control_space = sets.Box(jnp.array(min_controls),
                                     jnp.array(max_controls))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    def open_loop_dynamics(self, state, time):
        y, yaw, w, d = state
        return jnp.array([
            0.,
            w,
            0.,
            0.
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.],            
        ])
    
    def disturbance_dynamics(self, state, time, disturbance):
        x, yaw, w, delta = state
        v_x, v_y = disturbance
        return jnp.array([
            v_x*jnp.sin(yaw) + v_y*jnp.cos(yaw),
            0.,
            -C/I_z * (
                v_y * (l_f-l_r)/clip_mag(v_x, e) + 
                w * (l_f**2 + l_r**2)/clip_mag(v_x, e) + 
                delta*l_f),
            0.
        ])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction

        y, yaw, w, delta = state
        def w_v_y_disturbance_value_function(v_xs, v_ys):
            v_x_grid, v_y_grid = jnp.meshgrid(v_xs, v_ys)
            val0 = v_x_grid*jnp.sin(yaw) + v_y_grid*jnp.cos(yaw)
            val2 = -v_y_grid/clip_mag(v_x_grid, e)*(l_f - l_r) - w/clip_mag(v_xs, e)*(l_f**2 + l_r**2) - delta*l_f
            return grad_value[0] * val0 + grad_value[2] * val2
        
        n = 50
        v_xs = jnp.array([self.disturbance_space.lo[0] + i*(self.disturbance_space.hi[0] - self.disturbance_space.lo[0])/n for i in range(n)])
        v_ys = jnp.array([self.disturbance_space.lo[1] + i*(self.disturbance_space.hi[1] - self.disturbance_space.lo[1])/n for i in range(n)])
        vals = w_v_y_disturbance_value_function(v_xs, v_ys)
        v_x_disturbance = 0
        v_y_disturbance = 0

        if self.disturbance_mode == "min":
            idx, idy = jnp.unravel_index(vals.argmin(), vals.shape)
            v_x_disturbance = v_xs[idx]
            v_y_disturbance = v_ys[idy]
        else:
            idx, idy = jnp.unravel_index(vals.argmax(), vals.shape)
            v_x_disturbance = v_xs[idx]
            v_y_disturbance = v_ys[idy]
        
        return (self.control_space.extreme_point(control_direction),
                jnp.array([v_x_disturbance, v_y_disturbance]))
    
    def __call__(self, state, control, disturbance, time):
        """Implements the affine dynamics `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t)`."""
        return (self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control +
                self.disturbance_dynamics(state, time, disturbance))

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del value, grad_value_box  # unused
        # An overestimation; see Eq. (25) from https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.pdf.
        return (jnp.abs(self.open_loop_dynamics(state, time)) +
                jnp.abs(self.control_jacobian(state, time)) @ self.control_space.max_magnitudes +
                jnp.abs(self.disturbance_dynamics(state, time, self.disturbance_space.max_magnitudes)))
