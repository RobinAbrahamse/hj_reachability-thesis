import jax.numpy as jnp
# from scipy.optimize import minimize
from jaxopt import ScipyBoundedMinimize

from hj_reachability import dynamics
from hj_reachability import sets


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

        self.m = 5.0639
        self.I_z = 0.0772
        self.I_t = 0.0004778
        self.l_r = 0.174
        self.l_f = 0.321 - self.l_r
        self.C = 8e-1

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
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(delta))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
        

    def open_loop_dynamics(self, state, time):
        x, y, yaw, v_x, v_y, yaw_rate, delta = state
        F_cr = -self.C * (v_y + yaw_rate*self.l_r)
        F_cf = -self.C * (v_y - yaw_rate*self.l_f)
        return jnp.array([
            v_x * jnp.cos(yaw) - v_y * jnp.sin(yaw),
            v_x * jnp.sin(yaw) + v_y * jnp.cos(yaw),
            yaw_rate,
            yaw_rate * v_y,
            -yaw_rate * v_x + 2/self.m * (F_cr + F_cf*jnp.cos(delta)),
            2/self.I_z * (self.l_f*F_cf - self.l_r*F_cr),
            0.
        ])

    def control_jacobian(self, state, time):
        x, y, yaw, v_x, v_y, yaw_rate, delta = state
        F_cf = self.C * (v_y - yaw_rate*self.l_f)
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
        return jnp.identity(7)

class X_vx_vy_d(dynamics.Dynamics):

    def __init__(self,
                 min_controls, 
                 max_controls,
                 min_disturbances, 
                 max_disturbances,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        self.phi_max = 26.0
        self.m = 5.0639
        self.I_z = 0.0772
        # self.I_t = 0.0004778
        self.l_r = 0.174
        self.l_f = 0.321 - self.l_r
        self.C = 8e-1

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

    ## Dynamics
    # Implements the affine dynamics 
    #   `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`.
    # 
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(delta))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
        

    def open_loop_dynamics(self, state, time):
        x, v_x, v_y, delta = state
        # F_cr = -self.C * (v_y + w*self.l_r)
        # F_cf = -self.C * (v_y - w*self.l_f)
        return jnp.array([
            0., # v_x * jnp.cos(yaw) - v_y * jnp.sin(yaw),
            0., # w * v_y,
            -2/self.m*self.C*v_y*(1 + jnp.cos(delta)), # -w * v_x + 2/self.m * (F_cr + F_cf*jnp.cos(delta)),
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
            [v_y],
            [-v_x - 2/self.m * self.C * (self.l_r - self.l_f*jnp.cos(delta))],
            [0.]
        ])
    
    def disturbance_dynamics(self, state, time, disturbance):
        x, v_x, v_y, delta = state
        yaw, w = disturbance
        return jnp.array([
            v_x * jnp.cos(yaw) - v_y * jnp.sin(yaw),
            w * v_y, # w * v_y,
            -w * v_x - 2/self.m*self.C*w*(self.l_r - self.l_f*jnp.cos(delta)), # -w * v_x + 2/self.m * (F_cr + F_cf*jnp.cos(delta)),
            0.
        ])
    
    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction
        disturbance_direction = grad_value[1:] @ self.affine_disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction

        w_disturbance = self.disturbance_space.extreme_point(jnp.array([0., disturbance_direction[0]]))[1]
        x, v_x, v_y, delta = state
        def yaw_disturbance_value_function(yaw):
            return grad_value[0] * (v_x * jnp.cos(yaw) - v_y * jnp.sin(yaw))
        
        n = 50
        yaws = jnp.array([i*(self.disturbance_space.hi[0] - self.disturbance_space.lo[0])/n for i in range(n)])
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
                 min_disturbances, 
                 max_disturbances,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        self.phi_max = 26.0
        self.m = 5.0639
        self.I_z = 0.0772
        # self.I_t = 0.0004778
        self.l_r = 0.174
        self.l_f = 0.321 - self.l_r
        self.C = 8e-1

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

    ## Dynamics
    # Implements the affine dynamics 
    #   `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`.
    # 
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(delta))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
        
    def open_loop_dynamics(self, state, time):
        y, v_x, v_y, delta = state
        # F_cr = -self.C * (v_y + w*self.l_r)
        # F_cf = -self.C * (v_y - w*self.l_f)
        return jnp.array([
            0., # v_x * jnp.sin(yaw) + v_y * jnp.cos(yaw),
            0., # w * v_y,
            -2/self.m*self.C*v_y*(1 + jnp.cos(delta)), # -w * v_x + 2/self.m * (F_cr + F_cf*jnp.cos(delta)),
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
        # return jnp.identity(4)
        x, v_x, v_y, delta = state
        return jnp.array([
            [v_y],
            [-v_x - 2/self.m * self.C * (self.l_r - self.l_f*jnp.cos(delta))],
            [0.]
        ])
    
    def disturbance_dynamics(self, state, time, disturbance):
        y, v_x, v_y, delta = state
        yaw, w = disturbance
        return jnp.array([
            v_x * jnp.sin(yaw) + v_y * jnp.cos(yaw), # NOT VIRT DISTURBANCE AFFINE
            w * v_y, # w * v_y,
            -w * v_x - 2/self.m*self.C*w*(self.l_r - self.l_f*jnp.cos(delta)), # -w * v_x + 2/self.m * (F_cr + F_cf*jnp.cos(delta)),
            0.
        ])
    
    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction
        disturbance_direction = grad_value[1:] @ self.affine_disturbance_jacobian(state, time)
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
                 min_disturbances, 
                 max_disturbances,
                 control_mode="max",
                 disturbance_mode="min",
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0] * 3
        if max_disturbances is None:
            max_disturbances = [0] * 3

        control_space = sets.Box(jnp.array([0.]),
                                 jnp.array([0.]))
        
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
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(delta))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
        

    def open_loop_dynamics(self, state, time):
        x, yaw = state
        return jnp.array([
            0.,
            0.
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.]
        ])

    def disturbance_jacobian(self, state, time):
        # disturbance: v_x, v_y, w
        x, yaw = state
        return jnp.array([
            [jnp.cos(yaw), -jnp.sin(yaw), 0.],
            [0.,           0.,            1]
        ])    

class Y_yaw(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0] * 3
        if max_disturbances is None:
            max_disturbances = [0] * 3

        if control_space is None:
            control_space = sets.Box(jnp.array([0.]),
                                     jnp.array([0.]))
        
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
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(delta))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
        

    def open_loop_dynamics(self, state, time):
        y, yaw = state
        return jnp.array([
            0.,
            0.
        ])

    def control_jacobian(self, state, time):
        y, yaw = state
        return jnp.array([
            [0.],
            [0.]
        ])

    def disturbance_jacobian(self, state, time):
        # disturbances: v_x, v_y, w
        y, yaw = state
        return jnp.array([
            [jnp.sin(yaw), jnp.cos(yaw), 0.],
            [0.,           0.,           1.]
        ])

class vx_vy_w_d(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_controls,
                 max_controls,
                #  min_ddelta, max_ddelta,
                #  min_acc, max_acc,
                #  min_disturbances=None, 
                #  max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        self.phi_max = 26.0
        self.m = 5.0639
        self.I_z = 0.0772
        # self.I_t = 0.0004778
        self.l_r = 0.174
        self.l_f = 0.321 - self.l_r
        self.C = 8e-1

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

    ## Dynamics
    # Implements the affine dynamics 
    #   `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`.
    # 
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(delta))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
        

    def open_loop_dynamics(self, state, time):
        v_x, v_y, yaw_rate, delta = state
        F_cr = -self.C * (v_y + yaw_rate*self.l_r)
        F_cf = -self.C * (v_y - yaw_rate*self.l_f)
        return jnp.array([
            yaw_rate * v_y,
            -yaw_rate * v_x + 2/self.m * (F_cr + F_cf*jnp.cos(delta)),
            2/self.I_z * (self.l_f*F_cf - self.l_r*F_cr),
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
        return jnp.identity(4)


class yaw_w(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):

        self.I_z = 0.0772
        self.l_r = 0.174
        self.l_f = 0.321 - self.l_r
        self.C = 8e-1

        if min_disturbances is None:
            min_disturbances = [0]
        if max_disturbances is None:
            max_disturbances = [0]

        if control_space is None:
            control_space = sets.Box(jnp.array([0.]),
                                     jnp.array([0.]))
        
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
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(delta))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
        

    def open_loop_dynamics(self, state, time):
        yaw, w = state
        # F_cr = -self.C * (v_y + w*self.l_r)
        # F_cf = -self.C * (v_y - w*self.l_f)
        return jnp.array([
            w,
            -2 * w / self.I_z * self.C * (self.l_f**2 + self.l_r**2) # 2/self.I_z * (self.l_f*F_cf - self.l_r*F_cr)
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
        ])

    def disturbance_jacobian(self, state, time):
        # disturbances: v_y
        return -2 / self.I_z * self.C * jnp.array([
            [0.],
            [self.l_f - self.l_r],
        ])
