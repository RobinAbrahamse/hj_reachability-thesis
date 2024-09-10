from scipy.integrate import odeint, solve_ivp
from scipy.optimize import least_squares
import numpy as np

g = 9.81
m = 5.0639
I_z = 0.0772
I_t = 0.0004778
l_r = 0.174
l_f = 0.321 - l_r
r_e = 0.051
F_x = 0.
e = 1e-4
gear_ratio = 25.85
torque_const = 0.0053
nominal_voltage = 9.3
R = 0.006
max_velocity = 3.6

def vel2torque(vel, w_t):
    # returns torque on wheel
    w_motor = w_t*gear_ratio
    input_voltage = vel/max_velocity*nominal_voltage
    I = (input_voltage - w_motor*torque_const)/R
    return I * torque_const * gear_ratio / 100

def f(t, y, input_data, theta): 
    """define the ODE system in terms of 
        dependent variable y,
        independent variable t, 
        input, and parameters"""
    # state variables
    v_x = y[0]
    v_y = y[1]
    w_z = y[2]
    w_t = y[3]

    # inputs
    (t_data, u_data) = input_data
    i = max(0, np.argmax(t_data>t) - 1) #np.abs(t_data - t).argmin()
    T = vel2torque(u_data[0,i], w_t)
    delta = u_data[1,i]
    # print(t, w_t, w_t*gear_ratio*torque_const)

    # parameters
    h = theta[0]
    mu0 = theta[1]
    C_ki = theta[2]
    C_ai = theta[3]

    if abs(v_x) < 5e-2 and abs(w_t) < 3 and False:
        F_t = 0.5*T*r_e
        F_xf = F_t/2.0
        F_xr = F_t/2.0
        F_yf = 0.
        F_yr = 0.
    else:
        # Dugoff tire model
        F_zf = l_r/(l_f + l_r)*m*g + F_x*h/l_f
        F_zr = l_f/(l_f + l_r)*m*g - F_x*h/l_r
        # k_i = np.clip((w_t*r_e - v_x)/max(abs(v_x), 1e-1), -100, 100) # if abs(v_x) > 1e-2 else np.sign(T)
        k_i = (w_t*r_e - v_x)/max(abs(v_x), 1e-1)
        a_f = (v_y - l_f*w_z)/max(abs(v_x), 1e-1) # if abs(v_x) > 1e-2 else 0
        a_r = (v_y + l_r*w_z)/max(abs(v_x), 1e-1) # if abs(v_x) > 1e-2 else 0
        # Vs = v_x * (k_i**2 + a_i**2)**0.5
        mu = mu0 #* (1 - 0.01*Vs)
        lam_f = mu*F_zf*(1+k_i)/(2*((C_ki*k_i)**2 + (C_ai*a_f)**2)**0.5) if abs(k_i) > e or abs(a_f) > e else 1.
        lam_r = mu*F_zr*(1+k_i)/(2*((C_ki*k_i)**2 + (C_ai*a_r)**2)**0.5) if abs(k_i) > e or abs(a_r) > e else 1.
        f_f = (2-lam_f)*lam_f if lam_f < 1 else 1.
        f_r = (2-lam_r)*lam_r if lam_r < 1 else 1.

        F_xf = C_ki*k_i/(1+k_i)*f_f if abs(1+k_i) > e else 0.
        F_xr = C_ki*k_i/(1+k_i)*f_r if abs(1+k_i) > e else 0.
        F_yf = C_ai*a_f/(1+k_i)*f_f if abs(1+k_i) > e else 0.
        F_yr = C_ai*a_r/(1+k_i)*f_r if abs(1+k_i) > e else 0.

        # print(F_xf, F_xr, F_yf, F_yr)

    # F_zf = l_r/(l_f + l_r)*m*g
    # F_zr = l_f/(l_f + l_r)*m*g

    # linear tire model
    # F_t = 0.995*T/r_e
    # F_xf = F_t/2.0
    # F_xr = F_t/2.0
    # F_yf = C_ai*np.arctan(a_f)
    # F_yr = C_ai*np.arctan(a_r)

    # state space model
    # v_x_dot = (F_xf*np.cos(delta) + F_xr + F_yf*np.sin(delta))/m + v_y*w_z
    # v_y_dot = (-F_xf*np.sin(delta) + F_yf*np.cos(delta) + F_yr)/m - v_x*w_z
    # w_z_dot = (l_f*(-F_xf*np.sin(delta) + F_yf*np.cos(delta)) - l_r*F_yr)/I_z
    # w_z_dot = (l_r*F_yr - l_f*(F_yf*np.cos(delta) - F_xf*np.sin(delta)))/I_z

    # test
    v_x_dot = (F_xf*np.cos(delta) + F_xr - F_yf*np.sin(delta))/m + v_y*w_z
    v_y_dot = (F_yf*np.cos(delta) + F_xf*np.sin(delta) + F_yr)/m - v_x*w_z
    w_z_dot = (l_f*(F_yf*np.cos(delta) - F_xf*np.sin(delta) - l_r*F_yr))/I_z
    w_t_dot = (T - (F_xf + F_xr)*r_e)/I_t

    # print("\ntime: ", t)
    # print("input: ", (u_data[0,i], T, delta))
    # print("state: ", y)
    # print("tire forces:", (F_xf, F_xr, F_yf, F_yr))
    # print((v_x_dot,
    #         v_y_dot,
    #         w_z_dot,
    #         w_t_dot))

    return (v_x_dot,
            v_y_dot,
            w_z_dot,
            w_t_dot)

def f_int(y0, t_data, input_data, theta):
    """definition of function for LS fit
        x gives evaluation points,
        theta is an array of parameters to be varied for fit"""
    # create an alias to f which passes the optional params    
    f2 = lambda y,t: f(y, t, input_data, theta)
    # breakpoint()
    # r = odeint(f2, y0, t_data, tfirst=True)
    r = solve_ivp(f2, (t_data[0], t_data[-1]), y0, t_eval=t_data, method='LSODA').y
    # r = r.T
    return r[:,1:]



def create_eval_sections(data, start_idx=0, section_len=20, section_separation=10):
    y_data, t_data, u_data = data
    n = len(t_data)
    i = start_idx
    y_sections = []
    t_sections = []
    u_sections = []
    while i+section_len <= n:
        y_sections.append(y_data[:,i:i+section_len])
        t_sections.append(t_data[i:i+section_len])
        u_sections.append(u_data[:,i:i+section_len])
        i += section_separation
    return (y_sections, t_sections, u_sections)

def f_resid(y_data, t_data, u_data, theta):
    """ function to pass to optimize.leastsq
        The routine will square and sum the values returned by 
        this function""" 
    y_sections, t_sections, u_sections = create_eval_sections((y_data, t_data, u_data), start_idx=8, section_len=20)
    input_sections = list(zip(t_sections, u_sections))
    n_sections = len(t_sections)
    # err = []
    err = np.array([0.,0.,0.,0.])
    for i in range(n_sections):
        model_output = f_int(y_sections[i][:,0], t_sections[i], input_sections[i], theta)
        # err += [np.sum(np.sum((y_sections[i][:,1:] - model_output)**2, axis=0)**0.5)] # TODO: normalize/weight errors in dimensions
        err += np.sum((y_sections[i][:,1:] - model_output)**2, axis=1) # TODO: normalize/weight errors in dimensions
    print(err)
    return err

def estimate(y_data, t_data, u_data, theta_0, bounds):
    #solve the system
    f_est = lambda theta: f_resid(y_data, t_data, u_data, theta)
    sol = least_squares(f_est, theta_0, jac='3-point', bounds=bounds, xtol=1e-8, ftol=1e-5)
    print(sol)
    return sol



if __name__ == "__main__":
    exec(open('process_rosbag.py').read())