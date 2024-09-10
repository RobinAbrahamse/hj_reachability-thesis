from scipy.integrate import odeint, solve_ivp
from scipy.optimize import least_squares
import numpy as np

g = 9.81
m = 5.0639
I_z = 0.0772
l_r = 0.174
l_f = 0.321 - l_r
e = 1e-1

def clip_mag(x, e):
    sign = x >= 0
    sign = sign.astype(int) * 2 - 1
    return sign * np.maximum(np.abs(x), e)

def f(t, y, input_data, theta): 
    """define the ODE system in terms of 
        dependent variable y,
        independent variable t, 
        input, and parameters"""
    # state variables
    v_x = y[0]
    v_y = y[1]
    w_z = y[2]

    # inputs
    (t_data, u_data) = input_data
    i = max(0, np.argmax(t_data>t) - 1) #np.abs(t_data - t).argmin()
    a_x = u_data[0,i]
    delta = u_data[1,i]

    # parameters
    C = theta[0]

    F_yr = -C * (v_y - w_z*l_r)/clip_mag(v_x, e)
    F_yf = -C * ((v_y + w_z*l_f)/clip_mag(v_x, e) - delta)

    # state space model
    v_x_dot = a_x + v_y*w_z# - F_yf*np.sin(delta)/m
    v_y_dot = (F_yf*np.cos(delta) + F_yr)/m - v_x*w_z
    w_z_dot = (l_f*F_yf - l_r*F_yr)/I_z

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
            w_z_dot)

def f_int(y0, t_data, input_data, theta):
    """definition of function for LS fit
        x gives evaluation points,
        theta is an array of parameters to be varied for fit"""
    # create an alias to f which passes the optional params    
    f2 = lambda y,t: f(y, t, input_data, theta)
    # breakpoint()
    # r = odeint(f2, y0, t_data, tfirst=True)
    r = solve_ivp(f2, (t_data[0], t_data[-1]), y0, t_eval=t_data, method='RK45').y
    # r = r.T
    return r[:,1:]



def create_eval_sections(data, start_idx=0, section_len=20, section_separation=10):
    y_data, t_data, u_data = data
    n = len(t_data)
    i = start_idx
    y_sections = []
    t_sections = []
    u_sections = []
    while i+section_len < n:
        y_sections.append(y_data[:,i:i+section_len])
        t_sections.append(t_data[i:i+section_len])
        u_sections.append(u_data[:,i:i+section_len])
        i += section_separation
    return (y_sections, t_sections, u_sections)

def f_resid(y_data, t_data, u_data, theta):
    """ function to pass to optimize.leastsq
        The routine will square and sum the values returned by 
        this function""" 
    y_sections, t_sections, u_sections = create_eval_sections((y_data, t_data, u_data), start_idx=8, section_len=20, section_separation=10)
    input_sections = list(zip(t_sections, u_sections))
    n_sections = len(t_sections)
    err = np.array([0.,0.,0.])
    for i in range(n_sections):
        model_output = f_int(y_sections[i][:,0], t_sections[i], input_sections[i], theta)
        weights = np.tile(np.array([2.0, 0.3, 2.5])[:,np.newaxis], (1,model_output.shape[1]))
        err += np.sum(
            ((y_sections[i][:,1:] - model_output)/weights)**2, axis=1)
    print(err)
    return err

def f_predic(y_data, t_data, u_data, theta):
    """ function to pass to optimize.leastsq
        The routine will square and sum the values returned by 
        this function""" 
    y_sections, t_sections, u_sections = create_eval_sections((y_data, t_data, u_data), start_idx=8, section_len=20, section_separation=10)
    input_sections = list(zip(t_sections, u_sections))
    n_sections = len(t_sections)
    out = []
    ts = []
    for i in range(n_sections):
        model_output = f_int(y_sections[i][:,0], t_sections[i], input_sections[i], theta)
        out.append(np.concatenate((y_sections[i][:,0][:,np.newaxis], model_output), axis=1))
        ts.append(t_sections[i])
    return (out,ts)


def estimate(y_data, t_data, u_data, theta_0, bounds):
    #solve the system
    f_est = lambda theta: f_resid(y_data, t_data, u_data, theta)
    sol = least_squares(f_est, theta_0, jac='3-point', bounds=bounds, xtol=1e-18, ftol=1e-15, method='dogbox', x_scale=1e-4)
    print(sol)
    return sol



if __name__ == "__main__":
    exec(open('process_rosbag.py').read())