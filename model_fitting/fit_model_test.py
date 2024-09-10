from scipy.integrate import odeint
from scipy.optimize import least_squares
import numpy as np


def f_test(y, t, input_data, theta):
    y1 = y[0]
    y2 = y[1]

    a = theta[0]
    b = theta[1]

    (t_data, u_data) = input_data
    i = max(0, np.argmax(t_data>t) - 1)
    u = u_data[i]

    y1_dot = a*y1 + 20.0*y2**-3 + u
    y2_dot = b*y2/(1+a*y1)

    return (y1_dot, y2_dot)

def f_int_test(y0, t_data, input_data, theta):
    f2 = lambda y,t: f_test(y, t, input_data, theta)
    # breakpoint()
    # calculate ode solution, return values for each entry of "x"
    r = odeint(f2, y0, t_data)
    # in this case, we only need one of the dependent variable values
    return r[:,:].T

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
        u_sections.append(u_data[i:i+section_len])
        i += section_separation
    return (y_sections, t_sections, u_sections)

def f_resid_test(t_data, u_data, theta):
    """ function to pass to optimize.leastsq
        The routine will square and sum the values returned by 
        this function""" 
    input_data = (t_data, u_data)
    theta_real = [0.1, 5.0]
    y0 = [2.0, 0.2]
    y_data = f_int_test(y0, t_data, input_data, theta_real)

    y_sections, t_sections, u_sections = create_eval_sections((y_data, t_data, u_data), section_len=40, section_separation=20)
    input_sections = list(zip(t_sections, u_sections))
    n_sections = len(t_sections)
    err = []
    for i in range(n_sections):
        model_output = f_int_test(y_sections[i][:,0], t_sections[i], input_sections[i], theta)
        err += [np.sum(np.sum((y_sections[i] - model_output)**2, axis=0)**0.5)]
    # print(err)
    return err

def estimate_test():
    #solve the system
    n = 500
    t_data = np.arange(0.0, 5.0, 5.0/n)
    u_data = [12.3] * n
    # u_data = np.concatenate(([0.0] * 100, [10.0] * 100, [0.2] * 300))
    theta_0 = [2.9, 10.0]

    f_est = lambda theta: f_resid_test(t_data, u_data, theta)
    # sol = least_squares(f_est, theta_0, jac='3-point', bounds=bounds)
    bounds = ([0.01, 0.5], [3.0, 20.0])
    sol = least_squares(f_est, theta_0, jac='2-point', bounds=bounds)
    print(sol)
    return sol


if __name__ == "__main__":
    sol = estimate_test()