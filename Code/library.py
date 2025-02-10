import numpy as np
from scipy.integrate import solve_ivp
from ddeint import ddeint
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from geneticalgorithm import geneticalgorithm as ga

def solve_differential_equation(model, t_end, initial_condition, params):
    """
    Solves a differential equation defined by the model function.
    
    Parameters:
    - model: A function that defines the ODE system. It should accept time t, state y, and params.
    - t_end: The end time for the simulation.
    - initial_condition: The initial state of the system (e.g., [y0]).
    - params: A dictionary or list of parameters needed by the model function.
    
    Returns:
    - t: Array of time points.
    - y: Array of solution values for each time point.
    """

    # Define a wrapper function that incorporates parameters
    def model_with_params(t, y):
        return model(t, y, *params)

    # Define the time span for the simulation
    t_span = (0, t_end)

    # Solve the ODE
    solution = solve_ivp(model_with_params, t_span, initial_condition, dense_output=True)

    # Extract time points and solution values
    t = np.linspace(0, t_end, 100)  # Generate 100 time points for the output
    y = solution.sol(t)  # Evaluate the solution at the given time points

    return t, y


def solve_dde_equation(model, t_end, initial_history, params):
    """
    Solves a delay differential equation defined by the model function.
    
    Parameters:
    - model: A function that defines the DDE system. It should accept state y, time t, and params.
    - t_end: The end time for the simulation.
    - initial_history: A function that returns the initial history of the system state.
    - params: list of parameters needed by the model function.
    
    Returns:
    - t: Array of time points.
    - y: Array of solution values for each time point.
    """
    t = np.linspace(0, t_end, 1000)
    y = ddeint(model, initial_history, t, fargs=params)
    return t, y



def fit_OGTT_ms(t_data, glucose_data, pmin, pmax):
    def callable_function(t, absortion, glucose_rate, insulin_rate, delay):
        t_end = 120
        bolus = 0.015
        params = (bolus, absortion, glucose_rate, insulin_rate, delay)
        tt, yy = solve_dde_equation(glucose_regulation_model, t_end, glucose_initial_history, params)
        fnctn = CubicSpline(tt,yy)
        return fnctn(t)[:, 0]

    popt, pcov = curve_fit(callable_function, t_data, glucose_data, bounds=(pmin, pmax), method="dogbox")

    return popt

def fit_OGTT_ga(t_data, glucose_data):
    def callable_function(t, absortion, glucose_rate, insulin_rate, delay):
        t_end = 120
        bolus = 0.015
        params = (bolus, absortion, glucose_rate, insulin_rate, delay)
        tt, yy = solve_dde_equation(glucose_regulation_model, t_end, glucose_initial_history, params)
        fnctn = CubicSpline(tt,yy)
        return fnctn(t)[:, 0]

    def error(par):
        return (np.sum(callable_function(t_data, *par) - glucose_data))**2

    bounds = np.array([[0.075, .1],[0.01, .04],[0.01, .04],[4, 6]])
    model=ga(function=error,dimension=4,variable_type='real',variable_boundaries=bounds)
    model.run()
    return model.output_dict

    
def exponential_decay_model(t, y, decay_rate):
    """
    Defines the exponential decay model.
    
    Parameters:
    - t: Time variable (not used in this model).
    - y: Current state (amount).
    - decay_rate: The decay rate of the model.
    
    Returns:
    - dydt: The derivative of y with respect to time.
    """
    dydt = -decay_rate * y  # Exponential decay equation
    return dydt


def logistic_growth_model(t, y, r, K):
    """
    Defines the logistic growth model.
    
    Parameters:
    - t: Time variable (not used in this model).
    - y: Current population size.
    - r: Growth rate.
    - K: Carrying capacity of the environment.
    
    Returns:
    - dydt: The derivative of y with respect to time.
    """
    dydt = r * y * (1 - y / K)  # Logistic growth equation
    return dydt


def gene_expression_model(t, y, basal_rate, maximum_rate):
    """
    Defines the gene expression model with positive feedback regulation.
    
    Parameters:
    - t: Time variable (not used in this model).
    - y: Current gene expression level.
    - basal_rate: Basal expression rate.
    - maximum_rate: Maximum expression rate.
    
    Returns:
    - dydt: The derivative of y with respect to time.
    """
    dydt = basal_rate + maximum_rate * y**4 / (y**4 + 1) - y  # Gene expression equation
    return dydt
    

def glucose_regulation_model(y, t, bolus, absortion, glucose_rate, insulin_rate, delay):
    """
    Defines the glucose regulation model.
    
    Parameters:
    - t: Time variable.
    - y: Current glucose level.
    - bolus: Glucose.
    - absortion: Intestinal absortion rate.
    - glucose_rate: Rate constant for insulin-independent glucose consumption
    - insulin_rate: Rate constant for unsulin-dependent glucose consumption
    - delay: Time delay of insulin-dependent response
    
    Returns:
    - dydt: The derivative of y with respect to time.
    """
    return bolus*t*np.exp(-absortion*t) + glucose_rate*(1 - y(t)) + insulin_rate*(1 - y(t-delay))


def glucose_initial_history(t):
    """
    Defines the initial history of the system state for the glucose regulation model.
    
    Parameters:
    - t: Time variable (not used in this model).
    
    Returns:
    - 1: Assuming that the system has been in its basal state for a while.
    """
    return 1


