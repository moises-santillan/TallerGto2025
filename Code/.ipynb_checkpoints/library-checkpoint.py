import numpy as np
from scipy.integrate import solve_ivp

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

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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

