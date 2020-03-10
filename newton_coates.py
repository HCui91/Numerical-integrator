import numpy as np


def trapezine(func, z_min, z_max, epsilon):
    """
    Perform extended trapezoidal rule to calculate numerical integration

    Input:
        func: the function to be integrated
        z_min: lower limit of the integration
        z_max: upper limit of the integration
        epsilon: desired accuracy of the integration. e.g. 1e-8
    Returns:
        new_I: the numerical integration result with desired accuracy
        err: error of the process O(h^3)"""
    # Setting global variable
    evaluation = 0

    # Initial values of iteration
    h = z_max - z_min  # range of integration
    n = 2       # number of points
    # Initial value of integration with 1 step
    new_I = h * (func(z_max) + func(z_min)) / 2
    relative_accuracy = 1    # Assign a non-zero value to start the loop
    evaluation += 2

    # iteration until reach the desired relative accuracy or reach the machine
    # accuracy (machine epsilon)
    while relative_accuracy >= epsilon:
        h = h / 2  # update step-size
        z_array = np.arange(z_min, z_max + h, h)
        n = len(z_array)
        old_I = new_I  # update compared value
        # Half the previous value and add new points - reduce the repeated
        # evaluations of function
        new_I = old_I / 2 + h * \
            sum(func(z_array[2 * i - 1]) for i in range(1, int((n - 1) / 2) + 1))
        relative_accuracy = abs((new_I - old_I) / old_I)
        # Half of the terms in the new integral is "truly evaluated" in this
        # iteration step
        evaluation += int(n / 2 - 1)
        # print(n,h,new_I,relative_accuracy)
    # Evaluating error of the integration
    err = new_I * 0.5 * relative_accuracy

    # number of terms evaluated
    print("Number of evaluated terms: ", int(evaluation))
    return new_I, err


def simpson(func, z_min, z_max, epsilon):
    """
    Perform extended Simpson's rule to calculate numerical integration value.

    Input:
        func: the function to be integrated
        z_min: lower limit of the integration
        z_max: upper limit of the integration
        epsilon: desired accuracy of the integration. e.g. 1e-8
    Returns:
        new_I: the numerical integration result with desired accuracy
        err: error of the process O(h^5)"""
    # Setting global variable
    evaluation = 0
    # Initial values
    h = z_max - z_min
    # Implement the trapezoidal summation for the simpson's rule
    # Inital summation of 2-point trapezine
    T_j = h * (func(z_max) + func(z_min)) / 2
    h = h / 2
    # Update z_array just like trapezoidal
    z_array = np.arange(z_min, z_max + h, h)
    n = len(z_array)
    # Intial iterated trapezodial term
    T_j1 = T_j / 2 + h * sum(func(z_array[int(2 * t - 1)])
                             for t in range(1, int((n - 1) / 2) + 1))
    # Inital integral of simpson's method with 2 terms from trapezoidal method
    new_I = 4 * T_j1 / 3 - T_j / 3
    relative_accuracy = 1      # Assign a non-zero value to old_I to start the while loop
    evaluation += 3
    # Iterate until reach desired accuracy
    while relative_accuracy >= epsilon:
        old_I = new_I
        h = h / 2         # Similar to trapezoidal, half the step-size each loop
        # update z-array just like trapezoidal
        z_array = np.arange(z_min, z_max + h, h)
        n = len(z_array)
        T_j = T_j1        # update the T(rapezodial) summation
        T_j1 = T_j / 2 + h * \
            sum(func(z_array[int(2 * t - 1)]) for t in range(1, int((n - 1) / 2) + 1))
        new_I = 4 * T_j1 / 3 - T_j / 3
        evaluation += n // 2         # Evaluation number added similar to trapezoidal rule
        relative_accuracy = abs((new_I - old_I) / old_I)
    # Evaluate the error
    err = new_I * 0.5 * relative_accuracy

    # number of terms evaluated
    # each loop uses 2 generations of trapezoidal summation so evaluation is
    # 1.5 times of the number of terms in the last trapezoidal summation
    print("Number of evaluated terms: ", int(evaluation))
    return new_I, err
