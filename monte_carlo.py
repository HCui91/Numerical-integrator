import numpy as np
import random
# numba is used to speed up with multithreading
from numba import jit, njit, prange


@njit(parallel=True)
def flat_monte_carlo(func, z_min, z_max, epsilon):
    """
    Operate Monte Carlo integration on specified function. Random numbers are generated with a flat distribution function.
    ----------
    Input:
        func: the function of integrand
        z_min: lower limit of the integration
        z_max: upper limit of the integration
        epsilon: desired relative accuracy of the result
    Returns:
        new_I: numerical integral with required relative accuracy
        err: error of estimation of the integral
        evaluations: count of function evaluations"""
    # Setting global variables
    evaluation = 0  # Counting for the number of evaluations

    # Initial Trail with 10 points
    n = 10          # Starting with 10 random points
    func_sum = 0    # Sum of function value at random points
    func_sq = 0     # Sum of function value squared at random points
    for _ in prange(n):    # prange is used to paralellizing the loop
        # Evaluate the function at a random point in the range of integration
        temp = func(random.uniform(z_min, z_max))
        func_sum += temp
        func_sq += temp**2
    evaluation += n     # Update the number of evaluations
    # estimate the integral by averaging the function values at n sample points
    new_I = (z_max - z_min) * func_sum / n
    relative_accuracy = 1   # Assign any non-zero value to start the while loop
    # loop until reaching the required relative accuracy
    while relative_accuracy > epsilon and relative_accuracy != 0:
        # Adding 1% more points to the evaluation each loop
        n += int(0.01 * n) + 1
        old_I = new_I     # Store the integral result from previous iteration, later calculate the relative accuracy
        func_sum = 0    # Sum of function value at random points
        func_sq = 0     # Sum of function value squared at random points
        for _ in prange(n):  # prange for parallelizing
            # Evaluate the function at a random point in the range of
            # integration
            temp = func(random.uniform(z_min, z_max))
            func_sum += temp
            func_sq += temp**2
        evaluation += n
        new_I = (z_max - z_min) * func_sum / n
        # calculate the relative accuracy with previous result
        relative_accuracy = abs((new_I - old_I) / old_I)
        # print(n,new_I,relative_accuracy)    # Show realtime calculation
    # Evaluate the error of estimation using standard deviation
    err = np.sqrt(func_sq / n - (func_sum / n)**2) / np.sqrt(n)
    return new_I, err, evaluation


@jit(nopython=True)
def translation_rand():
    """
    Translate uniform random number into random number with probability density function
    of pdf(x) = -0.48 x + 0.98
    Note: cdf(x) = 0.98 x - 0.48 x^2  is used to transform the uniform random number.
    ----------
    Input:
        None
    Return:
        x: random number with pdf(x) = -0.48 x + 0.98"""
    return (-0.98 + np.sqrt(0.98**2 - 4 * 0.24 * random.random())) / \
        (-0.48)  # inverting cdf(x) back to x


@njit(parallel=True)
def importance_sampling_integration(func, z_min, z_max, epsilon):
    """
    Perform importance sampling in Monte Carlo integral estimation. Random variable function is defined as translation_rand().
    ----------
    Input:
        func: the function of integrand
        z_min: lower limit of the integration
        z_max: upper limit of the integration
        epsilon: desired relative accuracy of the result
    Returns:
        new_I: numerical integral with required relative accuracy
        err: error of estimation of the integral
        evaluations: count of function evaluations"""
    # Setting global variable
    evaluation = 0

    # Initial trail
    n = 10      # Starting with 10 random points
    func_sum = 0    # Sum of function value at random points
    func_sq = 0     # Sum of function value squared at random points
    for _ in prange(n):  # prange for paralelizing computation
        z = translation_rand()  # take a random point with specific distribution
        pdf = -.48 * z + 0.98
        # the weighted function value with pdf of the random point
        temp = func(z) / pdf
        func_sum += temp
        func_sq += temp**2
    # No need to multiply integration range because pdf is normalised
    new_I = func_sum / n
    relative_accuracy = 1   # Assign any non-zero value to start the while loop
    while relative_accuracy > epsilon and relative_accuracy != 0:
        n += int(0.01 * n) + 1  # increment of 1%
        old_I = new_I
        func_sum = 0    # Sum of function value at random points
        func_sq = 0     # Sum of function value squared at random points
        for _ in prange(n):
            z = translation_rand()
            pdf = -.48 * z + 0.98
            temp = func(z) / pdf
            func_sum += temp
            func_sq += temp**2
        new_I = func_sum / n
        evaluation += n
        relative_accuracy = abs((new_I - old_I) / old_I)
        # print(n,new_I,relative_accuracy)    # Show realtime calculation
    # Evaluate error of estimation of the integral
    err = np.sqrt(func_sq / n - (func_sum / n)**2) / np.sqrt(n)
    return new_I, err, evaluation

# Adaptive monte carlo involved in complicated array operation, which is
# not well supported by numba. Thus, numba is not operated in this
# function.


def adaptive_monte_carlo(func, z_min, z_max, epsilon):
    """
    Perform adaptive Monte Carlo algorithm to a specific function. Uniform random variable is used in this case.
    The calculation starts from 10 division of the original function range. Each step, it will divide the region which has the largest variance.

    Input:
        func: the function of integrand
        z_min: lower limit of the integration
        z_max: upper limit of the integration
        epsilon: desired relative accuracy of the result
    Returns:
        new_I: numerical integral with required relative accuracy
        err: error of estimation of the integral
        evaluations: count of function evaluations"""

    # However, we can speed up this small sampling process inside each
    # sub-interval
    @jit(nopython=True)
    def loop(upper, lower, func, sampling_size):
        elements = []
        for _ in range(sampling_size):
            z = random.uniform(lower, upper)
            elements.append(func(z))
        return elements

    def monte_carlo():  # Monte Carlo integration in each of the sub-interval
        var_array = []
        I_array = []
        for i in range(len(intervals) - 1):
            # random sampling in each of the interval
            elements = loop(
                intervals[i], intervals[i + 1], func, sampling_size)
            # integral of segment of integration
            average = sum(elements) / sampling_size
            # weight of integral is correspond to the width of the sub-interval
            weight = intervals[i + 1] - intervals[i]
            I_array.append(weight * average)       # add up the integral value
            # calculate the variance of this segment of integration
            var = sum((elements[i] - average)**2 for i in range(sampling_size))
            var_array.append(var)  # add variance to the array
        # return the integral value and variance of each sub-interval in an
        # array
        return I_array, var_array

    evaluation = 0

    n = 10   # number of divisions
    sampling_size = 100    # 1000 sampling points in each division

    # Initial trail
    intervals = np.linspace(z_min, z_max, n)
    I_array, var_array = monte_carlo()
    evaluation += (len(intervals) - 1) * sampling_size
    new_I = sum(I_array)
    relative_accuracy = 1     # assign a non-zero value of initial relative accuracy
    while relative_accuracy >= epsilon and relative_accuracy != 0:
        old_I = new_I
        # adaption
        # find the index of the largest variance
        largest_var_index = var_array.index(max(var_array))
        # removing the result of section with largest variance
        I_array = np.delete(I_array, largest_var_index)
        var_array = np.delete(var_array, largest_var_index)
        # divide sub-interval with the largest variance into 10 more
        # sub-intervals
        intervals = np.insert(intervals,
                              largest_var_index + 1,
                              np.linspace(intervals[largest_var_index],
                                          intervals[largest_var_index + 1],
                                          n,
                                          endpoint=False))
        intervals = np.delete(intervals, largest_var_index)
        # run Monte Carlo in the new intervals
        I_array, var_array = monte_carlo()
        new_I = sum(I_array)
        # calculate relative accuracy
        relative_accuracy = abs((new_I - old_I) / old_I)
        # amount of evaluations increases by the number of intervals * random
        # points in each interval
        evaluation += (len(intervals) - 1) * sampling_size
        # print((len(intervals)-1)*sampling_size,new_I,relative_accuracy)    #
        # show realtime evaluations
    err = 0
    for i in range(len(intervals) - 1):
        # sum up the variance of each interval
        err += ((intervals[i + 1] - intervals[i]) /
                (z_max - z_min))**2 * var_array[i]
    # divide the standard deviation by sqrt of n to get standard error (error
    # of estimation)
    err = np.sqrt(err / (len(intervals) * sampling_size))
    return new_I, err, evaluation
