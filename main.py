###################################
#       Numerical integrator      #
###################################

import numpy as np
import newton_coates as nc
import monte_carlo as mc
from numba import jit   # jit for speed up numpy numerical operations
import time  # evaluate the time used to find the integral


@jit(nopython=True)
def Psi2(z):
    """
    Time-dependent 1D QM wavefunction of a single particle - squared.
    Input:
        z: spacial position of the particle
    Output:
        Psi^2(z): modulus squared value of the wavefunction at postion z.
    """
    return np.exp(-z**2) / (np.pi**0.5) # Modify your integrand here!


print("####################")
print("# 1D QM integrator #")
print("####################")

# common argument used in the function [z_min,z_max,epsilon]
z_min = float(input("Input your lower limit: "))
z_max = float(input("Input your upper limit: "))
epsilon = float(input("Input your desired accuracy e.g. 1e-4: "))
arg = [z_min, z_max, epsilon]

# Method selector
print()
print("Select the definite integrating method:")
print("1: Extended Trapezoidal rule")
print("2: Extended Simpson's rule")
print("3: Monte Carlo with flat pdf")
print("4: Monte Carlo with importance sampling")
print("5: Adaptive Monte Carlo")
choice = input("Input your choice: ")
print()

if choice == "1":
    print("Extended Trapezoidal rule")
    trapezine_result, trapezine_err = nc.trapezine(Psi2, *arg)
    print(
        "Integration with extended trapezine method:",
        trapezine_result,
        "with estimated error:",
        trapezine_err)
elif choice == "2":
    print("Extended Simpson's rule")
    simpson_result, simpson_err = nc.simpson(Psi2, *arg)
    print("Integration with extended Simpson's method:",
          simpson_result, "with estimated error:", simpson_err)
elif choice == "3":
    print("Monte Carlo integration with flat pdf")
    n = int(input("How many times of the method do you want to repeat? "))
    # setup array for recording results and evaluations used
    flat_monte_carlo_result = np.zeros(n)
    flat_err = np.zeros(n)
    flat_eva = np.zeros(n)
    start = time.time()
    for i in range(n):  # repeat the integrator for n times
        flat_monte_carlo_result[i], flat_err[i], flat_eva[i] = mc.flat_monte_carlo(
            Psi2, *arg)
        print(
            "Run {0}".format(
                i + 1),
            ": Result: ",
            flat_monte_carlo_result[i],
            ", standard error: ",
            flat_err[i],
            "function evaluation:",
            flat_eva[i])
    end = time.time()
    # take the mean of n integration values
    mean = np.mean(flat_monte_carlo_result)
    # find the standard error of the mean
    error = np.std(flat_monte_carlo_result) / np.sqrt(n)
    # Mean of the result with error of the mean and also the error of the
    # estimation averaged.
    print("Mean of results:", mean, ", with error of the mean: ",
          error, " and averaged standard error: ", np.mean(flat_err))
    print(
        "Mean of function evaluations: ",
        np.mean(flat_eva),
        " with standard error of the mean: ",
        np.std(flat_eva) /
        np.sqrt(n))  # Mean of evaluations with standard error of the mean.
    print("Average time usage: ", (end - start) / n, " seconds.")
elif choice == "4":
    print("Monte Carlo integration with importance sampling")
    n = int(input("How many times of the method do you want to repeat? "))
    # setup array for recording results and evaluations used
    importance_sampling_result = np.zeros(n)
    imp_err = np.zeros(n)
    imp_eva = np.zeros(n)
    start = time.time()
    for i in range(n):  # repeat the integrator for n times
        importance_sampling_result[i], imp_err[i], imp_eva[i] = mc.importance_sampling_integration(
            Psi2, *arg)
        print(
            "Run {0}".format(
                i + 1),
            ": Result: ",
            importance_sampling_result[i],
            ", standard error: ",
            imp_err[i],
            "function evaluation:",
            imp_eva[i])
    end = time.time()
    # take the mean of n integration values
    mean = np.mean(importance_sampling_result)
    # find the standard error of the mean
    error = np.std(importance_sampling_result) / np.sqrt(n)
    # Mean of the result with error of the mean and also the error of the
    # estimation averaged.
    print("Mean of results:", mean, ", with error of the mean: ",
          error, " and averaged standard error: ", np.mean(imp_err))
    print(
        "Mean of function evaluations: ",
        np.mean(imp_eva),
        " with standard error of the mean: ",
        np.std(imp_eva) /
        np.sqrt(n))  # Mean of evaluations with standard error of the mean.
    print("Average time usage: ", (end - start) / n, " seconds.")
elif choice == "5":
    print("Adaptive Monte Carlo Integration")
    n = int(input("How many times of the method do you want to repeat? "))
    # setup array for recording results and evaluations used
    adaptive_result = np.zeros(n)
    adaptive_err = np.zeros(n)
    adaptive_eva = np.zeros(n)
    start = time.time()
    for i in range(n):  # repeat the integrator for n times
        adaptive_result[i], adaptive_err[i], adaptive_eva[i] = mc.adaptive_monte_carlo(
            Psi2, *arg)
        print(
            "Run {0}".format(
                i + 1),
            ": Result: ",
            adaptive_result[i],
            ", standard error: ",
            adaptive_err[i],
            "function evaluation:",
            adaptive_eva[i])
    end = time.time()
    mean = np.mean(adaptive_result)  # take the mean of n integration values
    # find the standard error of the mean
    error = np.std(adaptive_result) / np.sqrt(n)
    print("Mean of results:", mean, ", with error of the mean: ", error,
          " and averaged standard error: ", np.mean(adaptive_err))
    print(
        "Mean of function evaluations: ",
        np.mean(adaptive_eva),
        " with standard error of the mean: ",
        np.std(adaptive_eva) /
        np.sqrt(n))
    print("Average time usage: ", (end - start) / n, " seconds.")
else:
    print("Invalid choice!")  # exit the program if invalid choice given.
