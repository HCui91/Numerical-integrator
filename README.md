# Numerical integrator
 Calculate integral with Newton-Coates/Monte Carlo methods.
## Introduction
Codes are written and tested under Python 3.7.1 64-bit with conda environment. Key topic on numerical integration of a 1D quantum-mechanical system. However, you can rewrite `main.py` to evaluate your own integral.

### The wave-function

The 1D QM wave-function is described as

![\Psi(x)=\frac{1}{\pi^{1/4}}e^{i a(x)}e^{-z^2/2}](https://render.githubusercontent.com/render/math?math=%5CPsi(x)%3D%5Cfrac%7B1%7D%7B%5Cpi%5E%7B1%2F4%7D%7De%5E%7Bi%20a(x)%7De%5E%7B-z%5E2%2F2%7D)

Observable is the probability density of this wave-function, written as:

![\Psi^{*}\Psi = \frac{1}{\pi^{1/2}}e^{-z^2}](https://render.githubusercontent.com/render/math?math=%5CPsi%5E%7B*%7D%5CPsi%20%3D%20%5Cfrac%7B1%7D%7B%5Cpi%5E%7B1%2F2%7D%7De%5E%7B-z%5E2%7D)

The probability of observing this wave-function in a specific region [a,b] can be written as:

![Prob = \int_b^a \Psi^{*}\Psi dx](https://render.githubusercontent.com/render/math?math=Prob%20%3D%20%5Cint_b%5Ea%20%5CPsi%5E%7B*%7D%5CPsi%20dx)

## Dependents
`numba` package is essential to run the project. It is a package which speeds up the calculation of `numpy` operations and multithreading loops. This code has a lot of repeated loops and calculations, so `numba` could translate it into machine code and even parallel computing them. Please use `pip install numba` or `conda install numba`, depends on your environment, to install it. `numba` version 0.41.0 used.

In the case you do not want to use `numba`, you need to remove codes `from numba import jit,njit,prange`,  `@jit(nopython=True)` and `@njit(parallel=True)` in file `main.py` and `monte_carlo.py`. Also, replace `prange()` with `range()` since automatic parallelization is used to speed up the loop.

Other essential packages:
- `numpy`: basic mathematical package
- `matplotlib`: for diagrams
- `random`: random number generator for Monte Carlo process

## How to use

`python main.py` and you will see instructions on your terminal.

## File structure
- The `main.py` contains the wavefunction and entry of arguments of integrator. 
- `newton_coates.py` and `monte_carlo.py` contains codes relating numerical integration methods used. No need to run them since they are imported in the `main.py`. 