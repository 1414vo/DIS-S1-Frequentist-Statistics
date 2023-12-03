r"""! @file estimation.py
@brief Module for parameter estimation.

@details Contains methods for parameter estimation. The most
common method used is iminuit's implemention of Maximum Likelihood Estimation.
Different methods for Unbinned and Binned estimation are included.

@author Created by I. Petrov on 29/11/2023
"""
from iminuit.cost import UnbinnedNLL, BinnedNLL
from iminuit import Minuit
from typing import Callable, Iterable, Tuple
import numpy as np


def unbinned_mle_estimation(
    X: np.ndarray, f: Callable[[Iterable], float], print_results=False, **kwargs
) -> Tuple:
    """! Unbinned Maximum Likelihood esimation for a sample. Reports the estimated values,
    as well as the errors

    @param X    Array of sample values.
    @param f    Probability density function of the distribution.

    @return     The estimated parameters, Hessian matrix and errors."""
    cost = UnbinnedNLL(data=X, pdf=f)
    model = Minuit(cost, **(kwargs["params"]))

    if "fixed" in kwargs:
        for param in kwargs["fixed"]:
            model.fixed[param] = True

    if "limits" in kwargs:
        for param in kwargs["limits"]:
            model.limits[param] = kwargs["limits"][param]

    model.migrad()

    if print_results:
        print("Fit status: ")
        print(model)

    return model.values, model.hesse, model.errors


def binned_mle_estimation(
    X: np.ndarray,
    cdf: Callable[[Iterable], float],
    n_bins=None,
    print_results=False,
    **kwargs
) -> Tuple:
    """! Binned Maximum Likelihood esimation for a sample. Reports the estimated values,
    as well as the errors.

    @param X                Array of sample values.
    @param cdf              Cumulative density function of the distribution.
    @param n_bins           Number of bins. Defaults to the square root of the number of data points.
    @param print_results    Whether to print the estimation results in the console.

    @return     The estimated parameters, Hessian matrix and errors."""

    if not n_bins:
        n_bins = int(len(X) ** 0.5)

    nh, xe = np.histogram(X, bins=n_bins)
    cost = BinnedNLL(nh, xe, cdf)
    model = Minuit(cost, **(kwargs["params"]))

    if "fixed" in kwargs:
        for param in kwargs["fixed"]:
            model.fixed[param] = True

    if "limits" in kwargs:
        for param in kwargs["limits"]:
            model.limits[param] = kwargs["limits"][param]

    model.migrad()

    if print_results:
        print("Fit status: ")
        print(model)

    return model.values, model.hesse, model.errors
