r"""! @file estimation.py
@brief Module for parameter estimation.

@details Contains methods for parameter estimation. The most
common method used is iminuit's implemention of Maximum Likelihood Estimation.
Different methods for Unbinned and Binned estimation are included.

@author Created by I. Petrov on 29/11/2023
"""
from iminuit.cost import UnbinnedNLL
from iminuit import Minuit
from typing import Callable, Iterable, Tuple
import numpy as np


def unbinned_mle_estimation(
    X: np.ndarray, f: Callable[[Iterable], float], **kwargs
) -> Tuple:
    """! Unbinned Maximum Likelihood esimation for a sample. Reports the estimated values,
    as well as the errors

    @param X    Array of sample values.
    @param f    Probability density function of the distribution.

    @return     The estimated parameters, Hessian matrix and errors."""
    cost = UnbinnedNLL(data=X, pdf=f)
    model = Minuit(cost, **kwargs)

    if "fixed" in kwargs:
        for param in kwargs["fixed"]:
            model.fixed[param] = True

    if "limits" in kwargs:
        for param in kwargs["limits"]:
            model.limits[param] = kwargs["limits"][param]

    model.migrad()

    print("Fit status: ")
    print(model)

    return model.values, model.hesse, model.errors
