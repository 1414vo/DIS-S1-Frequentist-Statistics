r"""! @file distribution.py
@brief Contains methods to compute properties of the used distribution.

@details Contains methods to compute and verify properties of the distribution
    detailed in the assignment. This includes checking whether the input parameters
    are valid, computing the normalization constant, or obtaining the pdf.
    The distribution pdf is given by:
    \(p^\prime(M; f,\lambda,\mu,\sigma,\alpha,\beta) = N(\alpha, \beta) \times
    (fs(M;\mu, \sigma) + (1-f)b(M; \lambda))\),
    where $N(\alpha, \beta) is a normalization constant.$

@author Created by I. Petrov on 18/11/2023
"""

import scipy.stats as stats
from scipy.special import erf
from functools import cache
import math
import warnings


def check_parameters(
    f: float, lam: float, mu: float, sigma: float, alpha: float, beta: float
) -> None:
    r"""! Determines whether a given set of parameters is valid.

    @param f        The fraction of the distribution attributed to the normal component.
                    Must be a real number between 0 and 1.
    @param lam      The $\lambda$ parameter of the exponential distribution.
                    Must be a positive number.
    @param mu       The mean of the normal distribution.
    @param sigma    The standard deviation of the normal distribution. Must be positive.
    @param alpha    The lower bound of the distribution. Must be non-negative.
    @param beta     The upper bound of the distribution. Must be larger than alpha.
    """
    if f < 0 or f > 1:
        raise ValueError('Fraction parameter "f" must be between 0 and 1.')

    if lam <= 0 or sigma <= 0:
        raise ValueError(
            'Invalid parameter values - decay parameter "lam" and\
            normal distribution standard deviation "sigma" must be positive.'
        )

    if beta < alpha:
        raise ValueError("Upper bound must be smaller than lower bound.")

    if alpha < 0:
        warnings.warn(
            "Exponential distribution has no physical meaning below 0,\
                      please ensure bounds cover only non-negative values."
        )


@cache
def normalization_constant(
    f: float, lam: float, mu: float, sigma: float, alpha: float, beta: float
) -> float:
    r"""! Computes the normalization for the distribution
    $p(M; f,\lambda,\mu,\sigma) = fs(M;\mu, \sigma) + (1-f)b(M; \lambda)$.

    @param f        The fraction of the distribution attributed to the normal component.
                    Must be a real number between 0 and 1.
    @param lam      The $\lambda$ parameter of the exponential distribution.
                    Must be a positive number.
    @param mu       The mean of the normal distribution.
    @param sigma    The standard deviation of the normal distribution. Must be positive.
    @param alpha    The lower bound of the distribution. Must be non-negative.
    @param beta     The upper bound of the distribution. Must be larger than alpha.
    """
    check_parameters(f, lam, mu, sigma, alpha, beta)

    root_2 = 2**0.5

    normal_component = (
        f
        / 2
        * (erf((beta - mu) / (root_2 * sigma)) - erf((alpha - mu) / (root_2 * sigma)))
    )
    exponential_component = (1 - f) * (math.exp(-lam * alpha) - math.exp(-lam * alpha))

    return 1 / (normal_component + exponential_component)


def distribution_pdf(
    M, f: float, lam: float, mu: float, sigma: float, alpha: float, beta: float
):
    r"""! Computes the normalization for the distribution
    $p(M; f,\lambda,\mu,\sigma) = fs(M;\mu, \sigma) + (1-f)b(M; \lambda)$.

    @param M        A value or iterable of observations
    @param f        The fraction of the distribution attributed to the normal component.
                    Must be a real number between 0 and 1.
    @param lam      The $\lambda$ parameter of the exponential distribution.
                    Must be a positive number.
    @param mu       The mean of the normal distribution.
    @param sigma    The standard deviation of the normal distribution. Must be positive.
    @param alpha    The lower bound of the distribution. Must be non-negative.
    @param beta     The upper bound of the distribution. Must be larger than alpha.

    @return         The value of the pdf at a given point
                    (or multiple points if input is iterable).
    """
    normal_component = f * stats.norm.pdf(M, loc=mu, scale=sigma)
    exponential_component = (1 - f) * stats.expon.pdf(M, scale=1 / lam)

    return (normal_component + exponential_component) * normalization_constant(
        f, lam, mu, sigma, alpha, beta
    )
