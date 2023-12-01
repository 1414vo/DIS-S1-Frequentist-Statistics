r"""! @file distribution.py
@brief Contains methods to compute properties of the used distribution.

@details Contains methods to compute and verify properties of the distribution
    detailed in the assignment. This includes checking whether the input parameters
    are valid, computing the normalization constant, or obtaining the pdf.
    The distribution pdf is given by:
    \f$p^\prime(M; f,\lambda,\mu,\sigma,\alpha,\beta) = N(\alpha, \beta) \times
    (fs(M;\mu, \sigma) + (1-f)b(M; \lambda))\f$,
    where \f$N(\alpha, \beta)\f$ is a normalization constant.

@author Created by I. Petrov on 18/11/2023
"""

import scipy.stats as stats
import scipy.optimize as optimize
from scipy.special import erf
from functools import cache
from typing import Callable
import numpy as np
import math
import warnings


def check_parameters(
    f: float, lam: float, mu: float, sigma: float, alpha: float, beta: float
) -> None:
    r"""! Determines whether a given set of parameters is valid.

    @param f        The fraction of the distribution attributed to the normal component.
                    Must be a real number between 0 and 1.
    @param lam      The \f$\lambda\f$ parameter of the exponential distribution.
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
    f: float,
    lam: float,
    mu: float,
    sigma: float,
    alpha: float,
    beta: float,
    disable_checks: bool = False,
) -> float:
    r"""! Computes the normalization for the distribution
    \f$p(M; f,\lambda,\mu,\sigma) = fs(M;\mu, \sigma) + (1-f)b(M; \lambda)\f$.

    @param f                The fraction of the distribution attributed to the normal component.
                            Must be a real number between 0 and 1.
    @param lam              The \f$\lambda\f$ parameter of the exponential distribution.
                            Must be a positive number.
    @param mu               The mean of the normal distribution.
    @param sigma            The standard deviation of the normal distribution. Must be positive.
    @param alpha            The lower bound of the distribution. Must be non-negative.
    @param beta             The upper bound of the distribution. Must be larger than alpha.
    @param disable_checks   Whether to check for parameter validity. Should be disabled when
                            performing MLE.

    @return                 The normalization constants for both the normal and exponential components.
    """
    if not disable_checks:
        check_parameters(f, lam, mu, sigma, alpha, beta)

    root_2 = 2**0.5

    normal_component = 0.5 * (
        erf((beta - mu) / (root_2 * sigma)) - erf((alpha - mu) / (root_2 * sigma))
    )
    exponential_component = (
        (math.exp(-lam * alpha) - math.exp(-lam * beta))
        if alpha >= 0
        else (1 - math.exp(-lam * beta))
    )

    return normal_component, exponential_component


def distribution_pdf(
    M, f: float, lam: float, mu: float, sigma: float, alpha: float, beta: float
):
    r"""! Computes the normalization for the distribution
    \f$p(M; f,\lambda,\mu,\sigma) = fs(M;\mu, \sigma) + (1-f)b(M; \lambda)\f$.

    @param M        A value or iterable of observations
    @param f        The fraction of the distribution attributed to the normal component.
                    Must be a real number between 0 and 1.
    @param lam      The \f$\lambda\f$ parameter of the exponential distribution.
                    Must be a positive number.
    @param mu       The mean of the normal distribution.
    @param sigma    The standard deviation of the normal distribution. Must be positive.
    @param alpha    The lower bound of the distribution. Must be non-negative.
    @param beta     The upper bound of the distribution. Must be larger than alpha.

    @return         The value of the pdf at a given point
                    (or multiple points if input is iterable).
    """
    normal_constant, exponential_constant = normalization_constant(
        f, lam, mu, sigma, alpha, beta
    )

    normal_component = f * stats.norm.pdf(M, loc=mu, scale=sigma)
    exponential_component = (1 - f) * stats.expon.pdf(M, scale=1 / lam)

    return (
        normal_component / normal_constant
        + exponential_component / exponential_constant
    )


def partial_pdf(pdf=distribution_pdf, *args, **kwargs):
    r"""Generates a wrapper pdf from relevant parameters.

    @return A partially instantiated pdf from a set of parameters."""
    return lambda X: pdf(X, *args, **kwargs)


@cache
def background_only_normalization(lam: float, alpha: float, beta: float):
    r"""! Computes the normalization for the distribution
    \f$p(M; \lambda) = N \lambda e^{-\lambda M}\f$ in the range \f$[\alpha, \beta]\f$.

    @param f                The fraction of the distribution attributed to the normal component.
                            Must be a real number between 0 and 1.
    @param lam              The \f$\lambda\f$ parameter of the exponential distribution.
                            Must be a positive number.
    @param alpha            The lower bound of the distribution. Must be non-negative.
    @param beta             The upper bound of the distribution. Must be larger than alpha.\

    @return                 The normalization constant for the exponential distribution.
    """
    if alpha >= 0:
        return math.exp(-lam * alpha) - math.exp(-lam * beta)
    else:
        return 1 - math.exp(-lam * beta)


def background_only_distribution(M, lam: float, alpha: float, beta: float):
    r"""! Computes the probability density function for the distribution
    \f$p(M; \lambda) = N \lambda e^{-\lambda M}\f$ in the range \f$[\alpha, \beta]\f$.

    @param M                A value or iterable of observations.
    @param f                The fraction of the distribution attributed to the normal component.
                            Must be a real number between 0 and 1.
    @param lam              The \f$\lambda\f$ parameter of the exponential distribution.
                            Must be a positive number.
    @param alpha            The lower bound of the distribution. Must be non-negative.
    @param beta             The upper bound of the distribution. Must be larger than alpha.\

    @return                 The probability density functions of the observations.
    """
    return stats.expon.pdf(M, scale=1 / lam) / background_only_normalization(
        lam, alpha, beta
    )


def generate_sample(
    pdf: Callable[[float], float],
    min_x: float,
    max_x: float,
    N: int = 100000,
    random_seed: int = 0,
) -> np.ndarray:
    r"""! Creates a sample from a distribution using the accept-reject method.

    @param pdf          The probability density function of the distribution.
                        Must be normalized in the interval (min_x, max_x).
    @param min_x        The lower bound of the interval.
    @param max_x        The upper bound of the interval.
    @param N            The number of samples to be generated.
    @param random_seed  Random seed for random number generation.
    @return             The generated sample.
    """
    np.random.seed(seed=random_seed)

    max_value = optimize.brute(
        lambda X: -pdf(X), ranges=(slice(min_x, max_x, (max_x - min_x) / 1000),)
    )

    samples = []
    while len(samples) < N:
        x_samples = np.random.uniform(low=min_x, high=max_x, size=N // 10)
        y_samples = np.random.uniform(low=0, high=max_value, size=N // 10)

        correct_samples = y_samples <= pdf(x_samples)

        x_samples = x_samples[correct_samples]

        if len(samples) + len(x_samples) > N:
            samples.extend(x_samples[: N - len(x_samples)])
        else:
            samples.extend(x_samples)

    return np.array(samples)
