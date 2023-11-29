r"""! @file distribution_plots.py
@brief Contains methods for different visualizations of the distribution.

@details Contains methods for different visualizations of the distribution.
Includes plotting the true distribution or the estimated one from a sample.

@author Created by I. Petrov on 29/11/2023
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from src.distribution_utils.distribution import (
    partial_pdf,
    normalization_constant,
    generate_sample,
)


def plot_distribution_mix(
    f: float, lam: float, mu: float, sigma: float, alpha: float, beta: float
):
    r"""! Plots the overlay of the pdf and the 2 components of the distribution:
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
    """

    sns.set()
    X = np.linspace(alpha, beta, 1000)

    total_pdf = partial_pdf(f, lam, mu, sigma, alpha, beta)
    norm_constants = normalization_constant(f, lam, mu, sigma, alpha, beta)

    plt.plot(
        X,
        f * stats.norm.pdf(X, loc=mu, scale=sigma) / norm_constants[0],
        label="Normal component",
    )
    plt.plot(
        X,
        (1 - f) * stats.expon.pdf(X, scale=1 / lam) / norm_constants[1],
        label="Exponential component",
    )
    plt.plot(X, total_pdf(X), label="Total probability distribution", alpha=0.5)
    plt.legend()
    plt.show()


def plot_samples(
    f: float,
    lam: float,
    mu: float,
    sigma: float,
    alpha: float,
    beta: float,
    N: int = 100000,
):
    sns.set()
    X = np.linspace(alpha, beta, 1000)

    total_pdf = partial_pdf(f, lam, mu, sigma, alpha, beta)

    plt.plot(X, total_pdf(X), label="Total probability distribution", alpha=0.5)

    samples = generate_sample(total_pdf, alpha, beta, N)

    plt.hist(samples, bins=50, label="Sample distribution", density=True)

    plt.legend()
    plt.show()
