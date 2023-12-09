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
from src.distribution_utils.distributions import partial_pdf, normalization_constant


def get_bins(X: np.ndarray, n_bins: int):
    """! Obtains the density histogram properties for a sample with a given number of bins.

    @param X        The sample array.
    @param n_bins   The number of bins in the histogram.

    @return         The bin values, error in values, bin centers, and bin widths."""
    y, bins = np.histogram(X, bins=n_bins, density=True)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    widths = np.diff(bins)

    # Compute the corresponding error(on a normalized scale)
    y_err = np.sqrt(y / (len(X) * widths))

    return y, y_err, bin_centers, widths


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

    total_pdf = partial_pdf(f=f, lam=lam, mu=mu, sigma=sigma, alpha=alpha, beta=beta)
    norm_constants = normalization_constant(f, lam, mu, sigma, alpha, beta)

    plt.plot(
        X,
        f * stats.norm.pdf(X, loc=mu, scale=sigma) / norm_constants[0],
        label="Normal component",
        linestyle="--",
    )
    plt.plot(
        X,
        (1 - f) * stats.expon.pdf(X, scale=1 / lam) / norm_constants[1],
        label="Exponential component",
        linestyle="--",
    )
    plt.plot(X, total_pdf(X), label="Total probability distribution", alpha=0.5)
    plt.ylabel("Probability Density")
    plt.xlabel("M")
    plt.title("True distribution and its components.")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mle(
    sample: np.ndarray,
    f: float,
    lam: float,
    mu: float,
    sigma: float,
    alpha: float,
    beta: float,
):
    r"""! Plots the density of a sample versus the estimated distribution.

    @param sample   The sample used for the estimation.
    @param f        The estimated fraction of the distribution attributed to the normal component.
                    Must be a real number between 0 and 1.
    @param lam      The estimated \f$\lambda\f$ parameter of the exponential distribution.
                    Must be a positive number.
    @param mu       The estimated mean of the normal distribution.
    @param sigma    The estimated standard deviation of the normal distribution. Must be positive.
    @param alpha    The lower bound of the distribution. Must be non-negative.
    @param beta     The upper bound of the distribution. Must be larger than alpha.
    """

    sns.set()
    fig = plt.figure(figsize=(9, 10))
    gs = fig.add_gridspec(3, hspace=0, height_ratios=[3, 1, 1])
    axs = gs.subplots(sharex=True)
    fig.suptitle("Distributions of the sample and estimated fit.")
    X = np.linspace(alpha, beta, 1000)

    total_pdf = partial_pdf(f=f, lam=lam, mu=mu, sigma=sigma, alpha=alpha, beta=beta)
    norm_constants = normalization_constant(f, lam, mu, sigma, alpha, beta)

    axs[0].plot(
        X, total_pdf(X), label="Estimated probability distribution", color="black"
    )
    axs[0].plot(
        X,
        f * stats.norm.pdf(X, loc=mu, scale=sigma) / norm_constants[0],
        label="Estimated normal component",
        linestyle="--",
        color="tab:orange",
    )
    axs[0].plot(
        X,
        (1 - f) * stats.expon.pdf(X, scale=1 / lam) / norm_constants[1],
        label="Estimated exponential component",
        linestyle="--",
        color="tab:green",
    )
    y, y_err, bin_centers, widths = get_bins(sample, n_bins=100)
    axs[0].bar(
        bin_centers,
        y,
        width=widths,
        color="tab:blue",
        yerr=y_err,
        label="Sample distribution",
        alpha=0.3,
    )
    axs[0].set_ylabel("Probability Density")

    residuals = y - total_pdf(bin_centers)
    axs[1].errorbar(
        bin_centers, residuals, yerr=y_err, ls="none", marker=None, ecolor="black"
    )
    axs[1].scatter(bin_centers, residuals, s=10, color="black")
    axs[1].axhline(0, linestyle="--")
    axs[1].set_ylabel("Residual")

    pull = residuals / y_err
    axs[2].errorbar(bin_centers, pull, yerr=1, ls="none", marker=None, ecolor="black")
    axs[2].scatter(bin_centers, pull, s=10, color="black")
    axs[2].axhline(0, linestyle="--")
    axs[2].set_ylabel("Pull")
    plt.xlabel("M")
    axs[0].legend()
    plt.show()
