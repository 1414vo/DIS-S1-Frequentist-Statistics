r"""! @file optimized_distributions.py
@brief Contains methods to compute properties of the used distribution using optimized routines.

@details Contains methods to compute and verify properties of the distribution
    detailed in the assignment. This includes checking whether the input parameters
    are valid, computing the normalization constant, or obtaining the pdf.
    The distribution pdf is given by:
    \f$p^\prime(M; f,\lambda,\mu,\sigma,\alpha,\beta) = N(\alpha, \beta) \times
    (fs(M;\mu, \sigma) + (1-f)b(M; \lambda))\f$,
    where \f$N(\alpha, \beta)\f$ is a normalization constant. Routines utilize
    higher performance methods using the numba-stats package.

@author Created by I. Petrov on 02/12/2023
"""

from numba_stats import truncnorm, truncexpon, uniform
import numpy as np


def background_pdf_optimized(M, lam: float, alpha: float, beta: float):
    r"""! Computes the probability density function for the distribution
    \f$p(M; \lambda) = N \lambda e^{-\lambda M}\f$ in the range \f$[\alpha, \beta]\f$.

    @param M                A value or iterable of observations.
    @param lam              The \f$\lambda\f$ parameter of the exponential distribution.
                            Must be a positive number.
    @param alpha            The lower bound of the distribution. Must be non-negative.
    @param beta             The upper bound of the distribution. Must be larger than alpha.\

    @return                 The probability density functions of the observations.
    """

    return truncexpon.pdf(M, loc=alpha, scale=1 / lam, xmin=alpha, xmax=beta)


def distribution_pdf_optimized(
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

    return f * truncnorm.pdf(M, loc=mu, scale=sigma, xmin=alpha, xmax=beta) + (
        1 - f
    ) * truncexpon.pdf(M, loc=alpha, scale=1 / lam, xmin=alpha, xmax=beta)


def background_cdf_optimized(M, lam: float, alpha: float, beta: float):
    r"""! Computes the cumulative density function for the distribution
    \f$p(M; \lambda) = N \lambda e^{-\lambda M}\f$ in the range \f$[\alpha, \beta]\f$.

    @param M                A value or iterable of observations.
    @param lam              The \f$\lambda\f$ parameter of the exponential distribution.
                            Must be a positive number.
    @param alpha            The lower bound of the distribution. Must be non-negative.
    @param beta             The upper bound of the distribution. Must be larger than alpha.\

    @return                 The cumulative density functions of the observations.
    """

    return truncexpon.cdf(M, loc=alpha, scale=1 / lam, xmin=alpha, xmax=beta)


def distribution_cdf_optimized(
    M, f: float, lam: float, mu: float, sigma: float, alpha: float, beta: float
):
    r"""! Computes the cumulative density function for the distribution
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

    @return         The value of the cdf at a given point
                    (or multiple points if input is iterable).
    """
    return f * truncnorm.cdf(M, loc=mu, scale=sigma, xmin=alpha, xmax=beta) + (
        1 - f
    ) * truncexpon.cdf(M, loc=alpha, scale=1 / lam, xmin=alpha, xmax=beta)


def generate_background_samples(
    lam: float, alpha: float, beta: float, size: int, random_state: int = None
):
    r"""! Generates a sample for the distribution
    \f$p(M; \lambda) = N \lambda e^{-\lambda M}\f$ in the range \f$[\alpha, \beta]\f$.

    @param lam      The \f$\lambda\f$ parameter of the exponential distribution.
                    Must be a positive number.
    @param alpha    The lower bound of the distribution. Must be non-negative.
    @param beta     The upper bound of the distribution. Must be larger than alpha.
    @param size     The length of the sample.

    @return         A sample of size given by the size variable.
    """
    return truncexpon.rvs(
        xmin=alpha,
        xmax=beta,
        loc=alpha,
        scale=1 / lam,
        size=size,
        random_state=random_state,
    )


def generate_distribution_samples(
    f: float,
    lam: float,
    mu: float,
    sigma: float,
    alpha: float,
    beta: float,
    size: int,
    random_state: int = None,
):
    r"""! Generates a sample for the distribution
    \f$p(M; f,\lambda,\mu,\sigma) = fs(M;\mu, \sigma) + (1-f)b(M; \lambda)\f$.

    @param f            The fraction of the distribution attributed to the normal component.
                        Must be a real number between 0 and 1.
    @param lam          The \f$\lambda\f$ parameter of the exponential distribution.
                        Must be a positive number.
    @param mu           The mean of the normal distribution.
    @param sigma        The standard deviation of the normal distribution. Must be positive.
    @param alpha        The lower bound of the distribution. Must be non-negative.
    @param beta         The upper bound of the distribution. Must be larger than alpha.
    @param size         The length of the sample.
    @param random_state The random state seed.

    @return             A sample of size given by the size variable.
    """
    states = (
        range(random_state, random_state + 3)
        if random_state is not None
        else [None, None, None]
    )
    classes = uniform.rvs(a=0, w=1, size=size, random_state=states[0])
    exp = truncexpon.rvs(
        xmin=alpha,
        xmax=beta,
        loc=alpha,
        scale=1 / lam,
        size=size,
        random_state=states[1],
    )
    norm = truncnorm.rvs(
        loc=mu, scale=sigma, xmin=alpha, xmax=beta, size=size, random_state=states[2]
    )
    return np.where(classes < f, norm, exp)


def two_signal_distribution_pdf(M, f1, f2, lam, mu1, mu2, sigma, alpha, beta):
    r"""! Generates a sample for the distribution
    \f$p(M; f_1, f_2,\lambda,\mu_1,mu_2,\sigma) = f_1s(M;\mu_1, \sigma)
    + f_2s(M;\mu_2, \sigma) + (1-f_1-f_2)b(M; \lambda)\f$.

    @param M            A value or iterable of observations
    @param f1           The fraction of the distribution attributed to the first signal.
                        Must be a real number between 0 and 1.
    @param f2           The fraction of the distribution attributed to the second signal.
                        Must be a real number between 0 and 1.
    @param lam          The \f$\lambda\f$ parameter of the exponential distribution.
                        Must be a positive number.
    @param mu1          The mean of the first signal distribution.
    @param mu1          The mean of the second signal distribution.
    @param sigma        The standard deviation of the signal distributions. Must be positive.
    @param alpha        The lower bound of the distribution. Must be non-negative.
    @param beta         The upper bound of the distribution. Must be larger than alpha.

    @return             The PDF values for the corresponding sample set.
    """
    return (
        f1 * truncnorm.pdf(M, loc=mu1, scale=sigma, xmin=alpha, xmax=beta)
        + f2 * truncnorm.pdf(M, loc=mu2, scale=sigma, xmin=alpha, xmax=beta)
        + (1 - f1 - f2)
        * truncexpon.pdf(M, loc=alpha, scale=1 / lam, xmin=alpha, xmax=beta)
    )


def two_signal_distribution_cdf(M, f1, f2, lam, mu1, mu2, sigma, alpha, beta):
    r"""! Generates a sample for the distribution
    \f$p(M; f_1, f_2,\lambda,\mu_1,mu_2,\sigma) = f_1s(M;\mu_1, \sigma)
    + f_2s(M;\mu_2, \sigma) + (1-f_1-f_2)b(M; \lambda)\f$.

    @param M            A value or iterable of observations
    @param f1           The fraction of the distribution attributed to the first signal.
                        Must be a real number between 0 and 1.
    @param f2           The fraction of the distribution attributed to the second signal.
                        Must be a real number between 0 and 1.
    @param lam          The \f$\lambda\f$ parameter of the exponential distribution.
                        Must be a positive number.
    @param mu1          The mean of the first signal distribution.
    @param mu1          The mean of the second signal distribution.
    @param sigma        The standard deviation of the signal distributions. Must be positive.
    @param alpha        The lower bound of the distribution. Must be non-negative.
    @param beta         The upper bound of the distribution. Must be larger than alpha.

    @return             The CDF values for the given sample set.
    """
    return (
        f1 * truncnorm.cdf(M, loc=mu1, scale=sigma, xmin=alpha, xmax=beta)
        + f2 * truncnorm.cdf(M, loc=mu2, scale=sigma, xmin=alpha, xmax=beta)
        + (1 - f1 - f2)
        * truncexpon.cdf(M, loc=alpha, scale=1 / lam, xmin=alpha, xmax=beta)
    )


def generate_two_signal_distribution_samples(
    f1: float,
    f2: float,
    lam: float,
    mu1: float,
    mu2: float,
    sigma: float,
    alpha: float,
    beta: float,
    size: int,
    random_state: int = None,
):
    r"""! Generates a sample for the distribution
    \f$p(M; f_1, f_2,\lambda,\mu_1,mu_2,\sigma) = f_1s(M;\mu_1, \sigma)
    + f_2s(M;\mu_2, \sigma) + (1-f_1-f_2)b(M; \lambda)\f$.

    @param f1           The fraction of the distribution attributed to the first signal.
                        Must be a real number between 0 and 1.
    @param f2           The fraction of the distribution attributed to the second signal.
                        Must be a real number between 0 and 1.
    @param lam          The \f$\lambda\f$ parameter of the exponential distribution.
                        Must be a positive number.
    @param mu1          The mean of the first signal distribution.
    @param mu1          The mean of the second signal distribution.
    @param sigma        The standard deviation of the signal distributions. Must be positive.
    @param alpha        The lower bound of the distribution. Must be non-negative.
    @param beta         The upper bound of the distribution. Must be larger than alpha.
    @param size         The length of the sample.
    @param random_state The random state seed.

    @return             A sample of size given by the size variable.
    """
    states = (
        range(random_state, random_state + 4)
        if random_state is not None
        else [None, None, None, None]
    )
    classes = uniform.rvs(a=0, w=1, size=size, random_state=states[0])
    exp = truncexpon.rvs(
        xmin=alpha,
        xmax=beta,
        loc=alpha,
        scale=1 / lam,
        size=size,
        random_state=states[1],
    )
    norm1 = truncnorm.rvs(
        loc=mu1,
        scale=sigma,
        xmin=alpha,
        xmax=beta,
        size=size,
        random_state=states[2],
    )
    norm2 = truncnorm.rvs(
        loc=mu2,
        scale=sigma,
        xmin=alpha,
        xmax=beta,
        size=size,
        random_state=states[3],
    )
    return np.where(classes < f1, norm1, np.where(classes < f1 + f2, norm2, exp))
