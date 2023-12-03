r"""! @file distributions.py
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

from numba_stats import truncnorm, truncexpon


def background_pdf_optimized(M, lam: float, alpha: float, beta: float):
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

    return truncexpon.pdf(M, loc=0, scale=1 / lam, xmin=alpha, xmax=beta)


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
    ) * truncexpon.pdf(M, loc=0, scale=1 / lam, xmin=alpha, xmax=beta)


def background_cdf_optimized(M, lam: float, alpha: float, beta: float):
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

    return truncexpon.cdf(M, loc=0, scale=1 / lam, xmin=alpha, xmax=beta)


def distribution_cdf_optimized(
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

    return f * truncnorm.cdf(M, loc=mu, scale=sigma, xmin=alpha, xmax=beta) + (
        1 - f
    ) * truncexpon.cdf(M, loc=0, scale=1 / lam, xmin=alpha, xmax=beta)
