from src.distribution_utils.distributions import partial_pdf
from src.distribution_utils.estimation import (
    unbinned_mle_estimation,
    binned_mle_estimation,
)
from scipy.stats import chi2, poisson
from functools import partial
from typing import Callable, Tuple
from numpy.typing import ArrayLike
import numpy as np


def log_likelihood_ratio(
    sample: ArrayLike,
    h0: Callable[[ArrayLike], ArrayLike],
    h1: Callable[[ArrayLike], ArrayLike],
) -> float:
    """! Computes the log-likelihood ration for a sample under fits for the null
    and altrenate hypotheses. It is used only after reaching a stable fit,
    and is recommended in contrast to using Iminuit's "fval", especially in
    the case of a binned fit.

    @param sample   The distribution sample.
    @param h0       The fit of the null hypothesis pdf.
    @param h1       The fit of the alternate hypothesis pdf.

    @return         The negative log likelihood score."""
    h1_ll = np.sum(np.log(h1(sample)))
    h0_ll = np.sum(np.log(h0(sample)))
    # Cap the results at 0, as the fit is incorrect otherwise.
    return -2 * (h0_ll - h1_ll) if h0_ll < h1_ll else 0


def hypothesis_test(
    sample: ArrayLike,
    h0,
    h1,
    statistic: Callable = log_likelihood_ratio,
    binned: bool = True,
    **kwargs
) -> float:
    """! Performs a single hypothesis test on a sample under two given hypotheses.

    @param sample       The dataset sample.
    @param h0           The underlying null hypothesis. Is in the form of a dictionary containing keys
    "pdf", "cdf" and "rvs". They are respectively used to calculate the PDF, CDF of the distribution and to generate
    a set of random values.
    @param h1           The underlying alternate hypothesis. Is in the form of a dictionary containing keys
    "pdf", "cdf" and "rvs". They are respectively used to calculate the PDF, CDF of the distribution and to generate
    a set of random values. Has to be a superset of h0.
    @param statistic    The test statistic to be used for the test.
    @param binned       Whether to use a binned or unbinned MLE estimation.

    @return             The resulting statistic for the given sample.
    """
    if "null_fit" in kwargs:
        null_fit = kwargs["null_fit"]
    else:
        null_fit = {}

    if "null_limits" in kwargs:
        null_limits = kwargs["null_limits"]
    else:
        null_limits = {}

    if "alt_fit" in kwargs:
        alt_fit = kwargs["alt_fit"]
    else:
        alt_fit = {}

    if "alt_limits" in kwargs:
        alt_limits = kwargs["alt_limits"]
    else:
        alt_limits = {}

    # Set limits on the boundaries.
    null_limits["alpha"] = (None, min(sample) - 1e-5)
    null_limits["beta"] = (max(sample) + 1e-5, None)
    alt_limits["alpha"] = (None, min(sample) - 1e-5)
    alt_limits["beta"] = (max(sample) + 1e-5, None)

    # Execute the correct estimation method.
    if binned:
        null_fit_params, _, _ = binned_mle_estimation(
            sample, h0["cdf"], params=null_fit, limits=null_limits
        )

        alternate_fit_params, _, _ = binned_mle_estimation(
            sample, h1["cdf"], params=alt_fit, limits=alt_limits
        )
    else:
        null_fit_params, _, _ = unbinned_mle_estimation(
            sample, h0["pdf"], params=null_fit, limits=null_limits
        )

        alternate_fit_params, _, _ = unbinned_mle_estimation(
            sample, h1["pdf"], params=alt_fit, limits=alt_limits
        )

    # Create functions corresponding to the MLE fits.
    h0_fit = partial_pdf(pdf=h0["pdf"], **null_fit_params.to_dict())
    h1_fit = partial_pdf(pdf=h1["pdf"], **alternate_fit_params.to_dict())

    return statistic(sample, h0_fit, h1_fit)


def test_statistic_distribution(
    h0,
    h1,
    statistic: Callable,
    n_iter: int = 10000,
    sample_size: int = 1000,
    binned: bool = True,
    store: dict = None,
    **kwargs
) -> Tuple[ArrayLike, ArrayLike]:
    """! Performs a set of hypothesis tests to obtain the test statistic distribution.

    @param h0           The underlying null hypothesis. Is in the form of a dictionary containing keys
    "pdf", "cdf" and "rvs". They are respectively used to calculate the PDF, CDF of the distribution and to generate
    a set of random values.
    @param h1           The underlying alternate hypothesis. Is in the form of a dictionary containing keys
    "pdf", "cdf" and "rvs". They are respectively used to calculate the PDF, CDF of the distribution and to generate
    a set of random values. Has to be a superset of h0.
    @param statistic    The test statistic to be used for the test.
    @param n_iter       The number of iterations to be run. A higher number results in more precise
    test statistic distributions.
    @param sample_size  The size of the samples for each test.
    @param binned       Whether to use a binned or unbinned MLE estimation.
    @param store        Whether to store the test statistic distribution. Is None if not,
    else a dictionary structure is required.

    @return             A tuple of sampled test statistics under the null and alternate hypotheses respectively.
    """
    null_stat = []
    alt_stat = []
    for _ in range(n_iter):
        # Generate a random Null hypothesis sample(deterministic if np.random.seed has been set)
        state = np.random.randint(1, 1000000)
        null_sample = h0["rvs"](
            **kwargs["null_fit"], size=sample_size, random_state=state
        )
        # Generate a random alternate hypothesis sample(deterministic if np.random.seed has been set)
        state = np.random.randint(1, 1000000)
        alt_sample = h1["rvs"](
            **kwargs["alt_fit"], size=sample_size, random_state=state
        )

        # Perform the hypothesis tests for the given samples
        null_stat.append(
            hypothesis_test(null_sample, h0, h1, statistic, binned=binned, **kwargs)
        )
        alt_stat.append(
            hypothesis_test(alt_sample, h0, h1, statistic, binned=binned, **kwargs)
        )
    # Store distributions if configured to do so.
    if store is not None:
        store[sample_size] = (np.array(null_stat), np.array(alt_stat))

    return np.array(null_stat), np.array(alt_stat)


def get_t2_error(
    h0_distribution: np.ndarray,
    h1_distribution: np.ndarray,
    t1_error_rate: float,
    df_init: float = 2.5,
):
    """! Computes the Type 2 error given the distribution of the statistic under the null and alternate hypotheses.

    @param h0_distribution  The test statistic samples when generating from the null hypothesis.
    @param h1_distribution  The test statistic samples when generating from the alternate hypothesis.
    @param t1_error_rate    The Type 1 error rate.
    @param df_init          The initial guess for the Chi-squared degrees of freedom.

    @return                 The estimated type 2 error rate."""

    def chi2_pdf(X, df):
        return chi2.pdf(X, df=df)

    # Clear out outliers and likely invalid fits.
    is_not_outlier = h0_distribution < np.quantile(h0_distribution, 0.99)
    is_fit_invalid = h0_distribution < 1e-8
    h0_sample = h0_distribution[
        np.isfinite(h0_distribution) & is_not_outlier & ~is_fit_invalid
    ]

    # Fit a Chi-squared distribution on the H0 statistic distribution
    chi2_fit, _, _ = unbinned_mle_estimation(
        h0_sample, chi2_pdf, params={"df": df_init}, limits={"df": (0, None)}
    )
    t_threshold = chi2.ppf(1 - t1_error_rate, df=chi2_fit["df"])

    return np.count_nonzero(h1_distribution < t_threshold) / len(h1_distribution)


def binary_threshold_search(
    h0,
    h1,
    t1_error_rate=2.9e-7,
    t2_error_rate=0.1,
    n_iter=10000,
    n_0=500,
    store_distributions=False,
    binned=True,
    df_diff=3,
    fit_configuration={},
):
    r"""! Searches for the number of data points such that \$H_0\$ is falsely accepted with a Type 1 Error
    of t1_error_rate, and \$H_1\$ is falsely rejected with a Type 2 Error of at most t2_error_rate.
    The search is executed using a binary search algorithm, and yields a single value as a result.

    @param h0                       The underlying null hypothesis. Is in the form of a dictionary containing keys
    "pdf", "cdf" and "rvs". They are respectively used to calculate the PDF, CDF of the distribution and to generate
    a set of random values.
    @param h1                       The underlying alternate hypothesis. Is in the form of a dictionary containing keys
    "pdf", "cdf" and "rvs". They are respectively used to calculate the PDF, CDF of the distribution and to generate
    a set of random values. Has to be a superset of h0.
    @param t1_error_rate            The type 1 error rate for rejecting the null hypothesis.
    @param t2_error_rate            The type 2 error rate for accepting the alternate hypothesis.
    @param n_iter                   The number of bootstrap iterations done per hypothesis test.
    @param n_0                      The initial guess for the size threshold.
    @param store_distributions      Whether to store the T-statistic distributions from the hypothesis tests.
    @param binned                   Whether to use a binned fit.
    @param df_diff                  The difference in degrees of freedom between the 2 hypotheses.
    @param fit_configuration        A dictionary containing the configuration for fitting the models.
    Should contain the keys "null_fit", "null_limits", "alt_fit", "alt_limit", "alt_init".

    @return                         A tuple of the sampled dataset sizes, the determined range,
    and the T-statistic storage if one was required.
    """
    results = []
    val_range = (1, n_0)
    if store_distributions:
        store = {}
    else:
        store = None
    # Define a simplified function for obtaining the test statistic distribution
    get_statistic = partial(
        test_statistic_distribution,
        h0=h0,
        h1=h1,
        statistic=log_likelihood_ratio,
        null_fit=fit_configuration["null_fit"],
        null_limits=fit_configuration["null_limits"],
        alt_fit=fit_configuration["alt_fit"],
        alt_limits=fit_configuration["alt_limits"],
        alt_init=fit_configuration["alt_init"],
        n_iter=n_iter,
        binned=binned,
        store=store,
    )
    # Define a simplified function for obtaining the Type 2 Error from the test statistic distribution
    t2_error = partial(get_t2_error, t1_error_rate=t1_error_rate, df_init=df_diff)
    high_score = t2_error(*get_statistic(sample_size=n_0))
    results.append((val_range[1], high_score))
    # Until we have obtained a range that covers both sides of the threshold, double the upper limit.
    while high_score > t2_error_rate:
        high_score = t2_error(*get_statistic(sample_size=2 * val_range[1]))
        val_range = (val_range[1], val_range[1] * 2)
        results.append((val_range[1], high_score))

    # Narrow down the range to 1 number
    while val_range[1] - val_range[0] != 1:
        # Calculate the midpoint score
        sample_value = (val_range[0] + val_range[1]) // 2
        new_score = t2_error(*get_statistic(sample_size=sample_value))
        results.append((sample_value, new_score))
        # If its lower than threshold, use new value as upper limit
        if new_score < t2_error_rate:
            val_range = (val_range[0], sample_value)
            high_score = new_score
        # If its higher than threshold, use new value as lower limit
        else:
            val_range = (sample_value, val_range[1])

    return results, val_range[1], store


def weighted_threshold_search(
    h0,
    h1,
    t1_error_rate=2.9e-7,
    t2_error_rate=0.1,
    n_iter=10000,
    n_0=500,
    store_distributions=False,
    binned=True,
    df_diff=3,
    fit_configuration={},
):
    r"""! Searches for the number of data points such that \$H_0\$ is falsely accepted with a Type 1 Error
    of t1_error_rate,and \$H_1\$ is falsely rejected with a Type 2 Error of at most t2_error_rate.
    The search is executed using a weighted search algorithm, further described in the report,
    and yields a range of probable values.

    @param h0                       The underlying null hypothesis. Is in the form of a dictionary containing keys
    "pdf", "cdf" and "rvs". They are respectively used to calculate the PDF, CDF of the distribution and to generate
    a set of random values.
    @param h1                       The underlying alternate hypothesis. Is in the form of a dictionary containing keys
    "pdf", "cdf" and "rvs". They are respectively used to calculate the PDF, CDF of the distribution and to generate
    a set of random values. Has to be a superset of h0.
    @param t1_error_rate            The type 1 error rate for rejecting the null hypothesis.
    @param t2_error_rate            The type 2 error rate for accepting the alternate hypothesis.
    @param n_iter                   The number of bootstrap iterations done per hypothesis test.
    @param n_0                      The initial guess for the size threshold.
    @param store_distributions      Whether to store the T-statistic distributions from the hypothesis tests.
    @param binned                   Whether to use a binned fit.
    @param df_diff                  The difference in degrees of freedom between the 2 hypotheses.
    @param fit_configuration        A dictionary containing the configuration for fitting the models.
    Should contain the keys "null_fit", "null_limits", "alt_fit", "alt_limit", "alt_init".

    @return                         A tuple of the sampled dataset sizes, the determined range,
    and the T-statistic storage if one was required.
    """
    results = []
    val_range = [1, n_0]
    low_score = 0
    if store_distributions:
        store = {}
    else:
        store = None
    # Define a simplified function for obtaining the test statistic distribution
    get_statistic = partial(
        test_statistic_distribution,
        h0=h0,
        h1=h1,
        statistic=log_likelihood_ratio,
        null_fit=fit_configuration["null_fit"],
        null_limits=fit_configuration["null_limits"],
        alt_fit=fit_configuration["alt_fit"],
        alt_limits=fit_configuration["alt_limits"],
        alt_init=fit_configuration["alt_init"],
        n_iter=n_iter,
        binned=binned,
        store=store,
    )
    # Define a simplified function for obtaining the Type 2 Error from the test statistic distribution
    t2_error = partial(get_t2_error, t1_error_rate=t1_error_rate, df_init=df_diff)
    # Calculate the score for the initial guess
    high_score = t2_error(*get_statistic(sample_size=n_0))
    results.append(
        (
            val_range[1],
            high_score,
            poisson.cdf(high_score * n_iter, mu=t2_error_rate * n_iter),
        )
    )
    # Until we have obtained a range that covers both sides of the threshold, double the upper limit.
    while high_score > t2_error_rate:
        low_score, high_score = high_score, t2_error(
            *get_statistic(sample_size=2 * val_range[1])
        )
        val_range = [val_range[1], val_range[1] * 2]
        results.append(
            (
                val_range[1],
                high_score,
                poisson.cdf(high_score * n_iter, mu=t2_error_rate * n_iter),
            )
        )
    lower_hit = False
    upper_hit = False
    low_p = poisson.cdf(low_score * n_iter, mu=t2_error_rate * n_iter)
    high_p = poisson.cdf(high_score * n_iter, mu=t2_error_rate * n_iter)

    # Iterate until we have obtained limits that cover the threshold on both sides.
    while not (lower_hit and upper_hit):
        # Calculate the score for the midpoint
        sample_value = (val_range[0] + val_range[1]) // 2
        mid_score = t2_error(*get_statistic(sample_size=sample_value))
        mid_p = poisson.cdf(mid_score * n_iter, mu=t2_error_rate * n_iter)

        # Recompute the range using a p-value weighted shift.
        val_range[0], val_range[1] = sample_value - int(
            (val_range[1] - val_range[0]) / 2 * (low_p - mid_p) / (low_p - high_p)
        ), sample_value + int(
            (val_range[1] - val_range[0]) / 2 * (mid_p - high_p) / (low_p - high_p)
        )

        # Calculate the score for the ends of the new range
        low_score = t2_error(*get_statistic(sample_size=val_range[0]))
        high_score = t2_error(*get_statistic(sample_size=val_range[1]))
        low_p = poisson.cdf(low_score * n_iter, mu=t2_error_rate * n_iter)
        high_p = poisson.cdf(high_score * n_iter, mu=t2_error_rate * n_iter)

        # Check if the lower limit's range covers the threshold
        if (
            low_score * n_iter - (t2_error_rate * n_iter) ** 0.5
            <= t2_error_rate * n_iter
        ):
            lower_hit = True
            print("Lower Hit:", val_range[0])

        # Check if the upper limit's range covers the threshold
        if (
            t2_error_rate * n_iter
            <= high_score * n_iter + (t2_error_rate * n_iter) ** 0.5
        ):
            upper_hit = True
            print("Upper Hit:", val_range[1])

        # Store the sampling results
        results.append(
            (
                val_range[0],
                low_score,
                poisson.cdf(low_score * n_iter, mu=t2_error_rate * n_iter),
            )
        )
        results.append(
            (
                sample_value,
                mid_score,
                poisson.cdf(mid_score * n_iter, mu=t2_error_rate * n_iter),
            )
        )
        results.append(
            (
                val_range[1],
                high_score,
                poisson.cdf(high_score * n_iter, mu=t2_error_rate * n_iter),
            )
        )

    # Obtain the values such that
    valid_guesses = []
    for item in results:
        error = np.sqrt(item[1] / n_iter)
        lower_bound = item[1] - error
        upper_bound = item[1] + error
        if lower_bound < t2_error_rate < upper_bound:
            valid_guesses.append(item[0])

    return results, (min(valid_guesses), max(valid_guesses)), store
