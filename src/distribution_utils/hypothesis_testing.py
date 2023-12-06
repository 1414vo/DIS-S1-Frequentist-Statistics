from src.distribution_utils.distributions import partial_pdf
from src.distribution_utils.estimation import (
    unbinned_mle_estimation,
    binned_mle_estimation,
)
from scipy.stats import chi2, poisson
from functools import partial
import numpy as np


def log_likelihood_ratio(sample, h0, h1):
    h1_ll = np.sum(np.log(h1(sample)))
    h0_ll = np.sum(np.log(h0(sample)))
    return -2 * (h0_ll - h1_ll) if h0_ll < h1_ll else 0


def hypothesis_test(
    sample, h0, h1, statistic=log_likelihood_ratio, binned=True, **kwargs
):
    if "null_params" in kwargs:
        null_params = kwargs["null_params"]
    else:
        null_params = {}

    if "null_limits" in kwargs:
        null_limits = kwargs["null_limits"]
    else:
        null_limits = {}

    if "alternate_params" in kwargs:
        alternate_params = kwargs["alternate_params"]
    else:
        alternate_params = {}

    if "alternate_limits" in kwargs:
        alternate_limits = kwargs["alternate_limits"]
    else:
        alternate_limits = {}

    null_limits["alpha"] = (None, min(sample) - 1e-5)
    null_limits["beta"] = (max(sample) + 1e-5, None)
    alternate_limits["alpha"] = (None, min(sample) - 1e-5)
    alternate_limits["beta"] = (max(sample) + 1e-5, None)

    if binned:
        null_fit_params, _, _ = binned_mle_estimation(
            sample, h0["cdf"], params=null_params, limits=null_limits
        )

        alternate_fit_params, _, _ = binned_mle_estimation(
            sample, h1["cdf"], params=alternate_params, limits=alternate_limits
        )
    else:
        null_fit_params, _, _ = unbinned_mle_estimation(
            sample, h0["pdf"], params=null_params, limits=null_limits
        )

        alternate_fit_params, _, _ = binned_mle_estimation(
            sample, h1["pdf"], params=alternate_params, limits=alternate_limits
        )
    h0_fit = partial_pdf(pdf=h0["pdf"], **null_fit_params.to_dict())
    h1_fit = partial_pdf(pdf=h1["pdf"], **alternate_fit_params.to_dict())

    return statistic(sample, h0_fit, h1_fit)


def test_statistic_distribution(
    h0, h1, stat, n_samples=10000, sample_size=1000, binned=True, store=None, **kwargs
):
    null_stat = []
    alt_stat = []
    for _ in range(n_samples):
        null_sample = h0["rvs"](
            **kwargs["null_fit"], size=sample_size, random_state=None
        )
        alt_sample = h1["rvs"](**kwargs["alt_fit"], size=sample_size, random_state=None)

        args = {
            "null_params": kwargs["null_fit"],
            "null_limits": kwargs["null_limits"],
            "alternate_params": kwargs["alt_fit"],
            "alternate_limits": kwargs["alt_limits"],
        }
        null_args = args.copy()
        null_args["alternate_params"] = kwargs["alt_init"]
        null_stat.append(
            hypothesis_test(null_sample, h0, h1, stat, binned=binned, **args)
        )
        alt_stat.append(
            hypothesis_test(alt_sample, h0, h1, stat, binned=binned, **args)
        )

    if store:
        store[sample_size] = (np.array(null_stat), np.array(alt_stat))

    return np.array(null_stat), np.array(alt_stat)


def get_t2_error(h0_distribution, h1_distribution, t1_error_rate, df_init=2.5):
    def chi2_pdf(X, df):
        return chi2.pdf(X, df=df)

    is_not_outlier = h0_distribution < np.quantile(h0_distribution, 0.99)
    is_fit_invalid = h0_distribution < 1e-8
    sample = h0_distribution[
        np.isfinite(h0_distribution) & is_not_outlier & ~is_fit_invalid
    ]
    chi2_fit, _, _ = unbinned_mle_estimation(
        sample, chi2_pdf, params={"df": df_init}, limits={"df": (0, None)}
    )
    t_threshold = chi2.ppf(1 - t1_error_rate, df=chi2_fit["df"])

    return np.count_nonzero(h1_distribution < t_threshold) / len(h1_distribution)


def binary_threshold_search(
    h0, h1, t1_error_rate=2.9e-7, t2_error_rate=0.1, n_samples=10000, n_0=500, **kwargs
):
    results = []
    val_range = (1, n_0)
    get_statistic = partial(
        test_statistic_distribution,
        h0=h0,
        h1=h1,
        stat=log_likelihood_ratio,
        null_fit=kwargs["null_fit"],
        null_limits=kwargs["null_limits"],
        alt_fit=kwargs["alt_fit"],
        alt_limits=kwargs["alt_limits"],
        alt_init=kwargs["alt_init"],
        n_samples=n_samples,
    )
    high_score = get_t2_error(*get_statistic(sample_size=n_0), t1_error_rate)
    results.append((val_range[1], high_score))
    # Find upper limit of range
    while high_score > t2_error_rate:
        high_score = high_score, get_t2_error(
            *get_statistic(sample_size=2 * val_range[1]), t1_error_rate
        )
        val_range = (val_range[1], val_range[1] * 2)
        results.append((val_range[1], high_score))
    # Binary search
    while val_range[1] - val_range[0] != 1:
        sample_value = (val_range[0] + val_range[1]) // 2
        new_score = get_t2_error(
            *get_statistic(sample_size=sample_value), t1_error_rate
        )
        results.append((sample_value, new_score))
        if new_score < t2_error_rate:
            val_range = (val_range[0], sample_value)
            high_score = new_score
        else:
            val_range = (sample_value, val_range[1])

    return results, val_range[1]


def weighted_threshold_search(
    h0,
    h1,
    t1_error_rate=2.9e-7,
    t2_error_rate=0.1,
    n_samples=10000,
    n_0=500,
    store_distributions=False,
    df_diff=3,
    **kwargs
):
    results = []
    val_range = [1, n_0]
    low_score = 0
    if store_distributions:
        store = {}
    get_statistic = partial(
        test_statistic_distribution,
        h0=h0,
        h1=h1,
        stat=log_likelihood_ratio,
        null_fit=kwargs["null_fit"],
        null_limits=kwargs["null_limits"],
        alt_fit=kwargs["alt_fit"],
        alt_limits=kwargs["alt_limits"],
        alt_init=kwargs["alt_init"],
        n_samples=n_samples,
    )
    t2_error = partial(
        get_t2_error,
        t1_error_rate=t1_error_rate,
        df_init=df_diff,
        store=store if store_distributions else None,
    )
    high_score = t2_error(*get_statistic(sample_size=n_0))
    results.append(
        (
            val_range[1],
            high_score,
            poisson.cdf(high_score * n_samples, mu=t2_error_rate * n_samples),
        )
    )
    while high_score > t2_error_rate:
        low_score, high_score = high_score, t2_error(*get_statistic(sample_size=n_0))
        val_range = [val_range[1], val_range[1] * 2]
        print(val_range, high_score)
        results.append(
            (
                val_range[1],
                high_score,
                poisson.cdf(high_score * n_samples, mu=t2_error_rate * n_samples),
            )
        )

    low_p = poisson.cdf(low_score * n_samples, mu=t2_error_rate * n_samples)
    high_p = poisson.cdf(high_score * n_samples, mu=t2_error_rate * n_samples)

    # Binary search
    while not (
        low_score * n_samples - (t2_error_rate * n_samples) ** 0.5
        <= t2_error_rate * n_samples
        <= high_score * n_samples + (t2_error_rate * n_samples) ** 0.5
    ):
        sample_value = (val_range[0] + val_range[1]) // 2
        mid_score = t2_error(*get_statistic(sample_size=n_0))
        mid_p = poisson.cdf(mid_score * n_samples, mu=t2_error_rate * n_samples)
        val_range[0], val_range[1] = sample_value - int(
            (val_range[1] - val_range[0]) / 2 * (low_p - mid_p) / (low_p - high_p)
        ), sample_value + int(
            (val_range[1] - val_range[0]) / 2 * (mid_p - high_p) / (low_p - high_p)
        )

        low_score = t2_error(*get_statistic(sample_size=n_0))
        high_score = t2_error(*get_statistic(sample_size=n_0))
        low_p = poisson.cdf(low_score * n_samples, mu=t2_error_rate * n_samples)
        high_p = poisson.cdf(high_score * n_samples, mu=t2_error_rate * n_samples)
        print(val_range[0], low_score, low_p)
        print(sample_value, mid_score, mid_p)
        print(val_range[1], high_score, high_p)
        results.append(
            (
                val_range[0],
                low_score,
                poisson.cdf(low_score * n_samples, mu=t2_error_rate * n_samples),
            )
        )
        results.append(
            (
                sample_value,
                mid_score,
                poisson.cdf(mid_score * n_samples, mu=t2_error_rate * n_samples),
            )
        )
        results.append(
            (
                val_range[1],
                high_score,
                poisson.cdf(high_score * n_samples, mu=t2_error_rate * n_samples),
            )
        )

    valid_guesses = []
    for item in results:
        error = np.sqrt(item[1] / item[0])
        lower_bound = item[1] - error
        upper_bound = item[1] + error
        if lower_bound < t2_error_rate < upper_bound:
            valid_guesses.append(item[0])

    return results, (min(valid_guesses), max(valid_guesses)), store
