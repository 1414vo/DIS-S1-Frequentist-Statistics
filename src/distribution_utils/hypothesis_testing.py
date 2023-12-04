from src.distribution_utils.distributions import partial_pdf, generate_sample
from src.distribution_utils.estimation import (
    unbinned_mle_estimation,
    binned_mle_estimation,
)
from scipy.stats import chi2
from functools import partial
import numpy as np


def log_likelihood_ratio(sample, h0, h1):
    h1_ll = np.sum(np.log(h1(sample)))
    h0_ll = np.sum(np.log(h0(sample)))
    return -2 * (h0_ll - h1_ll) if h0_ll < h1_ll else 0


def hypothesis_test(sample, h0, h1, statistic=log_likelihood_ratio, **kwargs):
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

    null_fit_params, _, _ = unbinned_mle_estimation(
        sample, h0, params=null_params, limits=null_limits
    )
    alternate_fit_params, _, _ = unbinned_mle_estimation(
        sample, h1, params=alternate_params, limits=alternate_limits
    )

    h0_fit = partial_pdf(pdf=h0, **null_fit_params.to_dict())
    h1_fit = partial_pdf(pdf=h1, **alternate_fit_params.to_dict())

    return statistic(sample, h0_fit, h1_fit)


def binned_hypothesis_test(sample, h0, h1, statistic=log_likelihood_ratio, **kwargs):
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

    null_fit_params, _, _ = binned_mle_estimation(
        sample, h0["cdf"], params=null_params, limits=null_limits
    )

    alternate_fit_params, _, _ = binned_mle_estimation(
        sample, h1["cdf"], params=alternate_params, limits=alternate_limits
    )

    h0_fit = partial_pdf(pdf=h0["pdf"], **null_fit_params.to_dict())
    h1_fit = partial_pdf(pdf=h1["pdf"], **alternate_fit_params.to_dict())

    return statistic(sample, h0_fit, h1_fit)


def test_statistic_distribution(
    h0, h1, bounds, stat, n_samples=10000, sample_size=1000, **kwargs
):
    null_stat = []
    alt_stat = []
    h0_fit = partial_pdf(h0, **kwargs["null_fit"])
    h1_fit = partial_pdf(h1, **kwargs["alt_fit"])
    for _ in range(n_samples):
        null_sample = generate_sample(h0_fit, *bounds, N=sample_size, random_seed=None)
        alt_sample = generate_sample(h1_fit, *bounds, N=sample_size, random_seed=None)

        args = {
            "null_params": kwargs["null_fit"],
            "null_limits": kwargs["null_limits"],
            "alternate_params": kwargs["alt_fit"],
            "alternate_limits": kwargs["alt_limits"],
        }

        null_stat.append(hypothesis_test(null_sample, h0, h1, stat, **args))
        alt_stat.append(hypothesis_test(alt_sample, h0, h1, stat, **args))

    return np.array(null_stat), np.array(alt_stat)


def binned_test_statistic_distribution(
    h0, h1, stat, n_samples=10000, sample_size=1000, **kwargs
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
        null_args = args
        null_args["alternate_params"] = kwargs["alt_init"]
        null_stat.append(binned_hypothesis_test(null_sample, h0, h1, stat, **args))
        alt_stat.append(binned_hypothesis_test(alt_sample, h0, h1, stat, **args))

    return np.array(null_stat), np.array(alt_stat)


def get_t2_error(h0_distribution, h1_distribution, t1_error_rate):
    def chi2_pdf(X, df):
        return chi2.pdf(X, df=df)

    chi2_fit, _, _ = unbinned_mle_estimation(
        h0_distribution, chi2_pdf, params={"df": 3}, limits={"df": (0, None)}
    )
    t_threshold = chi2.ppf(1 - t1_error_rate, df=chi2_fit["df"])
    print(chi2_fit)
    return np.count_nonzero(h1_distribution < t_threshold) / len(h1_distribution)


def find_size_threshold(
    h0, h1, t1_error_rate=2.9e-7, t2_error_rate=0.1, n_samples=10000, n_0=500, **kwargs
):
    val_range = (0, n_0)
    low_score = 1
    get_statistic = partial(
        binned_test_statistic_distribution,
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
    print(val_range, low_score, high_score)
    # Find upper limit of range
    while high_score > t2_error_rate:
        low_score, high_score = high_score, get_t2_error(
            *get_statistic(sample_size=2 * val_range[1]), t1_error_rate
        )
        val_range = (val_range[1], val_range[1] * 2)
        print(val_range, low_score, high_score)

    # Binary search
    while val_range[1] - val_range[0] != 1:
        print(val_range, high_score)
        sample_value = (val_range[0] + val_range[1]) // 2
        new_score = get_t2_error(
            *get_statistic(sample_size=sample_value), t1_error_rate
        )
        if new_score < t2_error_rate:
            val_range = (val_range[0], sample_value)
            high_score = new_score
        else:
            val_range = (sample_value, val_range[1])
            low_score = new_score
        print(val_range, low_score, high_score)
    return val_range[1]
