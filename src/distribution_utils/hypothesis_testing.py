from src.distribution_utils.distribution import partial_pdf, generate_sample
from src.distribution_utils.estimation import (
    unbinned_mle_estimation,
    binned_mle_estimation,
)
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


def binned_hypothesis_test(
    sample, h0_pdf, h0_cdf, h1_pdf, h1_cdf, statistic=log_likelihood_ratio, **kwargs
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

    null_fit_params, _, _ = binned_mle_estimation(
        sample, h0_cdf, params=null_params, limits=null_limits
    )
    alternate_fit_params, _, _ = binned_mle_estimation(
        sample, h1_cdf, params=alternate_params, limits=alternate_limits
    )

    h0_fit = partial_pdf(pdf=h0_pdf, **null_fit_params.to_dict())
    h1_fit = partial_pdf(pdf=h1_pdf, **alternate_fit_params.to_dict())

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
    h0, h1, h0_cdf, h1_cdf, bounds, stat, n_samples=10000, sample_size=1000, **kwargs
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

        null_stat.append(
            binned_hypothesis_test(null_sample, h0, h0_cdf, h1, h1_cdf, stat, **args)
        )
        alt_stat.append(
            binned_hypothesis_test(alt_sample, h0, h0_cdf, h1, h1_cdf, stat, **args)
        )

    return np.array(null_stat), np.array(alt_stat)


def get_statistic_p_values(dof, alternate_distribution):
    pass


def find_size_threshold(h0, h1, t1_error_rate=2.9e-7, t2_error_rate=0.1, n_0=500):
    # Find upper limit of range
    pass
