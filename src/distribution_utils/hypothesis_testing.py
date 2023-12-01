from src.distribution_utils.distribution import (
    partial_pdf,
)
from src.distribution_utils.estimation import unbinned_mle_estimation
import numpy as np


def log_likelihood_ratio(sample, h0, h1):
    h1_ll = np.prod(np.log(h1(sample)))
    h0_ll = np.prod(np.log(h0(sample)))
    return -2 * (h0_ll - h1_ll)


def hypothesis_test(sample, h0, h1, **kwargs):
    if "null_params" in kwargs:
        null_params = kwargs["null_params"]
    else:
        null_params = []

    if "null_limits" in kwargs:
        null_limits = kwargs["null_limits"]
    else:
        null_limits = []

    if "alternate_params" in kwargs:
        alternate_params = kwargs["alternate_params"]
    else:
        alternate_params = []

    if "alternate_limits" in kwargs:
        alternate_limits = kwargs["alternate_limits"]
    else:
        alternate_limits = []

    null_fit_params, _, _ = unbinned_mle_estimation(
        sample, h0, params=null_params, limits=null_limits
    )
    alternate_fit_params, _, _ = unbinned_mle_estimation(
        sample, h1, params=alternate_params, limits=alternate_limits
    )

    h0_fit = partial_pdf(pdf=h0, **null_fit_params.to_dict())
    h1_fit = partial_pdf(pdf=h1, **alternate_fit_params)

    return log_likelihood_ratio(sample, h0_fit, h1_fit)
