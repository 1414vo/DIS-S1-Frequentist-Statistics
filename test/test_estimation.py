from src.distribution_utils.distributions import (
    partial_pdf,
    distribution_pdf,
    generate_sample,
)
from src.distribution_utils.estimation import unbinned_mle_estimation


def test_unbinned_mle():
    """! Tests whether the unbinned maximum likelihood estimation
    produces reasonable estimates.
    """

    fixed_parameters = ["alpha", "beta"]
    limits = {"f": (0, 1), "lam": (0, None), "sigma": (0, None)}
    params = {"f": 0.3, "lam": 0.5, "mu": 4.9, "sigma": 0.04, "alpha": 4, "beta": 5}

    pdf = partial_pdf(**params)
    sample = generate_sample(pdf, min_x=4, max_x=5)

    values, _, errors = unbinned_mle_estimation(
        sample, distribution_pdf, params=params, fixed=fixed_parameters, limits=limits
    )

    assert values["f"] - 2 * errors["f"] < 0.3 and 0.3 < values["f"] + 2 * errors["f"]
    assert (
        values["lam"] - 2 * errors["lam"] < 0.5
        and 0.5 < values["lam"] + 2 * errors["lam"]
    )
    assert (
        values["mu"] - 2 * errors["mu"] < 4.9 and 4.9 < values["mu"] + 2 * errors["mu"]
    )
    assert (
        values["sigma"] - 2 * errors["sigma"] < 0.04
        and 0.04 < values["sigma"] + 2 * errors["sigma"]
    )
