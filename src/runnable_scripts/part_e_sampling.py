r"""! @file part_e_sampling.py
@brief Executes a check to determine whether the distribution is properly normalized,
as described in part (c).

@details Usage: `python -m src.runnable_scripts.part_c_check <config_file>`.
Reads in a set of parameters and computes the integral of the pdf.
It then prints it out so the user can verify that the area is approximately 1
(quoted up to 5 s.f.) Configuration should contain a 'Parameters' section with lists
of values for all 6 parameters. If run without a specified config, a default is used.

@author Created by I. Petrov on 18/11/2023
"""

import sys
import configparser as cfg
from src.distribution_utils.distribution import (
    distribution_pdf,
    partial_pdf,
    generate_sample,
)
from src.distribution_utils.estimation import unbinned_mle_estimation
from src.plotting_utils.distribution_plots import plot_mle

if __name__ == "__main__":
    # Check configuration location
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
    elif len(sys.argv) == 1:
        input_file = "./configs/part_e_config.ini"
        print(input_file)
    else:
        print("This program accepts only a single command line argument")
        exit()

    config = cfg.ConfigParser()

    # Read from configuration
    config.read(input_file)

    params = []

    if "Parameters" not in config.sections():
        msg = f"Could not read section 'Parameters' from file {input_file}."
        print(msg)
        exit()

    # Read in the set of parameters
    for param_name in ["f", "lam", "mu", "sigma", "alpha", "beta"]:
        if param_name not in config["Parameters"]:
            print(f"No parameter called {param_name} was found - terminating.")
            exit()
        params.append(config.getfloat("Parameters", param_name))

    if "n_samples" in config["Parameters"]:
        n_samples = config["Parameters"]["n_samples"]
    else:
        print(
            "No 'n_samples' parameter found - defaulting to a sample of size 100,000."
        )
        n_samples = 100000

    fixed_parameters = ["alpha", "beta"]
    limits = {"f": (0, 1), "lam": (0, None), "sigma": (0, None)}
    params = {
        "f": params[0],
        "lam": params[1],
        "mu": params[2],
        "sigma": params[3],
        "alpha": params[4],
        "beta": params[5],
    }

    pdf = partial_pdf(**params)
    sample = generate_sample(pdf, min_x=params["alpha"], max_x=params["beta"])

    values, _, errors = unbinned_mle_estimation(
        sample, distribution_pdf, params=params, fixed=fixed_parameters, limits=limits
    )

    for param in params:
        print(
            f"Estimated value for {param}: {values[param]:.3f} \u00B1 {errors[param]:.3f}"
        )

    plot_mle(
        sample,
        f=values["f"],
        lam=values["lam"],
        mu=values["mu"],
        sigma=values["sigma"],
        alpha=values["alpha"],
        beta=values["beta"],
    )
