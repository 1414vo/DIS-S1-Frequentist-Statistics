r"""! @file part_f_ht.py
@brief Runs a set of hypothesis tests between a background and 1-signal distribution
to determine the number of minimum number of data points to differentiate the 2 with some confidence.

@details Usage: `python -m src.runnable_scripts.part_f_ht <config_file>`.
Reads in a set of parameters and runs a set of hypothesis tests between a background and 1-signal distributions.
The goal is to determine the number of minimum number of data points to differentiate the 2 with some confidence,
specified in the configuration.  Configuration should contain a 'Parameters' section with
values for all 6 parameters, an 'Initial Fit' section to be used as the initialization for the alternate model,
as well as a 'Hypothesis Testing' section, specifying the method and threshodls.
If run without a specified config, a default is used.

@author Created by I. Petrov on 08/12/2023
"""
import sys
import configparser as cfg
import numpy as np

from src.distribution_utils.hypothesis_testing import (
    weighted_threshold_search,
    binary_threshold_search,
)
from src.distribution_utils.optimized_distributions import (
    distribution_pdf_optimized,
    distribution_cdf_optimized,
    generate_distribution_samples,
    background_pdf_optimized,
    background_cdf_optimized,
    generate_background_samples,
)
from src.plotting_utils.hypothesis_plots import (
    plot_threshold_search,
    plot_t_statistic_distribution,
)

if __name__ == "__main__":
    # Check configuration location
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
    elif len(sys.argv) == 1:
        input_file = "./configs/part_f_config.ini"
    else:
        print("This program accepts only a single command line argument")
        exit(1)

    config = cfg.ConfigParser()

    # Read from configuration
    config.read(input_file)

    params = {}
    null_params = {}
    initial_fit = {}

    if "Parameters" not in config.sections():
        msg = f"Could not read section 'Parameters' from file {input_file}."
        print(msg)
        exit(1)

    # Read in the set of parameters
    for param_name in ["f", "lam", "mu", "sigma", "alpha", "beta"]:
        if param_name not in config["Parameters"]:
            print(f"No parameter called {param_name} was found - terminating.")
            exit(1)
        params[param_name] = config.getfloat("Parameters", param_name)

    # Read in the set of parameters for the null hypothesis
    for param_name in ["lam", "alpha", "beta"]:
        if param_name not in config["Parameters"]:
            print(f"No parameter called {param_name} was found - terminating.")
            exit(1)
        null_params[param_name] = config.getfloat("Parameters", param_name)

    if "Initial Fit" in config.sections():
        # Read in the set of parameters
        for param_name in ["f", "lam", "mu", "sigma", "alpha", "beta"]:
            if param_name not in config["Initial Fit"]:
                print(f"No parameter called {param_name} was found - terminating.")
                exit(1)
            initial_fit[param_name] = config.getfloat("Initial Fit", param_name)

    else:
        print("No initial fit configuration found - defaulting to given fit values.")
        initial_fit = params

    if "Hypothesis Testing" in config.sections():
        # Read in the the error rates
        if "t1_error_rate" not in config["Hypothesis Testing"]:
            print("No Type 1 Error found defaulting to 2.9e-7.")
            t1_error_rate = 2.9e-7
        else:
            t1_error_rate = config.getfloat("Hypothesis Testing", "t1_error_rate")

        if "t2_error_rate" not in config["Hypothesis Testing"]:
            print("No Type 2 Error found - defaulting to 10%.")
            t2_error_rate = 0.1
        else:
            t2_error_rate = config.getfloat("Hypothesis Testing", "t2_error_rate")

        if "binned" not in config["Hypothesis Testing"]:
            print("No binning configuration found - defaulting to a binned fit.")
            binned = True
        else:
            binned = config.getboolean("Hypothesis Testing", "binned")

        if "n_bootstraps" not in config["Hypothesis Testing"]:
            print("No number of bootstraps per test - defaulting to a 1000.")
            n_bootstraps = 1000
        else:
            n_bootstraps = config.getint("Hypothesis Testing", "n_bootstraps")

        if "algorithm" not in config["Hypothesis Testing"]:
            print("No algorithm specified - defaulting to a weighted search.")
            binary_search = False
        else:
            algorithm = config.get("Hypothesis Testing", "algorithm")
            if algorithm == "binary search":
                binary_search = True
            elif algorithm == "weighted search":
                binary_search = False
            else:
                print(
                    "Specified algorithm must be either 'binary search' or 'weighted search'"
                    + ". Defaulting to a weighted search"
                )
                binary_search = False
    else:
        print("No hypothesis testing configuratio - defaulting to:.")
        print("T1 Error Rate: 2.9e-7")
        print("T2 Error Rate: 10%")
        print("Binned Fit")
        print("1000 bootstrap iterations.")
        print("Weighted search algorithm.")
        t1_error_rate = 2.9e-7
        t2_error_rate = 0.1
        n_bootstraps = 1000
        binned = True
        binary_search = False

    h0 = {
        "pdf": background_pdf_optimized,
        "cdf": background_cdf_optimized,
        "rvs": generate_background_samples,
    }
    h1 = {
        "pdf": distribution_pdf_optimized,
        "cdf": distribution_cdf_optimized,
        "rvs": generate_distribution_samples,
    }
    fit_config = {
        "null_fit": null_params,
        "alt_fit": params,
        "alt_init": initial_fit,
        "null_limits": {},
        "alt_limits": {"f": (0.01, 1)},
    }

    np.random.seed(0)
    if binary_search:
        results = binary_threshold_search(
            h0,
            h1,
            n_iter=n_bootstraps,
            n_0=500,
            t1_error_rate=t1_error_rate,
            t2_error_rate=t2_error_rate,
            binned=binned,
            store_distributions=True,
            fit_configuration=fit_config,
        )
        print(f"The number of data points required is approximately {results[1]}.")
    else:
        results = weighted_threshold_search(
            h0,
            h1,
            n_iter=n_bootstraps,
            n_0=500,
            t1_error_rate=t1_error_rate,
            t2_error_rate=t2_error_rate,
            binned=binned,
            store_distributions=True,
            fit_configuration=fit_config,
        )

        print(
            f"The number of data points required is likely to lie in the range ({results[1][0]}, {results[1][1]})"
        )

    plot_threshold_search(results, t2_error_rate, save_path="./part_f_search.png")

    hypothesis_distributions = results[2]
    h0_distribution = []
    h1_distributions = {}
    for n_samples in hypothesis_distributions:
        h0_distribution.extend(hypothesis_distributions[n_samples][0])
        h1_distributions[n_samples] = hypothesis_distributions[n_samples][1]

    # Clear out values from an invalid fitting procedure
    h0_distribution = np.array(h0_distribution)
    h0_distribution = h0_distribution[
        (h0_distribution > 1e-8)
        & (h0_distribution < np.quantile(h0_distribution, 0.99))
    ]
    plot_t_statistic_distribution(
        h0_distribution,
        h1_distributions,
        t1_error_rate,
        t2_error_rate,
        save_path="./part_f_distributions.png",
    )
