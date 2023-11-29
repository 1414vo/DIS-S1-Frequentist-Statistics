r"""! @file part_d_plot.py
@brief Plots the two components of the distribution, alongside the

@details Usage: `python -m src.runnable_scripts.part_d_plot <config_file>`.
Reads in a set of parameters and plots the two components of the distribution,
on top of the true distribution. Configuration should contain a 'Parameters' section with
values for all 6 parameters. If run without a specified config, a default is used.

@author Created by I. Petrov on 18/11/2023
"""

import sys
import configparser as cfg
from src.plotting_utils.distribution_plots import plot_distribution_mix

if __name__ == "__main__":
    # Check configuration location
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
    elif len(sys.argv) == 1:
        input_file = "./configs/true_parameters.ini"
    else:
        print("This program accepts only a single command line argument")
        exit()

    config = cfg.ConfigParser()

    # Read from configuration
    config.read(input_file)

    params = []

    # Read in the set of parameters
    for param_name in ["f", "lam", "mu", "sigma", "alpha", "beta"]:
        try:
            params.append(config.getfloat("Parameters", param_name))
        except cfg.NoSectionError:
            msg = f"Could not read section 'Parameters' from file {input_file}."
            print(msg)
            exit()
    print(params)

    plot_distribution_mix(*params)
