r"""! @file part_c_check.py
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
import json
import numpy as np
from scipy import integrate
from src.distribution_utils.distribution import distribution_pdf

if __name__ == "__main__":
    # Check configuration location
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
    elif len(sys.argv) == 1:
        input_file = "./configs/distr_parameters.ini"
        print(input_file)
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
            params.append(json.loads(config.get("Parameters", param_name)))
        except cfg.NoSectionError:
            msg = f"Could not read section 'Parameters' from file {input_file}."
            print(msg)
            exit()

    params = np.array(params)

    for i in range(params.shape[1]):
        set_params = params[:, i]

        area = integrate.quad(
            lambda X: distribution_pdf(X, *set_params), set_params[-2], set_params[-1]
        )[0]
        print(f"Calculated area for parameter set {set_params}: {area:.5f}")
