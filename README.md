# Principles of Data Science Assignment - ivp24

This repository contains a solution for the PDS coursework assignment.

![Static Badge](https://img.shields.io/badge/build-stable-green) ![Static Badge](https://img.shields.io/badge/logo-gitlab-blue?logo=gitlab)

## Table of contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Running the scripts](#running-the-scripts)
4. [Features](#features)
5. [Frameworks](#frameworks)
6. [Build status](#build-status)
7. [Credits](#credits)

## Requirements

The user should have a version of Docker installed in order to ensure correct setup of environments. If that is not possible, please ensure you have all the specified packages in the `environment.yml` file correct.

## Setup


To correctly set up the environment, we utilise a Docker image. To build the image before creating the container, you can run.

```docker build -t ivp24_pds .```

The setup image will also add the necessary pre-commit checks to your git repository, ensuring the commits work correctly.

Afterwards, any time you want to use the code, you can launch a Docker container using:

```docker run --rm --name pds -ti ivp24_pds```

If you want to make changes to the repository, you would likely need to use your Git credentials. A safe way to load your SSH keys was to use the following command:

```docker run --rm -v <ssh folder on local machine>:/root/.ssh --name pds -ti ivp24_pds```

This copies your keys to the created container and you should be able to run all required git commands.

## Running the scripts
All scripts can be run in a similar way, as shown below. No configuration files have to be specified for the scripts, as the default parameters are used if no configuration is passed. All plots produced by the scripts can be viewed by attaching the container to a Visual Studio Code instance, or by copying to the local machine through the command:

```docker cp pds:/ivp24/<image_name> <destination>```


### Part (c)

In order to run the script for the relevant part, execute the following command while still in the main project directory:

```python -m src.runnable_scripts.part_c_check <config_file>```

A sample configuration file can be found in the `configs` folder. If a configuration is not specified, the default one will be used. If executed correctly, the program should return a list of parameters settings and the evaluated area for the pdf. If correctly implemented, all should return a value close to 1, showing a normalized probability function.

### Part (d)
In order to run the script for the relevant part, execute the following command while still in the main project directory:

```python -m src.runnable_scripts.part_d_plot <config_file>```

A sample configuration file can be found in the `configs` folder. If a configuration is not specified, the default one will be used. If executed correctly, the program should produce a plot of the distribution with the given parameters, as well as the background and signal components.

### Part (e)
In order to run the script for the relevant part, execute the following command while still in the main project directory:

```python -m src.runnable_scripts.part_e_sampling <config_file>```

A sample configuration file can be found in the `configs` folder. If a configuration is not specified, the default one will be used. If executed correctly, the program should produce a plot of the distribution. This includes a histogram of the produced samples, the estimated distribution,
as well as the (unnormalized) components of the estimate.

### Parts (f, g)
In order to run the script for the relevant parts, execute the following command while still in the main project directory:

```python -W ignore -m src.runnable_scripts.part_f_ht <config_file>``` for Part (f) or

```python -W ignore -m src.runnable_scripts.part_g_ht <config_file>``` for Part (g)

We recommend using the ignore warnings tag in the occasions that fits become invalid. A sample configuration file can be found in the `configs` folder. If a configuration is not specified, the default one will be used. The features of the configuration are displayed below. If correctly executed, the program will give either a possible range for the number of data points to differentiate the hypotheses, or a conrete value (which might have a high uncertainty). It will then produce plots for the fit procedure and show examples of different test statistic distributions for a range of dataset sizes.
## Features
While the implementations of parts (c), (d) and (e) are relatively straightforward, the configurations of (f) and (g) need to be explored further:
### Parameters
This section details the true parameters of the alternate hypothesis. In the case of part (g), the signal with higher $f$ value is taken as the generative one for the null hypothesis.
For each part that includes:

f: $f, \lambda, \mu, \sigma, \alpha, \beta$, given respectively using the tags `f`, `lam`, `mu`, `sigma`, `alpha`, `beta`. It is necessary that $\alpha < \beta$.

g: $f_1, f_2, \lambda, \mu_1, \mu_2, \sigma, \alpha, \beta$, given respectively using the tags `f1`, `f2`, `lam`, `mu1`, `mu2`, `sigma`, `alpha`, `beta`. It is necessary that $\alpha < \beta$.
### Initial fit
This section should contain identical members to the parameters one. The values are used to make an initial guess for the distribution when fitting the alternate hypothesis when the sample is generated from the null. Varying these numbers might ensure the fit is more stable.
### Hypothesis testing
This section describes the parameters, with which the threshold search is executed.
1. `t1_error_rate` - the type 1 error rate, i.e. with what probability do we want to incorrectly accept the null hypothesis.
2. `t2_error_rate` - the type 2 error_rate, i.e. with what probability we want to incorrectly reject the alternate hypothesis.
3. `binned` - whether we want to use a binned or unbinned fit for the distribution estimation.
4. `n_bootstraps` - the number of samples produced for a single dataset size.
5. `algorithm` - whether to use the "binary search" or "weighted search" algorithm. The former will produce a single value, while the latter will quote range. It is notable that binary search is more efficient, but might also be inaccurate, as the decisions the algorithm takes are affected by random chance. This is less so the case for the latter, but is also an effect not to be neglected.

## Build status
The build is still in the development phase and might be unstable.

## Credits

The `.pre-commit-config.yaml` configuration file content has been adapted from the Research Computing lecture notes.
