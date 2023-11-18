# Principles of Data Science Assignment - ivp24

This repository contains a solution for the PDS coursework assignment

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

`docker build -t ivp24_pds .`

The setup image will also add the necessary pre-commit checks to your git repository, ensuring the commits work correctly.

Afterwards, any time you want to use the code, you can launch a Docker container using:

`docker run --rm -ti ivp24_pds`

## Running the scripts

## Features

## Frameworks

## Build status

## Credits

The `.pre-commit-config.yaml` configuration file content has been adapted from the Research Computing lecture notes.
