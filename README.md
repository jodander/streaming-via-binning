# README

This repository contains code for reproducing the results in the preprint "Streaming Private Continual Counting via Binning".
The code is split into a few files.

## binning.py

Includes the code for approximating factorization mechanisms based on Algorithm 1 and computing their error.

## space_check.py

Includes code for double-checking that matrices have low space complexity.

## plots_in_paper.ipynb

Jupyter Notebook file for reproducing all the plots in the paper.
Running it with existing parameters, it will load in data for plots from 'plot_data/', and produce all quantitative plots from the paper, outputting them to the folder 'plots'.
Setting the parameter 'GENERATE_DATA' to 'False' will re-generate the data, overwriting the content of 'plot_data/'.

## On dependencies and Python environment

Only 'standard' (math, numpy, pandas, scipy) packages are utilized in the .py files.
The Jupyter Notebook uses the package 'scienceplots' to style the plots.

All the data for plots was generated by running 'plots_in_paper.ipynb' with Python version 3.9.6 and package versions numpy==1.23.3, pandas==1.5.3, and scipy==1.9.2.
