## Efficient Measurement-Driven Eigenenergy Estimation with Classical Shadows

This repository contains the code used to generate the figures in the Applications section of "Efficient Measurement-Driven Eigenenergy Estimation with Classical Shadows" (Shen et al., 2024 https://arxiv.org/abs/2409.13691). The base code for the multi-observable dynamic mode decomposition (MODMD) algorithm is contained in the source folder and each figure/experiment has its own Jupyter notebook. 

This code can be used to replicate the results or run additional experiments with new parameters, such as different Hamiltonians or observable sets. We use the Numpy, Scipy, cmath, and Qiskit (version 1.0 or greater) libraries. Below is a table with a brief description of each experiment and the Jupyter notebook/manuscript figure to which it corresponds.

| Manuscript Figure | Jupyter Notebook | Description |
|:-------------------:|:------------------:|-------------|
|2                   |  TFIM_K_convergence                |   Plot convergence of TFIM energy levels with respect to<br>the number of DMD snapshots K for ODMD and MODMD      |
|  3                 |       TFIM_gap_convergence           |      Plot convergence of first excited state energy in TFIM with <br>respect to the gap $E_1-E_0$ for ODMD and MODMD       |
|    4               |     LiH_K_convergence             | Plot convergence of LiH energy levels with respect to<br>the number of DMD snapshots K for ODMD and MODMD            |
|        5           |         LiH_noise_convergence         |     Plot convergence of LiH energy levels with respect to <br> the noise level $\epsilon_{\text{noise}}$ for ODMD and MODMD       |
|                   |                  |             |
