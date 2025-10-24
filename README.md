## Efficient Measurement-Driven Eigenenergy Estimation with Classical Shadows
Updated as of latest submission to PRX Quantum (10/24/25)

This repository contains the code used to generate the figures in the Applications section of "Efficient Measurement-Driven Eigenenergy Estimation with Classical Shadows" (Shen et al., 2024 https://arxiv.org/abs/2409.13691). The base code for the multi-observable dynamic mode decomposition (MODMD) algorithm is contained in the source folder and each figure/experiment has its own Jupyter notebook. 

This code can be used to replicate the results or run additional simulations with new parameters, such as different Hamiltonians or observable sets. See requirements.txt for package dependencies. Below is a table with a brief description of each simulation and the Jupyter notebook/manuscript figure to which it corresponds.

| Jupyter Notebook(s) | Figures (s) | Description |
|:-------------------:|:------------------:|-------------|
| Dynamics Prediction Notebooks| 8 | Plot predicted dynamics of TFIM observable signals for different values of $k^*$, the number of timesteps used to construct $A$.
|K Convergence Notebooks                  |  2, 4, 9, 10, 11, 12, 13                |   Plot convergence of energy levels with respect to the number of DMD snapshots K for ODMD and MODMD      |
|  Other/TFIM_gap_convergence                  |        3         |      Plot convergence of first excited state energy in TFIM with respect to the gap $E_1-E_0$ for ODMD and MODMD       |
|        Other/LiH_noise_convergence           |         5         |     Plot convergence of LiH energy levels with respect to <br> the noise level $\epsilon_{\text{noise}}$ for ODMD and MODMD       |
|Other/LiH_eigenstate_recovery Other/TFIM_eigenstate_recovery | 17, 18 | Plot convergence of eigenstates with respect to the number of DMD snapshots K for ODMD and MODMD |
|Other/qmegs_comparison Other/vqpe_comparison | 6, 7, 15, 16 | Performance comparisons of MODMD with other RTE methods (U)VQPE and QMEGS|
|Parameter Sweep Notebooks| 14 |Plot convergence of LiH eigenenergies using MODMD for different MODMD hyperparameters
