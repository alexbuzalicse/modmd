import numpy as np
from scipy.linalg import eigh, eig, toeplitz

def Eig(t, H_matrix, S_matrix, r_SVD=1E-2, eigid=0):
    
    """
    Compute solution to the generalized eigenvalue problems of VQPE.
    
    Parameters:
    - t: current time step
    - H_matrix: Toeplitz matrix with entries 
                U_{jk} = <ɸ|exp(iHjdt)Hexp(-iHkdt)|ɸ>_{jk} + eps_{jk}
    - S_matrix: Toeplitz matrix with entries
                S_{jk} = <ɸ|exp(iHjdt)exp(-iHkdt)|ɸ>_{jk} + eps_{jk}
    - r_SVD: Relative threshold value for SVD thresholding (0 < r_SVD < 1)
    - eigid: Index of the target eigenstate to be approximated
    
    Returns:
    - Approximate target eigenenergy estimated by VQPE
    """
    
    Ht, St = H_matrix[:t+1,:t+1], S_matrix[:t+1,:t+1]
    sval_S, rot = eigh(St)
    trunc = 0    
    sval_cut = r_SVD * np.max(sval_S)
             
    for j in range(St.shape[0]):
        if sval_S[j] > sval_cut:
            trunc = j                # singular value truncation 
            break
            
    Srot, Hrot = np.diag(sval_S[trunc:]), rot[:, trunc:].conj().T @ Ht @ rot[:, trunc:]
    eigval_H, eigvec_H = eigh(Hrot, Srot, eigvals_only=False) 
    return np.sort(eigval_H)[eigid]

###### This is the main function to call for running VQPE ######
def VQPE(S_row_complex, H_row_complex, steps, r_SVD=1E-2, eigid=0):
    
    """
    Run VQPE to estimate a target eigenenergy with SVD truncation.
    
    Parameters:
    - S_row_complex: 1d array containing the first row of the noisy Toeplitz overlap matrix
                     (<ɸ|exp(-iHldt)|ɸ> of size NT_max where NT_max counts the total number of time steps)
    - H_row_complex: 1d array containing the first row of the noisy Toeplitz Hamiltonian matrix
                     (<ɸ|H exp(-iHldt)|ɸ> of size NT_max where NT_max counts the total number of time steps)
    - steps: 1d array containing a set of timesteps 
    - r_SVD: Relative threshold value for SVD thresholding (0 < r_SVD < 1)
    - eigid: Index of the target eigenstate to be approximated
    
    Returns:
    - E_VQPE: 1D array containing approximate eigenenergy evaluated over N_T timesteps
    - S_matrix: full Toeplitz overlap matrix 
    - H_matrix: full Toeplitz Hamiltonian matrix
    
    *** Note: make sure to account for the timestep when computing the first rows of the matrices ***
    """
    
    E_VQPE = np.zeros(len(steps)) 
    S_matrix, H_matrix = toeplitz(S_row_complex).T, toeplitz(H_row_complex).T

    for j in range(len(steps)):
        E_VQPE[j] = Eig(steps[j], H_matrix, S_matrix, r_SVD, eigid)
    return E_VQPE


def EigU(t, U_matrix, S_matrix, r_SVD=1E-2, eigid=0):
    
    """
    Compute solution to the generalized eigenvalue problems of UVQPE.
    
    Parameters:
    - t: current time step
    - U_matrix: Toeplitz matrix with entries 
                U_{jk} = <ɸ|exp(iHjdt)Hexp(-iHkdt)|ɸ>_{jk} + eps_{jk}
    - S_matrix: Toeplitz matrix with entries
                S_{jk} = <ɸ|exp(iHjdt)exp(-iHkdt)|ɸ>_{jk} + eps_{jk}
    - r_SVD: Relative threshold value for SVD thresholding
    - eigid: Index of the target eigenstate to be approximated
    
    Returns:
    - Approximate target eigenenergy (multiplied by unit timestep) estimated by UVQPE
    """
    
    Ut, St = U_matrix[:t+1,:t+1], S_matrix[:t+1,:t+1]
    sval_S, rot = eigh(St)
    trunc = 0
    sval_cut = r_SVD * np.max(sval_S)
    
    for j in range(St.shape[0]):
        if sval_S[j] > sval_cut:
            trunc = j                # singular value truncation 
            break
            
    Srot, Urot = np.diag(sval_S[trunc:]), rot[:, trunc:].conj().T @ Ut @ rot[:, trunc:]
    eigval_U, eigvec_U = eig(Urot, Srot) 
    eigarg_U = - np.angle(eigval_U)
    return np.sort(eigarg_U)[eigid]


###### This is the main function to call for running UVQPE ######
def UVQPE(N_T, S_row_complex, steps, r_SVD=1E-2, eigid=0):
    
    """
    Run UVQPE to estimate a target eigenenergy.
    
    Parameters:
    - N_T: total number of time steps
    - S_row_complex: 1d array containing the first row of the noisy Toeplitz overlap matrix
                     (<ɸ|exp(-iHldt)|ɸ> of size (NT_max + 1) where NT_max counts the total number of time steps)
    - steps: 1d array containing a set of timesteps 
    - r_SVD: Relative threshold value for SVD thresholding
    - eigid: Index of the target eigenstate to be approximated
    
    Returns:
    - E_UVQPE: 1D array containing approximate eigenenergy (multiplied by unit timestep) evaluated over N_T timesteps
    """

    E_UVQPE = np.zeros(len(steps))
    S_matrix = toeplitz(S_row_complex[:N_T]).T
    U_col_complex = np.zeros(N_T, dtype=complex)
    U_col_complex[0], U_col_complex[1:] = S_row_complex[1], S_row_complex[:N_T-1].conj()
    U_matrix = toeplitz(U_col_complex, S_row_complex[1:N_T+1])
    
    for j in range(len(steps)):
        E_UVQPE[j] = EigU(steps[j], U_matrix, S_matrix, r_SVD, eigid)
    return E_UVQPE