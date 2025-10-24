"""
From https://github.com/zhiyanding/phase_estimation_methods/blob/main/QMEGS.py
"""

import numpy as np
from matplotlib import pyplot as plt
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
import scipy.linalg as la
from scipy.stats import truncnorm
import finufft
from itertools import permutations

def organize_spectrum_population(spectrum_raw, population_raw, p_list):
    """ -Input:
    
    spectrum_raw: np.array of original spectrum
    population_raw: np.array of original overlap
   
    -Ouput:
    
    spectrum: np.array of adjusted eigenvalues
    population: np.array of adjusted overlaps
    """
    p = np.array(p_list)
    spectrum = spectrum_raw /np.max(np.abs(spectrum_raw))#normalize the spectrum
    q = population_raw
    num_p = p.shape[0]
    print(q.shape)
    print(p.shape)
    q[0:num_p] = p/(1-np.sum(p))*np.sum(q[num_p:])
    return spectrum, q/np.sum(q)

def QMEGS_TFIM(L,J,g,p_list,d,eigenenergies=None, eigenstates=None, verbose=0): #NOTE: I CHANGED THIS TO NON-PERIODIC
    """ -Input:
    
    L: number of qubits
    J,g: parameters of TFIM
    p_list: required overlap
    d: final number of basis
    
    -Ouput:
    
    spectrum: np.array of eigenvalues
    population: np.array of overlaps
    
    Note: original initial state is uniformly drawn from unit circle.
    """
    ##----build TFIM Hamiltonian----##
    basis = spin_basis_1d(L=L)
    if verbose > 0:
        print(basis)
    
    # define site-coupling lists
    h_field=[[-g,i] for i in range(L)]
    J_zz=[[-J,i,(i+1)%L] for i in range(L-1)] # PBC
    static =[["zz",J_zz],["x",h_field]] # static part of H
    dynamic=[]
    # build Hamiltonian
    if verbose == 0:
        no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
        H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)
    else: 
        H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
    ##----Random initial state----##
    # if eigenenergies == None or eigenstates == None:
    #     eigenenergies, eigenstates = H.eigsh(k=d,which="SA")
    dim=len(eigenstates[:,0])
    initial_states = np.random.normal(0,1,dim)+1j*np.random.normal(0,1,dim)
    initial_states = initial_states/la.norm(initial_states)
    spectrum_raw = eigenenergies
    population_raw = np.abs(np.dot(eigenstates.conj().T, initial_states))**2
    ##----Reorganize spectrum and population----##
  
    spectrum, population=organize_spectrum_population(spectrum_raw, population_raw, p_list)
    # plt.plot(spectrum,population,'b-o');plt.show()
    
    return spectrum, population


def generate_Hadamard_test_data(spectrum,population,t_list,N_list):
    """ -Input:
    
    spectrum: np.array of eigenvalues
    population: np.array of overlap
    t_list: np.array of time points
    N_list: np.array of numbers of samples
   
    -Ouput:
    
    Z_Had: np.array of the output of Hadamard test (row)
    T_max: maximal Hamiltonian simulation time
    T_total: total Hamiltonian simulation time

    """
    if len(t_list)!=len(N_list):
       print('list error')
    t_list=np.array(t_list)
    N_list=np.array(N_list)
    N_list=N_list.flatten()
    N=len(t_list)
    Nsample=int(max(N_list))
    #generate true expectation
    z=population.dot(np.exp(-1j*np.outer(spectrum,t_list)))
    Re_true=(1+np.real(z))/2
    Im_true=(1+np.imag(z))/2
    #construct check matrix for different Nsample
    N_check=np.arange(Nsample).reshape([Nsample, 1])
    N_check=N_check*np.ones((1,N))
    Sign_check=np.ones((Nsample, 1))*(N_list-0.5)
    Re_check=(np.sign(N_check-Sign_check)-1)/(-2)
    Im_check=(np.sign(N_check-Sign_check)-1)/(-2)
    Re_true=np.multiply(Re_check,np.ones((Nsample, 1)) * Re_true)
    Im_true=np.multiply(Im_check,np.ones((Nsample, 1)) * Im_true)
    #simulate Hadamard test
    Re_random=np.random.uniform(0,1,(Nsample,N))
    Im_random=np.random.uniform(0,1,(Nsample,N))
    Re=np.sum(Re_random<Re_true,axis=0)/N_list
    Im=np.sum(Im_random<Im_true,axis=0)/N_list
    Z_Had = (2*Re-1)+1j*(2*Im-1)
    T_max = max(np.abs(t_list))
    T_total = sum(np.multiply(np.abs(t_list),N_list))
    return Z_Had, T_max, T_total

def generate_Hadamard_test_data_fast(spectrum, population, t_list, N_list,
                                     eps=1e-12, isign=-1):
    """
    Fast Hadamard test data generator using FINUFFT type-3 NUFFT.

    Computes:
        z(t) = Σ_j population[j] * exp(-i * spectrum[j] * t)
    much faster than np.exp(-1j * np.outer(spectrum, t_list)) for large arrays.

    Args:
        spectrum : array_like (float)
            Eigenvalue spectrum ω_j (nonuniform frequencies)
        population : array_like (complex)
            Corresponding complex amplitudes / overlaps
        t_list : array_like (float)
            Time points for evaluation (can be nonuniform)
        N_list : array_like (int)
            Sample counts for Hadamard test
        eps : float
            Desired relative precision for FINUFFT
        isign : int
            Sign convention (+1 or -1)

    Returns:
        Z_Had : ndarray (complex)
        T_max : float
        T_total : float
    """

    # --- Validation and shape prep ---
    t_list = np.ascontiguousarray(np.asarray(t_list, dtype=np.float64))
    N_list = np.ascontiguousarray(np.asarray(N_list, dtype=np.int64).flatten())
    spectrum = np.ascontiguousarray(np.asarray(spectrum, dtype=np.float64))
    population = np.ascontiguousarray(np.asarray(population, dtype=np.complex128))

    if t_list.size != N_list.size:
        raise ValueError("list error: t_list and N_list must have the same length")

    Nsample = int(np.max(N_list))
    N = t_list.size

    # --- FINUFFT plan setup ---
    # We’re computing: z_k = Σ_j population_j * exp(-i * spectrum_j * t_list_k)
    # Type 3: nonuniform -> nonuniform
    plan = finufft.Plan(3, 1, n_trans=1, eps=eps, isign=isign, dtype="complex128")
    plan.setpts(spectrum, s=t_list)
    z = plan.execute(population)

    # --- Compute true Re/Im Hadamard expectations ---
    Re_true = (1.0 + np.real(z)) / 2.0
    Im_true = (1.0 + np.imag(z)) / 2.0

    # --- Construct threshold check matrices efficiently ---
    # Avoid massive intermediate matrices by vectorized broadcasting
    N_check = np.arange(Nsample, dtype=np.float64)[:, None]
    Sign_check = (N_list - 0.5)[None, :]

    Re_check = (np.sign(N_check - Sign_check) - 1.0) / (-2.0)
    Im_check = Re_check  # identical

    Re_true_full = Re_check * Re_true
    Im_true_full = Im_check * Im_true

    # --- Simulate Hadamard test ---
    Re_random = np.random.random((Nsample, N))
    Im_random = np.random.random((Nsample, N))

    Re = np.sum(Re_random < Re_true_full, axis=0) / N_list
    Im = np.sum(Im_random < Im_true_full, axis=0) / N_list

    Z_Had = (2.0 * Re - 1.0) + 1j * (2.0 * Im - 1.0)
    T_max = np.max(np.abs(t_list))
    T_total = np.sum(np.abs(t_list) * N_list)

    return Z_Had, T_max, T_total


def generate_Z(spectrum,population,T,N,gamma):
    """ Generate Z samples for a given T,N,gamma
    Input:
    
    spectrum: np.array of eigenvalues
    population: np.array of overlap
    T : variance of Gaussian
    N : number of time samples
    gamma : truncated parameter
    
    Output: 
    
    Z_est: np.array of Z output
    t_list: np.array of time points
    T_max: maximal running time
    T_total: total running time
    """
    t_list = truncnorm.rvs(-gamma, gamma, loc=0, scale=T, size=N)
    N_list = np.ones(len(t_list))
    T_max = max(np.abs(t_list))
    T_total = sum(np.multiply(np.abs(t_list),N_list))
    Z_est, _ , _ = generate_Hadamard_test_data(spectrum,population,t_list,N_list)
    return Z_est, t_list, T_max, T_total

def generate_Z_fast(spectrum,population,T,N,gamma):
    """ Generate Z samples for a given T,N,gamma
    Input:
    
    spectrum: np.array of eigenvalues
    population: np.array of overlap
    T : variance of Gaussian
    N : number of time samples
    gamma : truncated parameter
    
    Output: 
    
    Z_est: np.array of Z output
    t_list: np.array of time points
    T_max: maximal running time
    T_total: total running time
    """
    t_list = truncnorm.rvs(-gamma, gamma, loc=0, scale=T, size=N)
    N_list = np.ones(len(t_list))
    T_max = max(np.abs(t_list))
    T_total = sum(np.multiply(np.abs(t_list),N_list))
    Z_est, _ , _ = generate_Hadamard_test_data_fast(spectrum,population,t_list,N_list)
    return Z_est, t_list, T_max, T_total


def QMEGS(Z_est, d_x, t_list, K, alpha, T):
    """
    QMEGS algorithm
    """
    N = len(Z_est)
    num_x=int(2*np.pi/d_x)
    x=np.arange(0,num_x)*d_x-np.pi
    G=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x)))/len(Z_est)) #Gaussian filter function
    Dominant_freq=np.zeros(K,dtype='float')
    for k in range(K):
        max_idx = np.argmax(G)
        Dominant_freq[k]=x[max_idx]
        interval_max=x[max_idx]+alpha/T
        interval_min=x[max_idx]-alpha/T
        G=np.multiply(G,x>interval_max)+np.multiply(G,x<interval_min) #eliminate interval
    return Dominant_freq

def QMEGS_new(Z_est, d_x, t_list, K, alpha, T):
    """
    QMEGS new algorithm
    
    Note: This code is slightly different from the algorithm in the paper. 
    
    To avoid long classical running time, we first do a rough search 
    then do a detailed search around the rough maximal point.
    """
    N = len(Z_est)
    num_x=int(2*np.pi/(d_x*10))
    num_x_detail=int(2*alpha/d_x/T)
    x_rough=np.arange(0,num_x)*d_x*10-np.pi
    G=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x_rough)))/len(Z_est)) #Gaussian filter function
    Dominant_freq=np.zeros(K,dtype='float')
    for k in range(K):
        max_idx_rough = np.argmax(G)
        Dominant_potential=x_rough[max_idx_rough]
        x=np.arange(0,num_x_detail)*d_x+Dominant_potential-alpha/T
        G_detail=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x)))/len(Z_est))
        max_idx_detail = np.argmax(G_detail)
        Dominant_freq[k]=x[max_idx_detail]
        interval_max=x[max_idx_detail]+alpha/T
        interval_min=x[max_idx_detail]-alpha/T
        G=np.multiply(G,x_rough>interval_max)+np.multiply(G,x_rough<interval_min) #eliminate interval
    return Dominant_freq


def QMEGS_new_fast(Z_est, d_x, t_list, K, alpha, T, eps=1e-12, isign=+1):
    """
    Faster QMEGS_new using FINUFFT type-3 (nonuniform -> nonuniform).
    Compatible with finufft 2.4.1.

    Returns:
        Dominant_freq : ndarray, shape (K,)
    """
    # -- ensure proper 1D shapes and dtypes --
    t_list = np.asarray(t_list, dtype=np.float64).reshape(-1)
    Z_est = np.asarray(Z_est, dtype=np.complex128).reshape(-1)

    if Z_est.size != t_list.size:
        raise ValueError(f"Z_est and t_list must have the same length; got "
                         f"{Z_est.size} vs {t_list.size}")

    N = len(Z_est)

    # grid sizes
    num_x = int(2 * np.pi / (d_x * 10))
    if num_x <= 0:
        raise ValueError("num_x computed <= 0, check d_x")
    num_x_detail = max(1, int(2 * alpha / d_x / T))

    # rough grid (1D, float64)
    x_rough = (np.arange(0, num_x) * d_x * 10 - np.pi).astype(np.float64).reshape(-1)

    # make sure arrays are contiguous
    t_list = np.ascontiguousarray(t_list, dtype=np.float64)
    x_rough = np.ascontiguousarray(x_rough, dtype=np.float64)
    Z_est = np.ascontiguousarray(Z_est, dtype=np.complex128)

    # rough plan (single execution)
    plan_rough = finufft.Plan(3, 1, n_trans=1, eps=eps, isign=isign, dtype="complex128")
    plan_rough.setpts(t_list, s=x_rough)
    G_complex = plan_rough.execute(Z_est)
    G = np.abs(G_complex / N)

    Dominant_freq = np.zeros(K, dtype=float)

    # Create a single detail plan and reuse it across iterations.
    # We still call setpts(...) each loop to update the target points.
    plan_detail = finufft.Plan(3, 1, n_trans=1, eps=eps, isign=isign, dtype="complex128")

    for k in range(K):
        # rough maximum
        max_idx_rough = int(np.argmax(G))
        Dominant_potential = float(x_rough[max_idx_rough])

        # detail grid centered around Dominant_potential
        x_detail = (np.arange(0, num_x_detail) * d_x + Dominant_potential - alpha / T).astype(np.float64).reshape(-1)
        x_detail = np.ascontiguousarray(x_detail, dtype=np.float64)

        # reuse plan_detail: set target points then execute
        plan_detail.setpts(t_list, s=x_detail)
        G_detail_complex = plan_detail.execute(Z_est)
        G_detail = np.abs(G_detail_complex / N)

        # find refined peak
        max_idx_detail = int(np.argmax(G_detail))
        Dominant_freq[k] = float(x_detail[max_idx_detail])

        # eliminate interval around found peak in the rough grid
        interval_max = x_detail[max_idx_detail] + alpha / T
        interval_min = x_detail[max_idx_detail] - alpha / T
        G = np.multiply(G, x_rough > interval_max) + np.multiply(G, x_rough < interval_min)

    return Dominant_freq