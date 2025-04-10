from numpy import ndarray
import numpy as np
from cmath import phase
from scipy.linalg import eig

def generate_X_elements(observables:list, max_d:int, max_K:int, reference_state: ndarray,
                        evolved_reference_states:ndarray) ->ndarray:
    """
    Generates the elements of the MODMD X matrix as per Equations 11 and 12. For any d <= max_d and K <= max_K,
    X can be constructed via a subset of the elements of X ∈ R^(dI x K+1). Therefore, when performing
    a convergence analysis where d and K are varied up to some values max_d and max_K, this function should be called
    with parameters max_d and max_K. Then, a subset of the returned array can be passed to X_matrices for any
    d < max_d and/or K < max_K. This prevents needless recalculation of any matrix elements.

    :param observables: Set of measured observables {O_i}
    :param max_d: maximum d value for X dimension
    :param max_K: maximum K value for X dimension
    :param reference_state: algorithm hyperparameter phi_0
    :param evolved_reference_states: phi_0(t) for t = k∆t and k = 0,1,...,(max_K+max_d) as discussed in Section III.A.
    """

    # Calculate each product <phi_0|O_i first as these are time independent
    left_prods = [np.conj(reference_state)@Oi for Oi in observables]

    # Caluclate inner products <phi_0|O_i|phi_0(t)
    X_elements = []
    for j in range(0, max_d + max_K + 1):
        for left_prod in left_prods:
            X_elements.append(left_prod @ evolved_reference_states[j])

    return np.array(X_elements)

def X_matrices(num_observables:int, d:int, K:int, X_elements:ndarray):
    """
    Returns MODMD X matrix and its shifted counterpart X' as per Equation 12.

    :param num_observables: number of observables I
    :param d: dimension d of the observable history obtained via Takens' Embedding
    :param k: dimension K of the DMD matrix, number of snapshots in DMD process
    :param X_elements: flattened array of elements in X. Useful to have as a parameter rather than calculating in this
     function so that X can be generated for different values of d and k without repeating matrix element calculation.
    """

    # Initialize matrices
    X = np.zeros((d * num_observables, K+1), dtype=complex)
    Xp = np.zeros((d * num_observables, K+1), dtype=complex)

    # Populate X
    for i in range(K+1):
        X[:,i] = X_elements[i*num_observables:(d+i)*num_observables]

    # Shift columns to form X'
    Xp[:, :-1] = X[:, 1:]
    Xp[:, -1] = X_elements[(K+1) * num_observables:(d + K + 1) * num_observables]

    return X, Xp

def A_matrix(noise_threshold:float, X:ndarray, Xp:ndarray, similarity_transform = True) -> ndarray:
    """
     Calculate A = X'X+ with singular value thresholding. If similarity_transform == True, we apply
     the spectrum-preserving transform detailed in Eq. 69 of the ODMD paper. This outputs an A that
     has smaller dimensions as X'X+ but the same spectrum, thus making eigenphase calculation more efficient.
     Set similarity_transform to False only if this function is being used to determine properties of
     the system that are not eigenenergies.

    :param noise_threshold: threshold factor at which singular values get zeroed out
    :param X: ODMD/MODMD X matrix
    :param Xp: ODMD/MODMD X' matrix
    :param spectral_transformation: boolean for determining whether to apply similarity transform to A
    """

    # Compute SVD
    U, singular_values, Vh = np.linalg.svd(X)
    V = Vh.T.conj()

    # Perform singular value thresholding and truncate matrices
    rank = np.sum(singular_values > noise_threshold * singular_values[0])
    truncated_U = U[:, :rank]
    truncated_sigma_inverse = np.diag(1/singular_values[:rank])
    truncated_V = V[:, :rank]

    # Return A = X'X+ ∼= U^†*X'*V*Σ^(-1)
    if similarity_transform:
        return np.transpose(np.conj(truncated_U)) @ Xp @ truncated_V @ truncated_sigma_inverse
    return Xp @ truncated_V @ truncated_sigma_inverse @ np.transpose(np.conj(truncated_U))

def modmd_eigenenergies(num_observables: int, noise_threshold: float, X_elements: ndarray,
                      delta_t:float, K:int, kd_ratio: float, max_energy_level: int) -> ndarray:
    
    """
    Runs MODMD for fixed dimensions K and d. Returns an array of estimated eigenenergies.

    :param num_observables: number of observables I
    :param noise_threshold: threshold factor at which singular values get zeroed out during construction of A matrix
    :param X_elements: elements to populate X matrix. Should be calculated using maximum K and d values in analysis
    (see modmd.py/generate_X_elements for more details).
    :param delta_t: hyperparameter ∆t representing the time step
    :param K: value of K for which we want to compute estimated eigenenergies
    :param kd_ratio: ratio of K/d
    :param max_energy_level: maximum energy level we want to compute (max_energy_level = 3 if we want to compute
    ground state energy and first three excited state energies)
    """

    # Calculate d based on K/d ratio, set d = 1 if int(K/ratio) == 0
    d = max(int(K/kd_ratio),1)

    # Generate system matrix as per Algorithm 1
    X, Xp = X_matrices(num_observables,d,K,X_elements)
    A = A_matrix(noise_threshold,X,Xp)

    # Extract eigenenergies from eigenphases of A (up to E_max_energy_level)
    A_eigenvalues = np.linalg.eigvals(A)
    eigenenergies = [-phase(z)/delta_t for z in A_eigenvalues]

    return np.unique(eigenenergies)[:max_energy_level + 1]

def modmd_eigenstates(num_observables: int, noise_threshold: float, X_elements: ndarray,
                      delta_t:float, K:int, kd_ratio: float, max_energy_level: int, evolved_Oi_phi_0_states) -> list:
    """
    Runs MODMD for fixed dimensions K and d. Returns an array of estimated eigenstates and associated eigenenergies
    as per Equations 15-18.

    :param num_observables: number of observables I
    :param noise_threshold: threshold factor at which singular values get zeroed out during construction of A matrix
    :param X_elements: elements to populate X matrix. Should be calculated using maximum K and d values in analysis
    (see modmd.py/generate_X_elements for more details).
    :param delta_t: hyperparameter ∆t representing the time step
    :param K: value of K for which we want to compute estimated eigenenergies
    :param kd_ratio: ratio of K/d
    :param max_energy_level: maximum energy level we want to compute (max_energy_level = 3 if we want to compute
    ground state energy and first three excited state energies)
    :evolved_Oi_phi_0_states: a 2d list where the i_th element is a list containing the states e^(iHa∆t) for a = 0,1,...
    as per Equation 18. Passed as a parameter so that this doesn't need to be recalculated for different values of K
    """

    # Calculate d based on K/d ratio, set d = 1 if int(K/ratio) == 0
    d = max(int(K/kd_ratio),1)

    # Generate system matrix as per Algorithm 1. Don't perform the similarity transform since we care
    # about the eigenvectors, not just eigenvalues
    X, Xp = X_matrices(num_observables,d,K,X_elements)
    A = A_matrix(noise_threshold,X,Xp,similarity_transform=False)
    wl, vl = eig(A,left=True,right=False)

    # Get left eigenvectors of A
    rounded_wl = np.round(wl,15)
    rounded_wl = rounded_wl[rounded_wl.nonzero()]    
    indices = np.argsort([-phase(z)/delta_t for z in rounded_wl])
    A_left_eigenvectors = [vl[:,i]/np.linalg.norm(vl[:,i]) for i in indices[:max_energy_level+1]]
    approximate_eigenenergies = sorted([-phase(z)/delta_t for z in rounded_wl])

    # Generate approximate eigenstates as per Equation 18
    approximate_eigenstates = []

    for n in range(max_energy_level+1):
        right_prods = []

        for i in range(1,num_observables+1):
            
            for a in range(d):
                
                z_ai = A_left_eigenvectors[n][a*num_observables + i - 1]
                right_prods.append(z_ai*evolved_Oi_phi_0_states[i-1][a])

        approximate_nth_eigenstate = np.sum(right_prods,0)
        approximate_eigenstates.append(approximate_nth_eigenstate/np.linalg.norm(approximate_nth_eigenstate))
        
    return approximate_eigenstates, approximate_eigenenergies