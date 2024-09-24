from numpy import ndarray
import numpy as np

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
    :return:
    """

    # Calculate each product <phi_0|O_i first as these are time independent
    left_prods = [reference_state@Oi for Oi in observables]

    # Caluclate inner products <phi_0|O_i|phi_0(t)
    X_elements = []
    for j in range(1, max_d + max_K + 1):
        for left_prod in left_prods:
            X_elements.append(left_prod @ evolved_reference_states[j])

    return np.array(X_elements)

def X_matrices(num_observables:int, d:int, K:int, X_elements:ndarray) ->(ndarray,ndarray):
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

    X_element_subset = np.array(X_elements[:num_observables*(d+K+1)])

    # Populate X
    for i in range(K+1):
        X[:,i] = X_element_subset[i*num_observables:(d+i)*num_observables]

    # Shift columns to form X'
    Xp[:, :-1] = X[:, 1:]
    Xp[:, -1] = X_element_subset[(K + 1) * num_observables:(d + K + 1) * num_observables]

    return X, Xp

def A_matrix(noise_threshold:float, X:ndarray, Xp:ndarray) -> ndarray:
    """
     Calculate A = X'X+ with singular value thresholding.

    :param noise_threshold: threshold factor at which singular values get zeroed out
    :param X: ODMD/MODMD X matrix
    :param Xp: ODMD/MODMD X' matrix
    :returns: A matrix.
    """

    # Compute SVD
    U, singular_values, Vh = np.linalg.svd(X)
    V = Vh.T.conj()

    # Perform singular value thresholding and truncate matrices
    rank = np.sum(singular_values > noise_threshold * singular_values[0])
    truncated_U = U[:, :rank]
    truncated_sigma_inverse = np.diag(1/singular_values[:rank])
    truncated_V = V[:, :rank]

    # Return A = X'X+ = U^†*X'*V*Σ^(-1)
    return np.transpose(np.conj(truncated_U)) @ Xp @ truncated_V @ truncated_sigma_inverse