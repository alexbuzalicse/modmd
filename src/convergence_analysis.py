from src.modmd import *

def padded_array(array: list, target_length: int, val: float) -> list:
    """
    In certain parameter regimes (usually very small K or d), the singular value thresholding will only allow for a few eigenenergies
    to be extracted. This leads to the results matrix obtained in some functions in this file not having constant row length. This
    function pads the array to a target row length, where val is inserted into each row as necessary.

    :param array: array to be padded
    :param target_length: target row length
    :param val: value with which to pad the array
    """

    for row in array:
        row += [val] * (target_length - len(row))
    return array

def varying_K_results(num_observables: int, noise_threshold: float, X_elements: ndarray,
                      delta_t:float, K_values:list, kd_ratio: float, max_energy_level: int) -> ndarray:
    """
    Returns a matrix with shape len(K_values) x (max_energy_level+1). The (i,j)th element of the matrix is
    the estimated j-th energy level when running MODMD with K = K_values[i]. If the algorithm failed to
    retrieve the j-th eigenenergy for K=K_values[i] the (i,j)th element is -infinity.

    :param num_observables: number of observables I
    :param noise_threshold: threshold factor at which singular values get zeroed out during construction of A matrix
    :param X_elements: elements to populate X matrix. Should be calculated using maximum K and d values in analysis
    (see modmd.py/generate_X_elements for more details).
    :param delta_t: hyperparameter ∆t representing the time step
    :param K_values: value of K for which we want to compute estimated eigenenergies
    :param kd_ratio: ratio of K/d to be held constant when varying K
    :param max_energy_level: maximum energy level we want to compute (max_energy_level = 3 if we want to compute
    ground state energy and first three excited state energies)
    """
    results = []

    for K in K_values:
        results.append(list(modmd_eigenenergies(num_observables,noise_threshold,X_elements,delta_t,K,kd_ratio,max_energy_level)))

    return padded_array(results,max_energy_level+1,-np.inf)

def varying_I_results(I_values: list, noise_threshold: float, X_elements: ndarray,
                      delta_t:float, K:int, kd_ratio: float, max_energy_level: int) -> list:
    
    """
    Returns a matrix with shape len(I_values) x (max_energy_level+1). The (i,j)th element of the matrix is
    the estimated j-th energy level when running MODMD with I = I_values[i]. If the algorithm failed to
    retrieve the j-th eigenenergy for I=I_values[i] the (i,j)th element is -infinity.

    :param I_values: array of values for number of observables I
    :param noise_threshold: threshold factor at which singular values get zeroed out during construction of A matrix
    :param X_elements: elements to populate X matrix. Should be calculated using maximum K and d values in analysis
    (see modmd.py/generate_X_elements for more details).
    :param delta_t: hyperparameter ∆t representing the time step
    :param K: values of K for which we want to compute estimated eigenenergies
    :param kd_ratio: ratio of K/d to be held constant when varying K
    :param max_energy_level: maximum energy level we want to compute (max_energy_level = 3 if we want to compute
    ground state energy and first three excited state energies)
    """
    
    results = []

    for I in I_values:
        X_element_subset = [X_elements[j] for j in range(len(X_elements)) if j % max(I_values) < I]
        results.append(list(modmd_eigenenergies(I,noise_threshold,X_element_subset,delta_t,K,kd_ratio,max_energy_level)))
    return padded_array(results,max_energy_level+1,-np.inf)