from src.modmd import *
from cmath import phase

def varying_K_results(num_observables: int, noise_threshold: float, X_elements: ndarray,
                      delta_t:float, K_values:list, kd_ratio: float, max_energy_level: int) -> ndarray:
    """
    Returns a matrix with shape len(K_values) x (max_energy_level+1). The (i,j)th element of the matrix is
    the estimated j-th energy level when running MODMD with K = K_values[i].

    :param num_observables: number of observables I
    :param noise_threshold: threshold factor at which singular values get zeroed out during construction of A matrix
    :param X_elements: elements to populate X matrix. Should be calculated using maximum K and d values in analysis
    (see modmd.py/generate_X_elements for more details).
    :param delta_t: hyperparameter ∆t representing the time step
    :param K_values: values of K for which we want to compute estimated eigenenergies
    :param kd_ratio: ratio of K/d to be held constant when varying K
    :param max_energy_level: maximum energy level we want to compute (max_energy_level = 3 if we want to compute
    ground state energy and first three excited state energies)
    :return: Matrix with results for convergence of eigenenergies with respect to K.
    """
    results = []

    for K in K_values:

        # Calculate d based on K/d ratio, set d = 1 if int(K/ratio) == 0
        d = max(int(K/kd_ratio),1)

        # Generate system matrix as per Algorithm 1
        X, Xp = X_matrices(num_observables,d,K,X_elements)
        A = A_matrix(noise_threshold,X,Xp)

        # Extract eigenenergies from eigenphases of A (up to E_max_energy_level)
        A_eigenvalues = np.linalg.eigvals(A)
        eigenenergies = [-phase(z)/delta_t for z in A_eigenvalues]
        results.append(list(np.unique(eigenenergies)[:max_energy_level + 1]))

        # For very small K or d values, the singular value thresholding will only allow for a few eigenenergies
        # to be extracted. This leads to the results matrix not having constant row length. The results array is padded
        # with -∞ where the eigenergies could not be extracted so that matrix operations (such as averaging) can still
        # be performed on the matrix. Using -∞ is practically equivalent to having a null value.
        for row in results:
            row += [-np.inf] * (max_energy_level + 1 - len(row))

    return np.array(results)