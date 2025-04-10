import numpy as np
from numpy import ndarray
from scipy.sparse._csr import csr_matrix

def get_color_set(system: str) -> list:
    """
    Color sets for MODMD paper.

    :param system: quantum system for which results are plotted
    """
    color_sets = {
        'TFIM': ['#c8f0a5','#7bccc4','#43a2ca','#0868ac'],
        'LiH': ['#fcc5c0','#f768a1','#7a0177', '#240046'],
        'BeH2': ['#1984c5', '#63bff0', '#de6e56', '#c23728'],
        'N2': ['#EC8F5E', '#F3B664', '#F1EB90', '#9FBB73'],
        'Heisenberg': ['#A67B00', '#D3AE36', '#848482', '#C1C1C1']
    }
    return color_sets[system]
    
def overlap(state1, state2):
    """
    Calculate the overlap, i.e. |<state1|state2>|^2, of two quantum states.

    :param state1: first state
    :param state2: second state
    """
    r = state1.conj()@state2
    return np.real(r*r.conj())

def vector_subspace_canonical_angle(v: ndarray, subspace_vectors: list, return_cosine:bool = True) -> float:
    """
    Calculate the canonical angle between the vector v and the subspace spanned by the vectors in the list
    subspace_vectors.

    :param v: vector
    :subspace_vectors: list of vectors that span the subspace
    :return_cosine: returns the canonical angle θ if False, otherwise returns cos(θ)
    """

    U = np.column_stack(subspace_vectors)
    Q, _ = np.linalg.qr(U)

    v_proj = Q @ (Q.conj().T @ v)

    cos_theta = np.real(np.vdot(v, v_proj)) / (np.linalg.norm(v) * np.linalg.norm(v_proj))

    return cos_theta if return_cosine else np.arccos(cos_theta)

def residual_norm(hamiltonian: csr_matrix, eigenstate: ndarray, eigenenergy: float, hamiltonian_norm: float) -> float:
    """
    Returns the residual norm for the approximate eigenstate/eigenenergy as per Eq. 47 in the ODMD paper.

    :param hamiltonian: system Hamiltonian
    :param eigenstate: approximate eigenstate recovered by MODMD
    :param eigenenergy: approximate eigenenergy recovered by MODMD
    :hamiltonian_norm: 2-norm of the Hamiltonian (normalization factor)
    """
    return np.linalg.norm(hamiltonian@eigenstate - eigenenergy*eigenstate,ord=2)/hamiltonian_norm

def trace_product(A: ndarray, B: ndarray) -> float:
    """
    Faster way to calculate Tr(AB).

    :param A: Left operand in matrix product
    :param B: Right operand in matrix product
    """
    return np.sum(A * B.T)