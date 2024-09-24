import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import csc_array

def bitstring_superposition_state(num_qubits: int, bitstrings: list):
    """
    Generate an even superposition of computational basis states denoted by a
    list of bitstrings.

    :param num_qubits: number of qubits in state
    :param bitstrings: list of bitstrings to be included in the superposition state
    """

    state = np.zeros(2**num_qubits)

    for bitstring in bitstrings:
        state[int(bitstring,2)] = 1

    return state/np.linalg.norm(state)

def random_one_local_paulis(num_qubits: int, size: int) -> list:
    """
    Generate a set of unique random one-local Pauli observables as sparse matrices.

    :param num_qubits: number of qubits in state 
    :param size: number of observables to be generated
    """

    pauli_string_set = ['I' * num_qubits]
    for i in range(num_qubits):
        pauli_string_set.extend(['I' * i + p + 'I'*(num_qubits - 1) for p in ['X','Y','Z']])

    random_subset = np.random.choice(pauli_string_set,size,False)
    return [csc_array(SparsePauliOp(p).to_matrix(sparse=True),dtype='float64') for p in random_subset]

def all_one_local_paulis(num_qubits: int):
    """
    Generates all one-local Pauli observables as sparse matrices.

    :param num_qubits: number of qubits in state
    """
    return random_one_local_paulis(num_qubits, 3 * num_qubits + 1)

