import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.operators import PolynomialTensor
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit.quantum_info import SparsePauliOp
from numpy import ndarray

def tfim_hamiltonian(num_qubits: int, J: float, h: float) -> SparsePauliOp:
    """
    Returns the transverse field Ising model Hamiltonian in SparsePauliOp format.

    :param num_qubits: number of qubits in system
    :param J: coupling parameter
    :param h: external field parameter
    """
    zz_strings = ['I' * i + 'ZZ' + 'I'* (num_qubits - i - 2) for i in range(num_qubits-1)]
    x_strings = ['I' * i + 'X' + 'I'* (num_qubits - i - 1) for i in range(num_qubits)]
    return SparsePauliOp.from_list([(s,-J) for s in zz_strings] + [(s,-h) for s in x_strings])

def heisenberg_hamiltonian(num_qubits: int, J: float, h: float) -> SparsePauliOp:
    """
    Returns the isotropic Heisenberg Hamiltonian in SparsePauliOp format.

    :param num_qubits: number of qubits in system
    :param J: coupling parameter
    :param h: external field parameter
    """
    xx_strings = ['I' * i + 'XX' + 'I'* (num_qubits - i - 2) for i in range(num_qubits-1)]
    yy_strings = ['I' * i + 'YY' + 'I'* (num_qubits - i - 2) for i in range(num_qubits-1)]
    zz_strings = ['I' * i + 'ZZ' + 'I'* (num_qubits - i - 2) for i in range(num_qubits-1)]
    z_strings = ['I' * i + 'Z' + 'I'* (num_qubits - i - 1) for i in range(num_qubits)]

    return SparsePauliOp.from_list([(s,-J) for s in (xx_strings+yy_strings+zz_strings)] + 
                                                  [(s,-h) for s in z_strings])

def molecular_hamiltonian(molecule: str, basis_set: str, mapper:str, charge: int = 0, spin: int = 0,
                          freeze_core: bool = False, nuclear_repulsion: bool = True) -> SparsePauliOp:
    """
    Returns the hamiltonian of a given molecule in SparsePauliOp format.

    :param molecule: string representing molecular geometry as used in PySCF
    :param basis_set: basis set name as recognized by PySCF, e.g. sto-3g
    :param mapper: string representing Qiskit Hamiltonian mapper, either "jw" or "parity"
    :param charge: charge of the molecule
    :param spin: spin of the molecule
    :param electronic_only: if True, does not include the nuclear repulsion energy in the Hamiltonian
    """

    # Define ElectronicStructureProblem using PySCF
    driver = PySCFDriver(atom=molecule, basis=basis_set, charge=charge, spin=spin)
    problem = driver.run()

    # Optionally apply Freeze-Core reduction
    if freeze_core:
        transformer = FreezeCoreTransformer()
        problem = transformer.transform(problem)  

    hamiltonian = problem.hamiltonian 

    # Add nuclear repulsion energy to Hamiltonian if specified
    if nuclear_repulsion:
        hamiltonian.electronic_integrals.alpha += PolynomialTensor({
            "": hamiltonian.nuclear_repulsion_energy
        })
        hamiltonian.nuclear_repulsion_energy = None

    # Map the Hamiltonian to qubits using Jordan-Wigner mapping
    second_q_op = hamiltonian.second_q_op()

    if mapper == 'jw':
        qiskit_mapper = JordanWignerMapper()
    elif mapper == 'parity':
        qiskit_mapper = ParityMapper(num_particles=problem.num_particles)

    qubit_hamiltonian = qiskit_mapper.map(second_q_op)

    return qubit_hamiltonian

def bitstring_superposition_state(num_qubits: int, bitstrings: list):
    """
    Generate an even superposition of computational basis states denoted by a
    list of bitstrings.

    :param num_qubits: number of qubits in state
    :param bitstrings: list of bitstrings to be included in the superposition state
    """

    state = np.zeros(2**num_qubits)

    for bitstring in bitstrings:
        state[int(bitstring,2)] += 1

    return state/np.linalg.norm(state)

def random_one_local_paulis(num_qubits: int, size: int) -> list:
    """
    Generate a set of unique random one-local Pauli observables as sparse matrices.

    :param num_qubits: number of qubits in state 
    :param size: number of observables to be generated
    """

    pauli_string_set = ['I' * num_qubits]
    for i in range(num_qubits):
        pauli_string_set.extend(['I' * i + p + 'I'*(num_qubits - i - 1) for p in ['X','Y','Z']])

    random_subset = np.random.choice(pauli_string_set,size,False)
    return [SparsePauliOp(p).to_matrix(sparse=True) for p in random_subset]

def random_two_local_paulis(num_qubits: int, size: int) -> list:
    """
    Generate a set of unique random two-local Pauli observables as sparse matrices.

    :param num_qubits: number of qubits in state 
    :param size: number of observables to be generated
    """

    pauli_string_set = ['I' * num_qubits]
    for i in range(num_qubits-1):
        for p1 in ['X','Y','Z']:
            for p2 in ['X','Y','Z']:
                pauli_string_set.append('I' * i + p1 + p2 + 'I'*(num_qubits - i - 2))

    random_subset = np.random.choice(pauli_string_set,size,False)
    return [SparsePauliOp(p).to_matrix(sparse=True) for p in random_subset]
        
def partial_sum_observables(hamiltonian: SparsePauliOp, indices: list, coeff_sorting:str = 'descending') -> list:
    """
    Returns O_i for each i in indices where O_i is a partial sum observable as specified in 
    Section III.C of the manuscript. For a Hamiltonian sum_{v=1}^M k_v*P_v where P_v are Pauli
    strings and k_v are coefficients sorted by magnitude, O_i = sum_{v=1}^{M-i} k_v * P_v.
    
    By default, the coefficients will be sorted from largest to smallest. However, it might be useful to leave
    out the highest-weighted terms across each partial sum so that the observables are sufficiently different, in 
    which case you can sort from smallest to largest.
    
    :param hamiltonian: Hamiltonian for which MODMD is being used
    :param indices: indices corresponding to which partial sum observables should be generated
    :coeff_sorting: if 'descending', coefficients will be sorted largest to smallest such that the smallest ones
    get left out of the partial sum first 
    """
    reverse = True if coeff_sorting == 'descending' else False

    pauli_strings, coeffs = zip(*sorted(hamiltonian.to_list(), key = lambda x: np.abs(np.real(x[1])), reverse = reverse))
    observables = []
    for i in indices:
        if i == 0:
            observables.append(hamiltonian)
        else:
            observables.append(SparsePauliOp.from_list(zip(pauli_strings[:-i],coeffs[:-i])))
    return observables

def total_spin_operator(num_qubits: int) -> SparsePauliOp:
    """
    Calculate total spin operator sum_{j=0}^{n-1} S_j^z. Can be used
    to verify symmetry sector of a Heisenberg Hamiltonian. Returns 
    operator in SparsePauliOp format.

    :param num_qubits: number of qubits in system
    """
    return SparsePauliOp.from_list([(i * 'I' + 'Z' + 'I' * (num_qubits - i - 1),1) for i in range(num_qubits)])