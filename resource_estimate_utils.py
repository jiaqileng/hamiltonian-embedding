import numpy as np

from scipy.sparse.linalg import expm_multiply, expm, norm
from scipy.sparse import diags, eye
from utils import *
from braket.circuits import Circuit

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator

from joblib import Parallel, delayed

def state_prep(dimension, N, amplitudes_list, encoding):
    # amplitudes_list: list of length dimension, where each element is a
    circuit = Circuit()

    for i in range(dimension):
        amplitudes = amplitudes_list[i]

        if encoding == "unary" or encoding == "antiferromagnetic":

            n = N - 1
            assert len(amplitudes) == N

            circuit.ry(i * n, 2 * np.arccos(amplitudes[0]))

            for k in np.arange(0, n-1):
                # Y rotation controlled on previous qubit
                if np.linalg.norm(amplitudes[k+1:], ord=2) > 1e-8:
                    # Basis change to Z
                    circuit.rx(i * n + k+1, np.pi/2)
                    # Controlled RZ (note the use of two cphaseshift to implement CRZ, braket uses big endian)
                    a = amplitudes[k+1] / np.linalg.norm(amplitudes[k+1:], ord=2)
                    circuit.cphaseshift(i * n + k, i * n + k+1, np.arccos(a))
                    circuit.cphaseshift10(i * n + k, i * n + k + 1, - np.arccos(a))
                    circuit.rx(i * n + k+1, -np.pi/2)

            # Just map from unary to antiferromagnetic encoding
            if encoding == "antiferromagnetic":
                for k in range(n):
                    if k % 2 == 1:
                        circuit.x(i * n + k)
            

        elif encoding == "one-hot":
            n = N
            # Start from 1000...0
            circuit.x(i * n + 0)
            # Y rotation
            circuit.ry(i * n + 1, 2 * np.arccos(amplitudes[0]))
            circuit.cnot(1, 0)
            for k in np.arange(1, N-1):
                if np.linalg.norm(amplitudes[k:], ord=2) > 1e-8:
                    # Y rotation controlled on previous qubit
                    # Basis change to Z
                    circuit.rx(i * n + k + 1, np.pi/2)
                    # Controlled Z
                    a = amplitudes[k] / np.linalg.norm(amplitudes[k:], ord=2)
                    circuit.cphaseshift(i * n + k, i * n + k + 1, np.arccos(a))
                    circuit.cphaseshift10(i * n + k, i * n + k + 1, - np.arccos(a))
                    circuit.rx(i * n + k + 1, -np.pi/2)

                    # CNOT
                    circuit.cnot(i * n + k + 1, i * n + k)
        else:
            raise ValueError("Encoding not supported")
    
    return circuit

def get_initial_product_state_circuit(amplitudes_list, dimension, N, encoding):
    assert len(amplitudes_list) == dimension, "Amplitudes list does not match dimension!"
    
    n = num_qubits_per_dim(N, encoding)


    amplitudes_abs_val_list = []
    for i in range(dimension):
        amplitudes_abs_val_list.append(np.abs(amplitudes_list[i]))

    circuit = state_prep(dimension, N, amplitudes_abs_val_list, encoding)

    # get the local phase
    for i in range(dimension):
        amplitudes = amplitudes_list[i]
        for j in range(N):
            
            theta = np.angle(amplitudes[j])
            if encoding == "unary":
                if j == 0:
                    circuit.cphaseshift00(i * n, (i + 1) * n - 1, theta)
                elif j == N - 1:
                    circuit.cphaseshift(i * n, (i + 1) * n - 1, theta)
                else:
                    circuit.cphaseshift10(i * n + j - 1, i * n + j, theta)
            elif encoding == "antiferromagnetic":
                if j == 0:
                    if N % 2 == 0:
                        circuit.cphaseshift00(i * n, (i + 1) * n - 1, theta)
                    else:
                        circuit.cphaseshift01(i * n, (i + 1) * n - 1, theta)
                elif j == N - 1:
                    if N % 2 == 0:
                        circuit.cphaseshift(i * n, (i + 1) * n - 1, theta)
                    else:
                        circuit.cphaseshift10(i * n, (i + 1) * n - 1, theta)
                else:
                    if j % 2 == 1:
                        circuit.cphaseshift10(i * n + j - 1, i * n + j, theta)
                    else:
                        circuit.cphaseshift01(i * n + j - 1, i * n + j, theta)
            elif encoding == "one-hot":
                circuit.rz(i * N + j, theta)
        
    return circuit

def get_trotter_state_vector(N, amplitudes_list, circ_one_trotter_step, r, dimension, bitstrings, encoding, device):

    # Initial state
    circ = get_initial_product_state_circuit(amplitudes_list, dimension, N, encoding)

    # Trotter
    for _ in range(r):
        circ.add_circuit(circ_one_trotter_step)

    circ.amplitude(state=bitstrings)

    task = device.run(circ)
    amplitudes = task.result().values[0]

    state_vector = np.zeros(N ** dimension, dtype=np.complex64)

    for i in range(N ** dimension):
        state_vector[i] = amplitudes[bitstrings[i]]
    
    return state_vector

def estimate_trotter_error_one_sample(N, H, circ_one_trotter_step, r, dimension, encoding, codewords, device):
    n = num_qubits_per_dim(N, encoding)
    bitstrings = get_bitstrings(N, dimension, encoding)
    assert len(codewords) == N ** dimension

    amplitudes_list = []
    for _ in range(dimension):
        initial_state = np.random.randn(N) + 1j * np.random.randn(N)
        initial_state /= np.linalg.norm(initial_state, ord=2)
        amplitudes_list.append(initial_state)

    # Trotter
    psi_trotter = get_trotter_state_vector(N, amplitudes_list, circ_one_trotter_step, r, dimension, bitstrings, encoding, device)

    # No Trotter
    circ = get_initial_product_state_circuit(amplitudes_list, dimension, N, encoding)
    circ.amplitude(state=bitstrings)
    task = device.run(circ)
    amplitudes = task.result().values[0]
    amplitudes_state_vector = np.zeros(N ** dimension, dtype=np.complex64)

    for i in range(N ** dimension):
        amplitudes_state_vector[i] = amplitudes[bitstrings[i]]

    psi_0 = np.zeros(2 ** (n * dimension), dtype=np.complex64)
    psi_0[codewords] = amplitudes_state_vector
    psi_no_trotter = expm_multiply(-1j * H, psi_0)[codewords]

    # Estimate Trotter error
    error = np.linalg.norm(psi_trotter - psi_no_trotter, ord=2)
    return error

def estimate_trotter_error(N, H, circ_one_trotter_step, r, dimension, encoding, codewords, device, num_samples, num_jobs):
    
    res = Parallel(n_jobs=num_jobs)(delayed(estimate_trotter_error_one_sample)(N, H, circ_one_trotter_step, r, dimension, encoding, codewords, device) for _ in range(num_samples))
  
    return max(res)


def get_gate_counts(ops):
    num_single_qubit_gates = 0
    num_two_qubit_gates = 0
    single_qubit_gates = ['rx', 'ry', 'rz', 'h']
    two_qubit_gates = ['rxx', 'ryy', 'rzz']

    for gate in single_qubit_gates:
        if gate in ops:
            num_single_qubit_gates += ops[gate]
    
    for gate in two_qubit_gates:
        if gate in ops:
            num_two_qubit_gates += ops[gate]
    
    return num_single_qubit_gates, num_two_qubit_gates

def commutator(A, B):
    return A @ B - B @ A

def trotter_error_bound(pauli_op, T, r):
    group_commuting_op = pauli_op.group_commuting()
    error = 0
    Gamma = len(group_commuting_op)
    for gamma_1 in range(Gamma):
        for gamma_2 in np.arange(gamma_1 + 1, Gamma):
            error += np.linalg.norm(commutator(group_commuting_op[gamma_2], group_commuting_op[gamma_1]).simplify().coeffs, ord=1)
    
    return (T ** 2 / (2 * r)) * error

    
def std_bin_trotter_fidelity(A, r):
    pauli_op = SparsePauliOp.from_operator(A / r)  

    circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op))

    U_one_trotter_layer = Operator(circuit).data
    U_trotter = np.linalg.matrix_power(U_one_trotter_layer, r)
    U_exact = expm(-1j * A)
    return np.abs((U_exact @ np.conj(U_trotter.T)).trace()) / A.shape[0]

def std_bin_trotter_error(A, pauli_op, r):

    circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op / r))

    U_one_trotter_layer = Operator(circuit).data
    U_trotter = np.linalg.matrix_power(U_one_trotter_layer, r)
    U_exact = expm(-1j * A)
    return np.linalg.norm(U_exact - U_trotter, ord=2)

def subspace_fidelity(n, H, adjacency_matrix, encoding):
    U_exact = expm(-1j * adjacency_matrix)
    codewords = get_codewords_1d(n, encoding=encoding, periodic=False)
    U_subspace = expm(-1j * H)[codewords][:,codewords]
    return np.abs((U_exact @ np.conj(U_subspace.T)).trace()) / U_exact.shape[0]

def one_hot_trotter_fidelity(H, circuit, r):
    circuit.remove_final_measurements(inplace=True)
    U_one_trotter_layer = Operator(circuit).data
    U_trotter = np.linalg.matrix_power(U_one_trotter_layer, r)
    U_exact = expm(-1j * H)
    return np.abs((U_exact @ np.conj(U_trotter.T)).trace()) / H.shape[0]

def subspace_trotter_error(A, B, T, r, n, encoding):

    U_exact = expm(-1j * csc_matrix(A+B) * T)

    U_one_trotter_layer = expm(-1j * csc_matrix(A) * (T / r)) @ expm(-1j * csc_matrix(B) * (T / r))

    U_trotter = U_one_trotter_layer ** r
    return subspace_error(U_exact, U_trotter, n, encoding)

# def subspace_trotter_error_estimate(A, B, T, r, n, encoding, num_samples):
    
#     # U_exact = expm(-1j * csc_matrix(A+B) * T)

#     # U_one_trotter_layer = expm(-1j * csc_matrix(A) * (T / r)) @ expm(-1j * csc_matrix(B) * (T / r))

#     # U_trotter = U_one_trotter_layer ** r

#     P_diag = np.zeros(2 ** n)
#     codewords = get_codewords_1d(n, encoding=encoding, periodic=False)
#     P_diag[codewords] = 1
#     P = diags(P_diag)

#     max_val = None
#     for _ in range(num_samples):

#         input_state = P @ (np.random.randn(2 ** n).astype(np.float32) + 1j * np.random.randn(2 ** n).astype(np.float32))
#         input_state /= np.linalg.norm(input_state)
#         exact_state = expm_multiply(-1j * csc_matrix(A+B) * T, input_state)
#         trotter_state = input_state
#         for _ in range(r):
#             trotter_state = expm_multiply(-1j * B * (T / r), trotter_state)
#             trotter_state = expm_multiply(-1j * A * (T / r), trotter_state)

#         error_val = np.linalg.norm(P @ (exact_state - trotter_state))
#         if max_val == None or error_val > max_val:
#             max_val = error_val
            
#     return error_val

# def full_trotter_error(A, B, T, r, n):

#     U_exact = expm(-1j * csc_matrix(A+B) * T)

#     U_one_trotter_layer = expm(-1j * csc_matrix(A) * (T / r)) @ expm(-1j * csc_matrix(B) * (T / r))
#     U_trotter = U_one_trotter_layer ** r
    
#     return norm(U_exact - U_trotter, ord=2)


def subspace_error(U1, U2, n, encoding):
    
    diff = U1 - U2
    codewords = get_codewords_1d(n, encoding=encoding, periodic=False)
    diff_subspace = diff[codewords][:,codewords]

    if encoding == "one-hot":
        assert diff_subspace.shape == (n, n)
    
    return norm(diff_subspace, ord=2)

# def subspace_ip_trotter_error(A, B, T, r, n, encoding):
#     U_exact = expm(-1j * csc_matrix(A+B) * T)
    
#     # Interaction picture with Trotter
#     W = eye(2 ** n)
#     for k in range(r):
#         t_k = k * T / r
#         W = expm(1j * A * t_k) @ expm(-1j * B * (T / r)) @ expm(-1j * A * t_k) @ W

#     U_trotter = expm(-1j * A * T) @ W

#     return subspace_error(U_exact, U_trotter, n, encoding)

# def one_hot_trotter_subspace_fidelity(A, circuit, r):
    
#     U_exact = expm(-1j * A)
#     circuit.draw()
#     circuit.remove_final_measurements(inplace=True)
#     U_one_trotter_layer = Operator(circuit).data
#     U_trotter = np.linalg.matrix_power(U_one_trotter_layer, r)
#     U_subspace = np.zeros_like(U_exact)
#     codewords = get_codewords_1d(n, encoding="one-hot", periodic=False)
#     for j in range(n):
#         for k in range(n):
#             U_subspace[j,k] = U_trotter[codewords[j],codewords[k]]

#     # plt.matshow(U_subspace.real)
#     # plt.matshow(U_exact.real)
#     # plt.show()
#     return np.abs((U_exact @ np.conj(U_subspace.T)).trace()) / A.shape[0]

def get_H_one_hot(n, lamb, A):

    H_Z = sum_h_z(n, lamb * np.ones(n))
    H_XX = sum_J_xx(n, A / 2)

    return H_Z, H_XX

def get_H_pauli_op(lamb, adjacency_matrix):
    n = adjacency_matrix.shape[0]
    H_pen = []
    for j in range(n):
        op = n * ['I']
        op[j] = 'Z'
        H_pen.append(SparsePauliOp(''.join(op), lamb))

    Q = []
    for j in range(n):
        for k in range(j):
            if adjacency_matrix[j,k] != 0:
                op = n * ['I']
                op[j] = 'X'
                op[k] = 'X'
                Q.append(SparsePauliOp(''.join(op), adjacency_matrix[j,k]))

    return (sum(H_pen) + sum(Q)).simplify()

def one_hot_get_trotter_step_circ(N, dimension, T, r, adjacency_matrix, lamb, order):
    circ = Circuit()

    if order == 1:
        # XX
        for i in range(dimension):
            for j in range(N):
                for k in range(j):
                    if adjacency_matrix[j,k] > 0:
                        circ.xx(N * i + j, N * i + k, 2 * T / r)
        # Z
        for i in range(N * dimension):
            circ.rz(i, 2 * lamb / r)
    elif order == 2:
        # Z
        for i in range(N * dimension):
            circ.rz(i, lamb / r)

        # XX
        for i in range(dimension):
            for j in range(N):
                for k in range(j):
                    if adjacency_matrix[j,k] > 0:
                        circ.xx(N * i + j, N * i + k, 2 * T / r)
        # Z
        for i in range(N * dimension):
            circ.rz(i, lamb / r)
    else:
        raise ValueError("Trotter order must be 1 or 2")
    return circ