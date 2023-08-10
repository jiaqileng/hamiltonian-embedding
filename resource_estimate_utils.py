import numpy as np

from scipy.sparse.linalg import expm_multiply, expm, norm
from utils import *
from braket.circuits import Circuit

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator

from joblib import Parallel, delayed
import networkx as nx

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

def get_trotter_state_vector(N, amplitudes_list, trotterized_circuit, dimension, bitstrings, encoding, device):

    # Initial state
    circ = get_initial_product_state_circuit(amplitudes_list, dimension, N, encoding)
    circ.add_circuit(trotterized_circuit)

    circ.amplitude(state=bitstrings)

    task = device.run(circ)
    amplitudes = task.result().values[0]

    state_vector = np.zeros(N ** dimension, dtype=np.complex64)

    for i in range(N ** dimension):
        state_vector[i] = amplitudes[bitstrings[i]]
    
    return state_vector

def estimate_trotter_error_one_sample(N, H, trotterized_circuit, dimension, encoding, codewords, device):
    n = num_qubits_per_dim(N, encoding)
    bitstrings = get_bitstrings(N, dimension, encoding)
    assert len(codewords) == N ** dimension

    amplitudes_list = []
    for _ in range(dimension):
        initial_state = np.random.randn(N) + 1j * np.random.randn(N)
        initial_state /= np.linalg.norm(initial_state, ord=2)
        amplitudes_list.append(initial_state)

    # Trotter
    psi_trotter = get_trotter_state_vector(N, amplitudes_list, trotterized_circuit, dimension, bitstrings, encoding, device)

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

def estimate_trotter_error(N, H_ebd, circuit, dimension, encoding, codewords, device, num_samples, num_jobs):

    res = Parallel(n_jobs=num_jobs)(delayed(estimate_trotter_error_one_sample)(N, H_ebd, circuit, dimension, encoding, codewords, device) for _ in range(num_samples))
  
    return max(res)

def estimate_trotter_error_one_hot_1d(N, H, r, dimension, encoding, codewords, device, num_samples, num_jobs, trotter_method):
    assert encoding == "one-hot"
    assert np.all(np.isreal(H)), "H should be real"
    assert (N,N) == H.shape
    assert dimension == 1
    
    H_terms, graph = get_H_terms_one_hot(N, H)

    line_graph = nx.line_graph(graph)
    coloring = nx.coloring.greedy_color(line_graph, strategy="independent_set")

    coloring_grouped = {}
    for edge in coloring.keys():
        if coloring[edge] in coloring_grouped:
            coloring_grouped[coloring[edge]].append(edge)
        else:
            coloring_grouped[coloring[edge]] = [edge]

    num_colors = len(coloring_grouped.keys())

    t = 1
    dt = t / r

    circuit = Circuit()

    if trotter_method == "first_order":
        for _ in range(r):
            for color in np.arange(0, num_colors):
                edge_list = coloring_grouped[color]
                for i,j in edge_list:
                    circuit.xx(i, j, dt * H[i,j])
                    circuit.yy(i, j, dt * H[i,j])
            for i in range(N):
                circuit.phaseshift(i, - dt * H[i,i])

    elif trotter_method == "second_order":
        for _ in range(r):
            for color in np.arange(0, num_colors):
                edge_list = coloring_grouped[color]
                for i,j in edge_list:
                    circuit.xx(i, j, dt * H[i,j] / 2)
                    circuit.yy(i, j, dt * H[i,j] / 2)

            for i in range(N):
                circuit.phaseshift(i, - dt * H[i,i])

            for color in np.arange(0, num_colors)[::-1]:
                edge_list = coloring_grouped[color]
                for i,j in edge_list:
                    circuit.yy(i, j, dt * H[i,j] / 2)
                    circuit.xx(i, j, dt * H[i,j] / 2)

    elif trotter_method == "randomized_first_order":
        np.random.seed(int(t * r))
        for _ in range(r):
            
            if np.random.rand() < 0.5:
                for color in np.arange(0, num_colors):
                    edge_list = coloring_grouped[color]
                    
                    for i,j in edge_list:
                        circuit.xx(i, j, dt * H[i,j])
                        circuit.yy(i, j, dt * H[i,j])
                for i in range(N):
                    circuit.phaseshift(i, - dt * H[i,i])
            else:
                for i in range(N):
                    circuit.phaseshift(i, - dt * H[i,i])
                for color in np.arange(0, num_colors)[::-1]:
                    edge_list = coloring_grouped[color]
                    
                    for i,j in edge_list:
                        circuit.yy(i, j, dt * H[i,j])
                        circuit.xx(i, j, dt * H[i,j])
    else:
        raise ValueError(f"{trotter_method} not supported")
    
    H_ebd = sum(H_terms)
    res = Parallel(n_jobs=num_jobs)(delayed(estimate_trotter_error_one_sample)(N, H_ebd, circuit, dimension, encoding, codewords, device) for _ in range(num_samples))
  
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

def get_trotter_number(pauli_op, t, epsilon, trotter_method):
    '''Uses analytical bound to compute the Trotter number to reach error threshold epsilon'''
    H = pauli_op.group_commuting()
    L = len(H)
    error = 0
    if trotter_method == "first_order":
        for j in range(L):
            for k in np.arange(j + 1, L):
                error += np.linalg.norm(commutator(H[k], H[j]).simplify().coeffs, ord=1)
        
        return int((t ** 2 / (2 * epsilon)) * error)
    elif trotter_method == "second_order":    
        for j in range(L):
            for k in np.arange(j+1, L):
                for l in np.arange(j+1, L):
                    error += np.linalg.norm(commutator(H[l], commutator(H[k], H[j])).simplify().coeffs, ord=1)

                error += 0.5 * np.linalg.norm(commutator(H[j], commutator(H[j], H[k])).simplify().coeffs, ord=1)

        return int(np.sqrt((t ** 3 / (12 * epsilon)) * error))
    else:
        raise ValueError(f"{trotter_method} not supported")

    
def std_bin_trotter_fidelity(H, r):
    pauli_op = SparsePauliOp.from_operator(H / r)  

    circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op))

    U_one_trotter_layer = Operator(circuit).data
    U_trotter = np.linalg.matrix_power(U_one_trotter_layer, r)
    U_exact = expm(-1j * H)
    return np.abs((U_exact @ np.conj(U_trotter.T)).trace()) / H.shape[0]

def std_bin_trotter_error(H, pauli_op, r, trotter_method):
    '''Computes the Trotter error exactly using the full unitary matrix'''

    if trotter_method == "first_order":
        circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op / r))

        U_one_trotter_layer = Operator(circuit).data
        U_trotter = np.linalg.matrix_power(U_one_trotter_layer, r)
        U_exact = expm(-1j * H)
        return np.linalg.norm(U_exact - U_trotter, ord=2)
    
    elif trotter_method == "second_order":
        circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op / r))

        U_one_trotter_layer = Operator(circuit).data
        U_trotter = np.linalg.matrix_power(U_one_trotter_layer, r)
        U_exact = expm(-1j * H)
        return np.linalg.norm(U_exact - U_trotter, ord=2)
    
    elif trotter_method == "randomized_first_order":
        circuit = QuantumCircuit(np.log2(A.shape[0]))

        single_step = (pauli_op / r).group_commuting()
        fwd_circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(single_step))
        bkwd_circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(single_step[::-1]))
        for _ in range(r):
            if np.random.rand() < 0.5:
                circuit &= fwd_circuit
            else:
                circuit &= bkwd_circuit

        U_trotter = Operator(circuit).data
        U_exact = expm(-1j * H)
        return np.linalg.norm(U_exact - U_trotter, ord=2)
    
    else:
        raise ValueError(f"{trotter_method} not supported")
    
def std_bin_trotter_error_one_sample(H, pauli_op, r, trotter_method):
    pauli_op_grouped = pauli_op.group_commuting()
    psi = np.random.randn(H.shape[0]) + 1j * np.random.randn(H.shape[0])
    psi /= np.linalg.norm(psi)

    psi_no_trotter = expm_multiply(-1j * H, psi)
    psi_trotter = psi

    if trotter_method == "first_order":
        for _ in range(r):
            for j in range(len(pauli_op_grouped)):
                H_j = pauli_op_grouped[j]
                psi_trotter = expm_multiply(-1j * H_j.to_matrix(sparse=True) / r, psi_trotter)

    elif trotter_method == "second_order":

        for _ in range(r):
            for j in range(len(pauli_op_grouped)):
                H_j = pauli_op_grouped[j]
                psi_trotter = expm_multiply(-1j * H_j.to_matrix(sparse=True) / (2 * r), psi_trotter)
            for j in range(len(pauli_op_grouped))[::-1]:
                H_j = pauli_op_grouped[j]
                psi_trotter = expm_multiply(-1j * H_j.to_matrix(sparse=True) / (2 * r), psi_trotter)
    else:
        raise ValueError(f"{trotter_method} not supported")
        
    error = np.linalg.norm(psi_no_trotter - psi_trotter, ord=2)
    return error

def std_bin_trotter_error_sampling(H, pauli_op, r, trotter_method, num_samples, num_jobs):
    '''Uses sampling to compute the Trotter error'''

    res = Parallel(n_jobs=num_jobs)(delayed(std_bin_trotter_error_one_sample)(H, pauli_op, r, trotter_method) for _ in range(num_samples))
  
    return max(res)

def subspace_fidelity(n, H_ebd, H, codewords):
    U_exact = expm(-1j * H)
    U_subspace = expm(-1j * H_ebd)[codewords][:,codewords]
    return np.abs((U_exact @ np.conj(U_subspace.T)).trace()) / U_exact.shape[0]

def subspace_trotter_error(A, B, T, r, n, encoding):

    U_exact = expm(-1j * csc_matrix(A+B) * T)

    U_one_trotter_layer = expm(-1j * csc_matrix(A) * (T / r)) @ expm(-1j * csc_matrix(B) * (T / r))

    U_trotter = U_one_trotter_layer ** r
    return subspace_error(U_exact, U_trotter, n, encoding)

def subspace_error(U1, U2, n, encoding):
    
    diff = U1 - U2
    codewords = get_codewords_1d(n, encoding=encoding, periodic=False)
    diff_subspace = diff[codewords][:,codewords]

    if encoding == "one-hot":
        assert diff_subspace.shape == (n, n)
    
    return norm(diff_subspace, ord=2)

def get_H_terms_one_hot(n, H):
    '''Returns a list of (possibly noncommuting) Hamiltonian terms for one-hot encoding'''

    graph = nx.Graph()
    for i in range(n):
        for j in range(i):
            if H[i,j] != 0:
                graph.add_edge(i,j)

    line_graph = nx.line_graph(graph)
    coloring = nx.coloring.greedy_color(line_graph, strategy="independent_set")

    coloring_grouped = {}
    for edge in coloring.keys():
        if coloring[edge] in coloring_grouped:
            coloring_grouped[coloring[edge]].append(edge)
        else:
            coloring_grouped[coloring[edge]] = [edge]

    num_colors = len(coloring_grouped.keys())

    H_terms = []
    for color in np.arange(0, num_colors):
        edge_list = coloring_grouped[color]
        
        A = np.zeros((n, n))
        for i,j in edge_list:
            A[i,j] = H[i,j]
        
        H_terms.append((sum_J_xx(n, A) + sum_J_yy(n, A)) / 2)
    
    H_terms.append(sum_delta_n(n, np.diag(H)))

    return H_terms, graph