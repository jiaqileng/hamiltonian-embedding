import numpy as np

from utils import *
from resource_estimate_utils import *
from os.path import join
from time import time
import multiprocessing

from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate

from braket.devices import LocalSimulator

def get_binary_resource_estimate(N, dimension, error_tol, order=1):
    print(f"N = {N}", flush=True)

    # Compute the optimal gamma
    L = get_laplacian_lattice(N, d=dimension)
    marked_vertex_1 = np.zeros(N)
    marked_vertex_1[N-1] = 1
    marked_vertex_2 = np.zeros(N)
    marked_vertex_2[0] = 1

    marked_vertex = np.kron(marked_vertex_1, marked_vertex_2)
    H_oracle = - csc_matrix(np.outer(marked_vertex, marked_vertex))
    
    # Sign is flipped here; this function minimizes the difference between the two largest eigenvalues of gamma * L + H_oracle
    gamma = scipy_get_optimal_gamma(L, -H_oracle, 0.3)

    H_spatial_search = (-gamma * L + H_oracle).toarray()
    H_spatial_search_padded = np.pad(H_spatial_search, (0, 2 ** int(np.ceil(np.log2(N ** 2))) - N ** 2))
    pauli_op = SparsePauliOp.from_operator(H_spatial_search_padded)
    
    if order == 1:
        circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
    elif order == 2:
        circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
    else:
        raise ValueError("Order must be 1 or 2")

    compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry'], optimization_level=3)

    ops = compiled_circuit.count_ops()

    num_single_qubit_gates, num_two_qubit_gates = get_gate_counts(ops)

    assert num_single_qubit_gates + num_two_qubit_gates == sum(ops.values())
    assert num_two_qubit_gates == compiled_circuit.num_nonlocal_gates()


    # Estimate number of Trotter steps required
    r_min, r_max = 1, 10
    while std_bin_trotter_error(H_spatial_search_padded, pauli_op, r_max) > error_tol:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if std_bin_trotter_error(H_spatial_search_padded, pauli_op, r) > error_tol:
            r_min = r
        else:
            r_max = r
    

    return num_two_qubit_gates, r_max

def get_H_spatial_search(n, lamb, gamma, encoding):
    if encoding == "unary" or encoding == "antiferromagnetic":
        if encoding == "antiferromagnetic":
            assert n % 2 == 1, "only works for odd number of qubits"
        J = np.zeros((2 * n, 2 * n))
        h = np.zeros(2 * n)

        for i in range(2):
            h[i * n] = 1
            if encoding == "unary":
                h[(i + 1) * n - 1] = -1
            else:
                h[(i + 1) * n - 1] = (-1) ** (n)
            
            for j in np.arange(i * n, (i + 1) * n - 1):
                if encoding == "unary":
                    J[j, j + 1] = -1
                else:
                    J[j, j + 1] = 1

        H_pen = lamb * (sum_J_zz(2 * n, J) + sum_h_z(2 * n, h))

        # Create the oracle Hamiltonian
        J = np.zeros((2 * n, 2 * n))
        J[n-1, n] = 1/4

        h = np.zeros(2 * n)
        h[n-1] = 1/4
        h[n] = -1/4

        H_oracle = sum_h_z(2 * n, h) + sum_J_zz(2 * n, J)

        # Correction term for Laplacian
        h_correction = np.zeros(2 * n)
        for i in range(2):
            h_correction[i * n] = 1/2
            if encoding == "unary":
                h_correction[(i + 1) * n - 1] = -1/2
            else:
                h_correction[(i + 1) * n - 1] = -(-1) ** (n+1) / 2
        H_correction = - gamma * sum_h_z(2 * n, h_correction)

        return H_pen + H_oracle + H_correction, - gamma * sum_x(2 * n)

def get_spatial_search_one_trotter_step_circuit(n, lamb, gamma, T, r, encoding, order=1):
    assert encoding == "unary" or encoding == "antiferromagnetic"
    one_step_circ = Circuit()
    
    if order == 1:
        # x rotations
        for i in range(2 * n):
            one_step_circ.rx(i, - 2 * gamma * T / r)

        # penalty term
        for i in range(2):
            one_step_circ.rz(i * n, 2 * lamb * T / r)
            if encoding == "unary":
                one_step_circ.rz((i + 1) * n - 1, - 2 * lamb * T / r)
            else:
                one_step_circ.rz((i + 1) * n - 1, (-1) ** (n) * 2 * lamb * T / r)
            
            for j in np.arange(i * n, (i + 1) * n - 1):
                if encoding == "unary":
                    one_step_circ.zz(j, j+1, -2 * lamb * T / r)
                else:
                    one_step_circ.zz(j, j+1, 2 * lamb * T / r)

        # oracle term
        one_step_circ.zz(n-1, n, (T / r) / 2)
        one_step_circ.rz(n-1, (T / r) / 2)
        one_step_circ.rz(n, -(T / r) / 2)

        # laplacian correction term
        for i in range(2):
            one_step_circ.rz(i * n, -gamma * T / r)
            if encoding == "unary":
                one_step_circ.rz((i + 1) * n - 1, gamma * T / r)
            else:
                one_step_circ.rz((i + 1) * n - 1, (-1) ** (n+1) * gamma * T / r)
    
    elif order == 2:
        # x rotations
        for i in range(2 * n):
            one_step_circ.rx(i, - gamma * T / r)

        # penalty term
        for i in range(2):
            one_step_circ.rz(i * n, 2 * lamb * T / r)
            if encoding == "unary":
                one_step_circ.rz((i + 1) * n - 1, - 2 * lamb * T / r)
            else:
                one_step_circ.rz((i + 1) * n - 1, (-1) ** (n) * 2 * lamb * T / r)
            
            for j in np.arange(i * n, (i + 1) * n - 1):
                if encoding == "unary":
                    one_step_circ.zz(j, j+1, -2 * lamb * T / r)
                else:
                    one_step_circ.zz(j, j+1, 2 * lamb * T / r)

        # oracle term
        one_step_circ.zz(n-1, n, (T / r) / 2)
        one_step_circ.rz(n-1, (T / r) / 2)
        one_step_circ.rz(n, -(T / r) / 2)

        # laplacian correction term
        for i in range(2):
            one_step_circ.rz(i * n, -gamma * T / r)
            if encoding == "unary":
                one_step_circ.rz((i + 1) * n - 1, gamma * T / r)
            else:
                one_step_circ.rz((i + 1) * n - 1, (-1) ** (n+1) * gamma * T / r)
            
        # x rotations
        for i in range(2 * n):
            one_step_circ.rx(i, - gamma * T / r)
    else:
        raise ValueError("Trotter order must be 1 or 2")

    return one_step_circ

if __name__ == "__main__":
    
    print("Resource estimation for spatial search.")
    num_jobs = 16
    print("Number of jobs:", num_jobs)

    dimension = 2
    error_tol = 1e-2
    trotter_order = 2
    T = 1
    print(f"Error tolerance: {error_tol : 0.2f}.")
    print(f"Trotter order: {trotter_order}")

    N_vals_binary = np.arange(4, 21)
    N_vals_unary = np.arange(4, 11)

    unary_trotter_steps = np.zeros(len(N_vals_unary), dtype=int)
    unary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary))

    print("Running resource estimation for standard binary encoding", flush=True)
    res = Parallel(n_jobs=num_jobs)(delayed(get_binary_resource_estimate)(N, dimension, error_tol) for N in N_vals_binary)
    binary_two_qubit_gate_count_per_trotter_step, binary_trotter_steps = zip(*res)
    binary_two_qubit_gate_count_per_trotter_step = np.array(binary_two_qubit_gate_count_per_trotter_step)
    binary_trotter_steps = np.array(binary_trotter_steps, dtype=int)

    print(binary_two_qubit_gate_count_per_trotter_step)
    print(binary_trotter_steps)
    np.savez(join("resource_data", "resource_estimation_spatial_search_binary.npz"),
                N_vals_binary=N_vals_binary,
                binary_trotter_steps=binary_trotter_steps,
                binary_two_qubit_gate_count_per_trotter_step=binary_two_qubit_gate_count_per_trotter_step)


    # Antiferromagnetic encoding
    encoding = "unary"
    print(f"Running resource estimation for {encoding} encoding", flush=True)
    device = LocalSimulator()
    num_samples = 100

    for i, N in enumerate(N_vals_unary):

        print(f"Running N = {N}", flush=True)
        start_time = time()
        n = num_qubits_per_dim(N, encoding)

        # Compute the optimal gamma
        L = get_laplacian_lattice(N, d=dimension)
        marked_vertex_1 = np.zeros(N)
        marked_vertex_1[N-1] = 1
        marked_vertex_2 = np.zeros(N)
        marked_vertex_2[0] = 1

        marked_vertex = np.kron(marked_vertex_1, marked_vertex_2)
        H_oracle = - csc_matrix(np.outer(marked_vertex, marked_vertex))
        # Sign is flipped here; this function minimizes the difference between the two largest eigenvalues of gamma * L + H_oracle
        gamma = scipy_get_optimal_gamma(L, -H_oracle, 0.3)
        codewords = get_codewords_2d(n, encoding, periodic=False)

        lamb = dimension * n
        
        unary_two_qubit_gate_count_per_trotter_step[i] = 2 * (n - 1) + 1

        A, B = get_H_spatial_search(n, lamb, gamma, encoding)

        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        while estimate_trotter_error(N, A + B, get_spatial_search_one_trotter_step_circuit(n, lamb, gamma, T, r_max, encoding, order=trotter_order), r_max, dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
            r_max *= 2
        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error(N, A + B, get_spatial_search_one_trotter_step_circuit(n, lamb, gamma, T, r, encoding, order=trotter_order), r, dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
                r_min = r
            else:
                r_max = r
        
        unary_trotter_steps[i] = r

        np.savez(join("resource_data", f"resource_estimation_spatial_search_{encoding}.npz"),
                    N_vals_unary=N_vals_unary[:i+1],
                    unary_trotter_steps=unary_trotter_steps[:i+1],
                    unary_two_qubit_gate_count_per_trotter_step=unary_two_qubit_gate_count_per_trotter_step[:i+1])
        
        
        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)
    