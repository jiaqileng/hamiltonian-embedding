import numpy as np
from utils import *
from resource_estimate_utils import *
from os.path import join
from time import time

from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate

from braket.devices import LocalSimulator

def get_real_space_H(N, a, b):
    p_sq = np.zeros((N,N))
    x_sq = np.zeros((N,N))
    x = np.zeros((N,N))

    # \hat{p}^2
    for j in range(N-1):
        p_sq[j,j] = 0.5 * (2 * j + 1)
    for j in range(N-2):
        p_sq[j,j+2] = -0.5 * np.sqrt((j+1) * (j+2))
        p_sq[j+2,j] = -0.5 * np.sqrt((j+1) * (j+2))
    
    # \hat{x}^2
    for j in range(N-1):
        x_sq[j,j] = 0.5 * (2 * j + 1)
    for j in range(N-2):
        x_sq[j,j+2] = 0.5 * np.sqrt((j+1) * (j+2))
        x_sq[j+2,j] = 0.5 * np.sqrt((j+1) * (j+2))
    
    # \hat{x}
    for j in range(N-1):
        x[j,j+1] = np.sqrt((j+1)/np.sqrt(2))
        x[j+1,j] = np.sqrt((j+1)/np.sqrt(2))

    H = 0.5 * p_sq + 0.5 * a * x_sq + b * x
    return H

def get_unary_real_space_H_ebd(N, a, b):
    n = num_qubits_per_dim(N, encoding="unary")

    J = np.zeros((n,n))
    for i in range(n-1):
        J[i, i+1] = np.sqrt((i + 1) * (i + 2))
    p_sq = sum_delta_n(n, np.ones(n)) - 0.5 * sum_J_xx(n, J)
    x_sq = sum_delta_n(n, np.ones(n)) + 0.5 * sum_J_xx(n, J)

    h = np.array([np.sqrt((j+1) / 2) for j in range(n)])
    x = sum_h_x(n, h)

    H_ebd = 0.5 * p_sq + 0.5 * a * x_sq + b * x
    return H_ebd

def get_unary_real_space_circuit(N, r, a, b, trotter_method):
    assert r > 0

    n = num_qubits_per_dim(N, encoding="unary")

    t = 1
    dt = t / r
    circuit = Circuit()

    if trotter_method == "first_order":
        for _ in range(r):
            # b * \hat{x}
            for j in range(n):
                circuit.rx(j, 2 * b * dt * np.sqrt((j+1) / 2))
            # 0.5 * a * \hat{x}^2 and 0.5 * \hat{p}
            for j in range(n-1):
                circuit.xx(j, j+1, 0.5 * dt * np.sqrt((j+1) * (j+2)) * (a - 1))
            for j in range(n):
                circuit.phaseshift(j, -0.5 * (1 + a) * dt)

    elif trotter_method == "second_order":
        for _ in range(r):
            for j in range(n):
                circuit.phaseshift(j, -0.25 * (1 + a) * dt)
            # b * \hat{x}
            for j in range(n):
                circuit.rx(j, 2 * b * dt * np.sqrt((j+1) / 2))
            # 0.5 * a * \hat{x}^2 and 0.5 * \hat{p}
            for j in range(n-1):
                circuit.xx(j, j+1, 0.5 * dt * np.sqrt((j+1) * (j+2)) * (a - 1))

            for j in range(n):
                circuit.phaseshift(j, -0.25 * (1 + a) * dt)

    elif trotter_method == "randomized_first_order":
        np.random.seed(int(t * r))
        for _ in range(r):
            if np.random.rand() < 0.5:
                for j in range(n):
                    circuit.phaseshift(j, -0.5 * (1 + a) * dt)
                # b * \hat{x}
                for j in range(n):
                    circuit.rx(j, 2 * b * dt * np.sqrt((j+1) / 2))
                # 0.5 * a * \hat{x}^2 and 0.5 * \hat{p}
                for j in range(n-1):
                    circuit.xx(j, j+1, 0.5 * dt * np.sqrt((j+1) * (j+2)) * (a - 1))
            else:
                # b * \hat{x}
                for j in range(n):
                    circuit.rx(j, 2 * b * dt * np.sqrt((j+1) / 2))
                # 0.5 * a * \hat{x}^2 and 0.5 * \hat{p}
                for j in range(n-1):
                    circuit.xx(j, j+1, 0.5 * dt * np.sqrt((j+1) * (j+2)) * (a - 1))
                for j in range(n):
                    circuit.phaseshift(j, -0.5 * (1 + a) * dt)

    else:
        raise ValueError(f"{trotter_method} not supported")

    return circuit


def get_binary_resource_estimate(N, error_tol, a, b, trotter_method):
    
    print(f"N = {N}", flush=True)

    H = get_real_space_H(N, a, b)
    H_padded = np.pad(H, (0, 2 ** int(np.ceil(np.log2(N))) - N))

    pauli_op = SparsePauliOp.from_operator(H_padded)

    # Compute number of gates per Trotter step
    if trotter_method == "first_order" or trotter_method == "randomized_first_order":
        circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
    elif trotter_method == "second_order":
        circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
    else:
        raise ValueError(f"{trotter_method} not supported")
    
    compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry'], optimization_level=3)

    ops = compiled_circuit.count_ops()

    # Gates per Trotter step
    num_single_qubit_gates, num_two_qubit_gates = get_gate_counts(ops)

    assert num_single_qubit_gates + num_two_qubit_gates == sum(ops.values())
    assert num_two_qubit_gates == compiled_circuit.num_nonlocal_gates()

    print(f"N = {N}, num two qubit gates:", num_two_qubit_gates)

    # Estimate number of Trotter steps required
    r_min, r_max = 1, 10
    while std_bin_trotter_error(H_padded, pauli_op, r_max, trotter_method) > error_tol:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if std_bin_trotter_error(H_padded, pauli_op, r, trotter_method) > error_tol:
            r_min = r
        else:
            r_max = r
    
    print(f"Finished N={N}, num two qubit gates={num_two_qubit_gates}, trotter steps={r_max}", flush=True)
    return num_two_qubit_gates, r_max

if __name__ == "__main__":

    DATA_DIR = "resource_data"
    TASK_DIR = "real_space"

    CURR_DIR = DATA_DIR
    check_and_make_dir(CURR_DIR)
    CURR_DIR = join(CURR_DIR, TASK_DIR)
    check_and_make_dir(CURR_DIR)
    
    print("Resource estimation for real-space simulation.")

    num_jobs = 16
    print("Number of jobs:", num_jobs)

    dimension = 1
    error_tol = 1e-2
    # trotter_method = "first_order"
    # trotter_method = "second_order"
    trotter_method = "randomized_first_order"

    a, b = 1, -1/2
    T = 1
    print(f"Error tolerance: {error_tol : 0.2f}.")
    print(f"Method: {trotter_method}")

    N_vals_binary = np.arange(3, 64)
    binary_trotter_steps = np.zeros(len(N_vals_binary))
    binary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary))

    N_vals_unary = np.arange(3, 17)
    unary_trotter_steps = np.zeros(len(N_vals_unary), dtype=int)
    unary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary))

    N_vals_one_hot = np.arange(3, 17)
    one_hot_trotter_steps = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot))

    print("Running resource estimation for standard binary encoding")
    for i, N in enumerate(N_vals_binary):

        binary_two_qubit_gate_count_per_trotter_step[i], binary_trotter_steps[i] = get_binary_resource_estimate(N, error_tol, a, b, trotter_method)

        np.savez(join(CURR_DIR, f"std_binary_{trotter_method}.npz"),
                N_vals_binary=N_vals_binary[:i+1],
                binary_trotter_steps=binary_trotter_steps[:i+1],
                binary_two_qubit_gate_count_per_trotter_step=binary_two_qubit_gate_count_per_trotter_step[:i+1])


    # Unary encoding
    encoding = "unary"
    print(f"Running resource estimation for {encoding} encoding", flush=True)
    device = LocalSimulator()
    num_samples = 100

    for i, N in enumerate(N_vals_one_hot):
        start_time = time()

        print(f"N = {N} for {encoding}", flush=True)

        H = get_real_space_H(N, a, b)
        n = num_qubits_per_dim(N, encoding)
        codewords = get_codewords_1d(n, encoding, periodic=False)

        tol = 0.5

        unary_two_qubit_gate_count_per_trotter_step[i] = 2 * (n-1)

        H_ebd = get_unary_real_space_H_ebd(N, a, b)
        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        if i > 0:
            r_min = unary_trotter_steps[i-1]
            r_max = unary_trotter_steps[i-1] * 2
        while estimate_trotter_error(N, H_ebd, get_unary_real_space_circuit(N, r_max, a, b, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
            r_max *= 2

        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error(N, H_ebd, get_unary_real_space_circuit(N, r, a, b, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
                r_min = r
            else:
                r_max = r

        unary_trotter_steps[i] = r_max

        # Save data
        np.savez(join(CURR_DIR, f"unary_{trotter_method}.npz"),
                 N_vals_unary=N_vals_unary[:i+1],
                 unary_trotter_steps=unary_trotter_steps[:i+1],
                 unary_two_qubit_gate_count_per_trotter_step=unary_two_qubit_gate_count_per_trotter_step[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)
    
    # One hot encoding
    encoding = "one-hot"
    print(f"Running resource estimation for {encoding} encoding", flush=True)
    device = LocalSimulator()
    num_samples = 100

    for i, N in enumerate(N_vals_one_hot):
        start_time = time()

        print(f"N = {N} for {encoding}", flush=True)

        H = get_real_space_H(N, a, b)
        n = num_qubits_per_dim(N, encoding)
        codewords = get_codewords_1d(n, encoding, periodic=False)

        tol = 0.5

        nonzero_entries = 0
        for j in range(N):
            for k in range(j):
                if H[j,k] != 0:
                    nonzero_entries += 1

        if trotter_method == "first_order" or trotter_method == "randomized_first_order":
            one_hot_two_qubit_gate_count_per_trotter_step[i] = 2 * nonzero_entries
        elif trotter_method == "second_order":
            one_hot_two_qubit_gate_count_per_trotter_step[i] = 4 * nonzero_entries
        else:
            raise ValueError(f"{trotter_method} not supported")
    
        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        if i > 0:
            r_min = one_hot_trotter_steps[i-1]
            r_max = one_hot_trotter_steps[i-1] * 2
        while estimate_trotter_error_one_hot_1d(N, H, r_max, dimension, encoding, codewords, device, num_samples, num_jobs, trotter_method) > error_tol:
            r_max *= 2

        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error_one_hot_1d(N, H, r, dimension, encoding, codewords, device, num_samples, num_jobs, trotter_method) > error_tol:
                r_min = r
            else:
                r_max = r

        one_hot_trotter_steps[i] = r_max

        # Save data
        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}.npz"),
                 N_vals_one_hot=N_vals_one_hot[:i+1],
                 one_hot_trotter_steps=one_hot_trotter_steps[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step=one_hot_two_qubit_gate_count_per_trotter_step[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)