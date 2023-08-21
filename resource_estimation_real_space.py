import numpy as np
from utils import *
from resource_estimate_utils import *
from os.path import join
from time import time

from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from pytket import OpType
from pytket.passes import RemoveRedundancies, CommuteThroughMultis, SequencePass, FullPeepholeOptimise, auto_rebase_pass
from pytket.extensions.qiskit import qiskit_to_tk

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

def get_H_pauli_op(N, a, b, encoding):

    n = num_qubits_per_dim(N, encoding=encoding)

    H = []
    if encoding == "unary":
        J = np.zeros((n,n))
        for i in range(n-1):
            op = n * ['I']
            op[i] = 'X'
            op[i+1] = 'X'
            H.append(SparsePauliOp(''.join(op), 0.25 * (a - 1) * np.sqrt((i+1) * (i+2))))

            J[i, i+1] = np.sqrt((i + 1) * (i + 2))
        
        for i in range(n):
            op = n * ['I']
            op[i] = 'Z'
            H.append(SparsePauliOp(''.join(op), -0.25 * (1 + a)))

        for j in range(n):
            op = n * ['I']
            op[j] = 'X'
            H.append(SparsePauliOp(''.join(op), b * np.sqrt((j+1) / 2)))

    elif encoding == "one-hot":
        for j in range(n-1):
            op = n * ['I']
            op[j] = 'X'
            op[j+1] = 'X'
            H.append(SparsePauliOp(''.join(op), b * np.sqrt((j+1) / 2)))

        for j in range(n-2):
            op = n * ['I']
            op[j] = 'X'
            op[j+2] = 'X'
            H.append(SparsePauliOp(''.join(op), 0.25 * (a - 1) * np.sqrt((j+1) * (j+2))))
        
        for j in range(n):
            op = n * ['I']
            op[j] = 'Z'
            H.append(SparsePauliOp(''.join(op), -2 * (j+1/2)))

    return sum(H).simplify()

# def get_unary_real_space_circuit(N, t, r, a, b, trotter_method):
#     assert r > 0

#     n = num_qubits_per_dim(N, encoding="unary")

#     dt = t / r
#     circuit = Circuit()

#     if trotter_method == "first_order":
#         for _ in range(r):
#             # b * \hat{x}
#             for j in range(n):
#                 circuit.rx(j, 2 * b * dt * np.sqrt((j+1) / 2))
#             # 0.5 * a * \hat{x}^2 and 0.5 * \hat{p}
#             for j in range(n-1):
#                 circuit.xx(j, j+1, 0.5 * dt * np.sqrt((j+1) * (j+2)) * (a - 1))
#             for j in range(n):
#                 circuit.phaseshift(j, -0.5 * (1 + a) * dt)

#     elif trotter_method == "second_order":
#         for _ in range(r):
#             for j in range(n):
#                 circuit.phaseshift(j, -0.25 * (1 + a) * dt)
#             # b * \hat{x}
#             for j in range(n):
#                 circuit.rx(j, 2 * b * dt * np.sqrt((j+1) / 2))
#             # 0.5 * a * \hat{x}^2 and 0.5 * \hat{p}
#             for j in range(n-1):
#                 circuit.xx(j, j+1, 0.5 * dt * np.sqrt((j+1) * (j+2)) * (a - 1))

#             for j in range(n):
#                 circuit.phaseshift(j, -0.25 * (1 + a) * dt)

#     elif trotter_method == "randomized_first_order":
#         np.random.seed(int(t * r))
#         for _ in range(r):
#             if np.random.rand() < 0.5:
#                 for j in range(n):
#                     circuit.phaseshift(j, -0.5 * (1 + a) * dt)
#                 # b * \hat{x}
#                 for j in range(n):
#                     circuit.rx(j, 2 * b * dt * np.sqrt((j+1) / 2))
#                 # 0.5 * a * \hat{x}^2 and 0.5 * \hat{p}
#                 for j in range(n-1):
#                     circuit.xx(j, j+1, 0.5 * dt * np.sqrt((j+1) * (j+2)) * (a - 1))
#             else:
#                 # b * \hat{x}
#                 for j in range(n):
#                     circuit.rx(j, 2 * b * dt * np.sqrt((j+1) / 2))
#                 # 0.5 * a * \hat{x}^2 and 0.5 * \hat{p}
#                 for j in range(n-1):
#                     circuit.xx(j, j+1, 0.5 * dt * np.sqrt((j+1) * (j+2)) * (a - 1))
#                 for j in range(n):
#                     circuit.phaseshift(j, -0.5 * (1 + a) * dt)

#     else:
#         raise ValueError(f"{trotter_method} not supported")

#     return circuit


def get_binary_resource_estimate(N, T, error_tol, a, b, trotter_method, num_samples, num_jobs):
    
    print(f"N = {N}", flush=True)

    # H_padded = get_real_space_H(2 ** int(np.ceil(np.log2(N))), a, b)
    # for i in np.arange(N, 2 ** int(np.ceil(np.log2(N)))):
    #     for j in range(N):
    #         H_padded[i,j] = 0
    #         H_padded[j,i] = 0
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
    
    compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
    tket_circuit = qiskit_to_tk(compiled_circuit)
    gateset = {OpType.Rx, OpType.Ry, OpType.Rz, OpType.XXPhase}
    rebase = auto_rebase_pass(gateset) 
    comp = SequencePass([FullPeepholeOptimise(), CommuteThroughMultis(), RemoveRedundancies(), rebase])
    comp.apply(tket_circuit)

    # Gates per Trotter step
    num_single_qubit_gates, num_two_qubit_gates = tket_circuit.n_1qb_gates(), tket_circuit.n_2qb_gates()
    print(f"1q gates: {num_single_qubit_gates}, 2q gates: {num_two_qubit_gates}")

    # Estimate number of Trotter steps required
    r_min, r_max = 1, 10
    while std_bin_trotter_error_sampling(H_padded, pauli_op, T, r_max, trotter_method, num_samples, num_jobs) > error_tol:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if std_bin_trotter_error_sampling(H_padded, pauli_op, T, r, trotter_method, num_samples, num_jobs) > error_tol:
            r_min = r
        else:
            r_max = r
    
    print(f"Finished N={N}, trotter steps={r_max}", flush=True)
    return num_single_qubit_gates, num_two_qubit_gates, r_max

if __name__ == "__main__":

    DATA_DIR = "resource_data"
    TASK_DIR = "real_space"

    CURR_DIR = DATA_DIR
    check_and_make_dir(CURR_DIR)
    CURR_DIR = join(CURR_DIR, TASK_DIR)
    check_and_make_dir(CURR_DIR)
    
    print("Resource estimation for real-space simulation.")

    num_jobs = 64
    print("Number of jobs:", num_jobs)
    num_samples = 100

    dimension = 1
    error_tol = 1e-2
    trotter_method = "randomized_first_order"
    print(f"Error tolerance: {error_tol : 0.2f}.")
    print(f"Method: {trotter_method}")

    a, b = 1, -1/2
    T = 5

    N_vals_binary = np.arange(3, 64)
    binary_trotter_steps = np.zeros(len(N_vals_binary))
    binary_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary))
    binary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary))

    # N_vals_unary = np.arange(3, 16)
    # unary_trotter_steps = np.zeros(len(N_vals_unary), dtype=int)
    # unary_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary))
    # unary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary))

    # N_vals_unary_bound = np.arange(3, 101)
    # unary_trotter_steps_bound = np.zeros(len(N_vals_unary_bound), dtype=int)

    N_vals_one_hot = np.arange(3, 16)
    one_hot_trotter_steps = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot))
    one_hot_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot))

    N_vals_one_hot_bound = np.arange(3, 64)
    one_hot_trotter_steps_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)
    one_hot_one_qubit_gate_count_per_trotter_step_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)

    print("\nRunning resource estimation for standard binary encoding")
    for i, N in enumerate(N_vals_binary):

        binary_one_qubit_gate_count_per_trotter_step[i], binary_two_qubit_gate_count_per_trotter_step[i], binary_trotter_steps[i] = get_binary_resource_estimate(N, T, error_tol, a, b, trotter_method, num_samples, num_jobs)

        np.savez(join(CURR_DIR, f"std_binary_{trotter_method}.npz"),
                N_vals_binary=N_vals_binary[:i+1],
                binary_trotter_steps=binary_trotter_steps[:i+1],
                binary_one_qubit_gate_count_per_trotter_step=binary_one_qubit_gate_count_per_trotter_step[:i+1],
                binary_two_qubit_gate_count_per_trotter_step=binary_two_qubit_gate_count_per_trotter_step[:i+1])


    # # Unary encoding
    # encoding = "unary"
    # print(f"\nRunning resource estimation for {encoding} encoding", flush=True)
    # device = LocalSimulator()

    # for i, N in enumerate(N_vals_unary):
    #     start_time = time()

    #     print(f"N = {N} for {encoding}", flush=True)

    #     H = get_real_space_H(N, a, b)
    #     n = num_qubits_per_dim(N, encoding)
    #     codewords = get_codewords(N, dimension, encoding, periodic=False)

    #     tol = 0.5

    #     unary_one_qubit_gate_count_per_trotter_step[i] = 2 * n
    #     unary_two_qubit_gate_count_per_trotter_step[i] = 0

    #     H_ebd = get_unary_real_space_H_ebd(N, a, b)
    #     # Estimate number of Trotter steps required
    #     r_min, r_max = 1, 10
    #     if i > 0:
    #         r_min = unary_trotter_steps[i-1]
    #         r_max = unary_trotter_steps[i-1] * 2
    #     while estimate_trotter_error(N, H_ebd, get_unary_real_space_circuit(N, T, r_max, a, b, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
    #         r_max *= 2

    #     # binary search for r
    #     while r_max - r_min > 1:
    #         r = (r_min + r_max) // 2
    #         if estimate_trotter_error(N, H_ebd, get_unary_real_space_circuit(N, T, r, a, b, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
    #             r_min = r
    #         else:
    #             r_max = r

    #     unary_trotter_steps[i] = r_max

    #     # Save data
    #     np.savez(join(CURR_DIR, f"unary_{trotter_method}.npz"),
    #              N_vals_unary=N_vals_unary[:i+1],
    #              unary_trotter_steps=unary_trotter_steps[:i+1],
    #              unary_one_qubit_gate_count_per_trotter_step=unary_one_qubit_gate_count_per_trotter_step[:i+1],
    #              unary_two_qubit_gate_count_per_trotter_step=unary_two_qubit_gate_count_per_trotter_step[:i+1])

    #     print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

    # # Unary encoding
    # print("Running resource estimation for unary encoding with analytical bound", flush=True)
    # encoding = "unary"

    # for i, N in enumerate(N_vals_unary_bound):
    #     start_time = time()

    #     print(f"Running N = {N}", flush=True)

    #     # Use bound to get Trotter number
    #     unary_trotter_steps_bound[i] = get_trotter_number(get_H_pauli_op(N, a, b, encoding), T, error_tol, trotter_method)

    #     # Save data
    #     np.savez(join(CURR_DIR, f"unary_{trotter_method}_bound.npz"),
    #              N_vals_unary_bound=N_vals_unary_bound[:i+1],
    #              unary_trotter_steps_bound=unary_trotter_steps_bound[:i+1])

    #     print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)
    
    # One hot encoding
    encoding = "one-hot"
    print(f"\nRunning resource estimation for {encoding} encoding", flush=True)
    device = LocalSimulator()

    for i, N in enumerate(N_vals_one_hot):
        start_time = time()

        print(f"N = {N} for {encoding}", flush=True)

        H = get_real_space_H(N, a, b)
        n = num_qubits_per_dim(N, encoding)
        codewords = get_codewords(N, dimension, encoding, periodic=False)

        tol = 0.5

        H_terms, graph = get_H_terms_one_hot(N, H)
        H_ebd = sum(H_terms)
        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        if i > 0:
            r_min = one_hot_trotter_steps[i-1]
            r_max = one_hot_trotter_steps[i-1] * 2
        while estimate_trotter_error(N, H_ebd, T, get_one_hot_circuit(N, H, T, r_max, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
            r_max *= 2

        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error(N, H_ebd, T, get_one_hot_circuit(N, H, T, r, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
                r_min = r
            else:
                r_max = r

        one_hot_trotter_steps[i] = r_max
        one_hot_one_qubit_gate_count_per_trotter_step[i], one_hot_two_qubit_gate_count_per_trotter_step[i] = get_gate_counts(get_one_hot_circuit(N, H, T, 1, trotter_method))

        # Save data
        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}.npz"),
                 N_vals_one_hot=N_vals_one_hot[:i+1],
                 one_hot_trotter_steps=one_hot_trotter_steps[:i+1],
                 one_hot_one_qubit_gate_count_per_trotter_step=one_hot_one_qubit_gate_count_per_trotter_step[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step=one_hot_two_qubit_gate_count_per_trotter_step[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

    print("\nRunning resource estimation for one-hot encoding with analytical bound", flush=True)
    encoding = "one-hot"
    for i, N in enumerate(N_vals_one_hot_bound):
        start_time = time()

        print(f"Running N = {N}", flush=True)
        H = get_real_space_H(N, a, b)

        # Use bound to get Trotter number
        one_hot_trotter_steps_bound[i] = get_trotter_number(get_H_pauli_op(N, a, b, encoding), T, error_tol, trotter_method)
        print(f"Trotter steps: {one_hot_trotter_steps_bound[i]}")
        # Get gate counts
        nonzeros_diag = 0
        nonzeros_off_diag = 0
        for j in range(N):
            if H[j,j] != 0:
                nonzeros_diag += 1
            for k in range(j):
                if H[j,k] != 0:
                    nonzeros_off_diag += 1

        if trotter_method == "first_order" or trotter_method == "randomized_first_order":
            one_hot_one_qubit_gate_count_per_trotter_step_bound[i] = nonzeros_diag
            one_hot_two_qubit_gate_count_per_trotter_step_bound[i] = 2 * nonzeros_off_diag
        elif trotter_method == "second_order":
            one_hot_one_qubit_gate_count_per_trotter_step_bound[i] = nonzeros_diag
            one_hot_two_qubit_gate_count_per_trotter_step_bound[i] = 4 * nonzeros_off_diag
        else:
            raise ValueError(f"{trotter_method} not supported")
        
        # Save data
        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}_bound.npz"),
                 N_vals_one_hot_bound=N_vals_one_hot_bound[:i+1],
                 one_hot_trotter_steps_bound=one_hot_trotter_steps_bound[:i+1],
                 one_hot_one_qubit_gate_count_per_trotter_step_bound=one_hot_one_qubit_gate_count_per_trotter_step_bound[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step_bound=one_hot_two_qubit_gate_count_per_trotter_step_bound[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)