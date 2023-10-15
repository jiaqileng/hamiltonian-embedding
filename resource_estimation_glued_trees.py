import numpy as np
from utils import *
from resource_estimate_utils import *
from os.path import join
from time import time
import networkx as nx
from random import shuffle, seed
from scipy.sparse import diags

from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from pytket import OpType
from pytket.passes import RemoveRedundancies, CommuteThroughMultis, SequencePass, FullPeepholeOptimise, auto_rebase_pass
from pytket.extensions.qiskit import qiskit_to_tk

from braket.devices import LocalSimulator

def get_glued_tree(h):
    seed(0)
    # Two binary trees of height h (2^(h+1) - 1 nodes each glued together)
    num_nodes_per_binary_tree = 2 ** (h+1) - 1
    num_nodes = 2 * num_nodes_per_binary_tree
    graph = nx.Graph()

    # Leaves
    leaves_first = []
    leaves_second = []
    for i in range(2 ** h):
        leaves_first.append(2 ** h - 1 + i)
        leaves_second.append(num_nodes - 1 - (2 ** h - 1) - i)

    for i in np.arange(1, num_nodes_per_binary_tree):

        # First binary tree
        graph.add_edge(int((i-1)/2), i)

        # Second binary tree
        graph.add_edge(num_nodes - 1 - int((i-1)/2), num_nodes - 1 - i)

    # Glue the two trees together
    # Shuffle the leaves to get a random cycle
    shuffle(leaves_first)
    shuffle(leaves_second)

    for i in range(2 ** h):
        graph.add_edge(leaves_first[i], leaves_second[i])
        graph.add_edge(leaves_second[i], leaves_first[(i+1) % (2 ** h)])

    return graph

def get_H_pauli_op(n, graph):

    H = []

    for i,j in graph.edges():
        op = n * ['I']
        op[i] = 'X'
        op[j] = 'X'
        H.append(SparsePauliOp(''.join(op), 1/2))

        op = n * ['I']
        op[i] = 'Y'
        op[j] = 'Y'
        H.append(SparsePauliOp(''.join(op), 1/2))

    return sum(H).simplify()

def get_binary_resource_estimate(h, error_tol, trotter_method, num_samples, num_jobs):
    
    T = get_T_s(h, s)
    graph = get_glued_tree(h)
    N = graph.order()
    print(f"h = {h}, N = {N}", flush=True)

    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    adjacency_matrix_padded = np.pad(adjacency_matrix, (0, 2 ** int(np.ceil(np.log2(N))) - N))

    pauli_op = SparsePauliOp.from_operator(adjacency_matrix_padded)

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
    while std_bin_trotter_error_sampling(adjacency_matrix_padded, pauli_op, T, r_max, trotter_method, num_samples, num_jobs) > error_tol:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if std_bin_trotter_error_sampling(adjacency_matrix_padded, pauli_op, T, r, trotter_method, num_samples, num_jobs) > error_tol:
            r_min = r
        else:
            r_max = r
    
    print(f"Finished N={N}, num two qubit gates={num_two_qubit_gates}, trotter steps={r_max}", flush=True)
    return num_single_qubit_gates, num_two_qubit_gates, r_max

def get_weighted_chain_hamiltonian(h):
    # weighted chain with 2 * (h + 1) vertices
    num_vertices = 2 * (h + 1)
    weights = np.sqrt(2) * np.ones(num_vertices - 1)
    weights[int((num_vertices - 1) / 2)] = 2
    return diags([weights, weights], offsets=[1,-1])

def get_T_s(h, s):
    num_time_points = 256
    T = h + 1
    t_vals = np.linspace(0, T, num_time_points)

    H_weighted_chain = get_weighted_chain_hamiltonian(h)
    psi_0 = np.zeros(2 * (h + 1))
    psi_0[0] = 1
    psi = expm_multiply(-1j * H_weighted_chain, psi_0, start=0, stop=T, num=num_time_points)
    dist = np.abs(psi) ** 2
    return t_vals[np.argmax(dist[:,-1] > s)]

if __name__ == "__main__":

    DATA_DIR = "resource_data"
    TASK_DIR = "glued_trees"

    CURR_DIR = DATA_DIR
    check_and_make_dir(CURR_DIR)
    CURR_DIR = join(CURR_DIR, TASK_DIR)
    check_and_make_dir(CURR_DIR)
    
    print("Resource estimation for QW on glued tree.")

    num_jobs = 64
    print("Number of jobs:", num_jobs)
    num_samples = 1000

    dimension = 1
    error_tol = 5e-2
    trotter_method = "randomized_first_order"

    s = 0.4
    print(f"Probability threshold s = {s : 0.2f}")
    print(f"Error tolerance: {error_tol : 0.2f}.")
    print(f"Method: {trotter_method}")

    h_vals_binary = np.arange(1, 6)
    N_vals_binary = 2 * (2 ** (h_vals_binary + 1) - 1)
    binary_trotter_steps = np.zeros(len(N_vals_binary))
    binary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary), dtype=int)
    binary_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary), dtype=int)

    h_vals_one_hot = np.arange(1, 3)
    N_vals_one_hot = 2 * (2 ** (h_vals_one_hot + 1) - 1)
    one_hot_trotter_steps = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot), dtype=int)

    h_vals_one_hot_bound = np.arange(1, 6)
    N_vals_one_hot_bound = 2 * (2 ** (h_vals_one_hot_bound + 1) - 1)
    one_hot_trotter_steps_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)
    one_hot_one_qubit_gate_count_per_trotter_step_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)

    print("\nRunning resource estimation for standard binary encoding")
    for i, h in enumerate(h_vals_binary):
        start_time = time()
        binary_one_qubit_gate_count_per_trotter_step[i], binary_two_qubit_gate_count_per_trotter_step[i], binary_trotter_steps[i] = get_binary_resource_estimate(h, error_tol, trotter_method, num_samples, num_jobs)

        np.savez(join(CURR_DIR, f"std_binary_{trotter_method}.npz"),
                N_vals_binary=N_vals_binary[:i+1],
                binary_trotter_steps=binary_trotter_steps[:i+1],
                binary_one_qubit_gate_count_per_trotter_step=binary_one_qubit_gate_count_per_trotter_step[:i+1],
                binary_two_qubit_gate_count_per_trotter_step=binary_two_qubit_gate_count_per_trotter_step[:i+1])
        
        print(f"Time = {time() - start_time} seconds.", flush=True)

    # One hot encoding
    print("\nRunning resource estimation for one-hot encoding", flush=True)
    encoding = "one-hot"
    device = LocalSimulator()

    for i, h in enumerate(h_vals_one_hot):
        start_time = time()
        T = get_T_s(h, s)
        graph = get_glued_tree(h)
        N = graph.order()
        n = num_qubits_per_dim(N, encoding)
        codewords = get_codewords(N, dimension, encoding, periodic=False)

        print(f"Running N = {N} for {encoding}", flush=True)
        adjacency_matrix = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    
        H_terms, graph = get_H_terms_one_hot(N, adjacency_matrix)
        H_ebd = sum(H_terms)

        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        if i > 0:
            r_min = one_hot_trotter_steps[i-1]
            r_max = one_hot_trotter_steps[i-1] * 2
        while estimate_trotter_error(N, H_ebd, T, get_one_hot_circuit(N, adjacency_matrix, T, r_max, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
            r_max *= 2

        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error(N, H_ebd, T, get_one_hot_circuit(N, adjacency_matrix, T, r, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
                r_min = r
            else:
                r_max = r
    
        one_hot_trotter_steps[i] = r_max
        one_hot_one_qubit_gate_count_per_trotter_step[i], one_hot_two_qubit_gate_count_per_trotter_step[i] = get_gate_counts(get_one_hot_circuit(N, adjacency_matrix, T, 1, trotter_method))

        # Save data
        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}.npz"),
                 N_vals_one_hot=N_vals_one_hot[:i+1],
                 one_hot_trotter_steps=one_hot_trotter_steps[:i+1],
                 one_hot_one_qubit_gate_count_per_trotter_step=one_hot_one_qubit_gate_count_per_trotter_step[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step=one_hot_two_qubit_gate_count_per_trotter_step[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

    # One hot encoding
    print("\nRunning resource estimation for one-hot encoding with analytical bound", flush=True)
    encoding = "one-hot"

    for i, h in enumerate(h_vals_one_hot_bound):
        start_time = time()

        T = get_T_s(h, s)
        graph = get_glued_tree(h)
        N = graph.order()
        n = num_qubits_per_dim(N, encoding)
        adjacency_matrix = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
        
        # Use bound to get Trotter number
        one_hot_trotter_steps_bound[i] = get_trotter_number(get_H_pauli_op(n, graph), T, error_tol, trotter_method)
        # Get gate counts
        if trotter_method == "first_order" or trotter_method == "randomized_first_order":
            one_hot_two_qubit_gate_count_per_trotter_step_bound[i] = 2 * graph.size()
        elif trotter_method == "second_order":
            one_hot_two_qubit_gate_count_per_trotter_step_bound[i] = 4 * graph.size()
        else:
            raise ValueError(f"{trotter_method} not supported")
        
        # Save data
        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}_bound.npz"),
                 N_vals_one_hot_bound=N_vals_one_hot_bound[:i+1],
                 one_hot_trotter_steps_bound=one_hot_trotter_steps_bound[:i+1],
                 one_hot_one_qubit_gate_count_per_trotter_step_bound=one_hot_one_qubit_gate_count_per_trotter_step_bound[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step_bound=one_hot_two_qubit_gate_count_per_trotter_step_bound[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)