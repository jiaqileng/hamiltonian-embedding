import numpy as np
from utils import *
from resource_estimate_utils import *
from os.path import join
from time import time
import networkx as nx
from random import shuffle

from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate

from braket.devices import LocalSimulator

def get_glued_tree(h):

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

def get_binary_resource_estimate(h, error_tol):
    
    graph = get_glued_tree(h)
    N = graph.order()
    print(f"h = {h}, N = {N}", flush=True)

    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    adjacency_matrix_padded = np.pad(adjacency_matrix, (0, 2 ** int(np.ceil(np.log2(N))) - N))

    # Compute number of gates per Trotter step
    pauli_op = SparsePauliOp.from_operator(adjacency_matrix_padded)
    circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))

    compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry'], optimization_level=3)

    ops = compiled_circuit.count_ops()

    num_single_qubit_gates, num_two_qubit_gates = get_gate_counts(ops)

    assert num_single_qubit_gates + num_two_qubit_gates == sum(ops.values())
    assert num_two_qubit_gates == compiled_circuit.num_nonlocal_gates()

    print(f"N = {N}, num two qubit gates:", num_two_qubit_gates)

    # Use randomized first-order Trotter
    # Estimate number of Trotter steps required
    r_min, r_max = 1, 10
    while std_bin_trotter_error_random(adjacency_matrix_padded, pauli_op, r_max) > error_tol:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if std_bin_trotter_error_random(adjacency_matrix_padded, pauli_op, r) > error_tol:
            r_min = r
        else:
            r_max = r
    
    print(f"Finished N={N}, num two qubit gates={num_two_qubit_gates}, trotter steps={r_max}", flush=True)
    return num_two_qubit_gates, r_max

if __name__ == "__main__":
    
    print("Resource estimation for QW on glued tree.")

    num_jobs = 64
    print("Number of jobs:", num_jobs)

    dimension = 1
    error_tol = 1e-2
    trotter_order = 2
    T = 1
    print(f"Error tolerance: {error_tol : 0.2f}.")
    print(f"Trotter order: {trotter_order}")

    h_vals_binary = np.arange(1,6)
    N_vals_binary = 2 * (2 ** (h_vals_binary + 1) - 1)
    binary_trotter_steps = np.zeros(len(N_vals_binary))
    binary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary))

    h_vals_one_hot = np.arange(1, 3)
    N_vals_one_hot = 2 * (2 ** (h_vals_one_hot + 1) - 1)
    one_hot_trotter_steps = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot))


    print("Running resource estimation for standard binary encoding")
    for i, h in enumerate(h_vals_binary):

        binary_two_qubit_gate_count_per_trotter_step[i], binary_trotter_steps[i] = get_binary_resource_estimate(h, error_tol)

        np.savez(join("resource_data", "resource_estimation_glued_tree_binary.npz"),
                N_vals_binary=N_vals_binary,
                binary_trotter_steps=binary_trotter_steps,
                binary_two_qubit_gate_count_per_trotter_step=binary_two_qubit_gate_count_per_trotter_step)


    # One hot encoding
    print("Running resource estimation for one-hot encoding", flush=True)
    encoding = "one-hot"
    device = LocalSimulator()
    num_samples = 100

    for i, h in enumerate(h_vals_one_hot):
        start_time = time()

        graph = get_glued_tree(h)
        N = graph.order()
        n = num_qubits_per_dim(N, encoding)
        codewords = get_codewords_1d(n, encoding, periodic=False)

        print(f"Running h = {h}, N = {N} for one-hot", flush=True)
        adjacency_matrix = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()

        tol = 0.5
        codewords = get_codewords_1d(n, encoding, periodic=False)

        one_hot_two_qubit_gate_count_per_trotter_step[i] = 2 * graph.size()
    
        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        if i > 0:
            r_min = one_hot_trotter_steps[i-1]
            r_max = one_hot_trotter_steps[i-1] * 2
        while estimate_trotter_error_random(N, graph, r_max, dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
            r_max *= 2

        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error_random(N, graph, r, dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
                r_min = r
            else:
                r_max = r

        one_hot_trotter_steps[i] = r_max

        # Save data
        np.savez(join("resource_data", "resource_estimation_glued_tree_one_hot.npz"),
                 N_vals_one_hot=N_vals_one_hot[:i+1],
                 one_hot_trotter_steps=one_hot_trotter_steps[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step=one_hot_two_qubit_gate_count_per_trotter_step[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)