import numpy as np

from utils import *
from resource_estimate_utils import *
from os.path import join
from time import time

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import norm

from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from pytket import OpType
from pytket.passes import RemoveRedundancies, CommuteThroughMultis, SequencePass, FullPeepholeOptimise, auto_rebase_pass
from pytket.extensions.qiskit import qiskit_to_tk

from braket.devices import LocalSimulator

def get_T_2d(N, H_spatial_search):
    dimension = 2
    T = N ** dimension
    num_time_points = 256
    t_vals = np.linspace(0, T, num_time_points)
    p = 4 * (np.log(N) / N) ** 2
    psi_0 = np.ones(N ** dimension)
    psi_0 /= np.linalg.norm(psi_0)

    psi = expm_multiply(-1j * H_spatial_search, psi_0, start=0, stop=T, num=num_time_points)
    dist = np.abs(psi) ** 2

    return t_vals[np.argmax(dist[:,-N] >= p)]

def get_binary_resource_estimate(N, dimension, error_tol, trotter_method, num_samples, num_jobs):
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

    T = get_T_2d(N, H_spatial_search)

    N_padded = 2 ** int(np.ceil(np.log2(N)))

    L = lil_matrix(get_laplacian_lattice(N_padded, d=dimension))
    binary_indices = []
    other_indices = []
    for i in range(N_padded ** dimension):
        col = i % N_padded
        row = i // N_padded
        if col < N and row < N:
            binary_indices.append(i)
        else:
            other_indices.append(i)

    H_spatial_search_padded = -gamma * L
    # Make sure the diagonal part is the same
    for i in range(len(binary_indices)):
        for j in range(len(binary_indices)):
            H_spatial_search_padded[binary_indices[i], binary_indices[j]] = H_spatial_search[i,j]
        
    # Remove all other edges
    for i in binary_indices:
        for j in other_indices:
            H_spatial_search_padded[i,j] = 0
            H_spatial_search_padded[j,i] = 0
    # for i in other_indices:
    #     for j in other_indices:
    #         H_spatial_search_padded[i,j] = 0

    pauli_op = SparsePauliOp.from_operator(H_spatial_search_padded.toarray())
    
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
    while std_bin_trotter_error_sampling(H_spatial_search_padded, pauli_op, T, r_max, trotter_method, num_samples, num_jobs) > error_tol:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if std_bin_trotter_error_sampling(H_spatial_search_padded, pauli_op, T, r, trotter_method, num_samples, num_jobs) > error_tol:
            r_min = r
        else:
            r_max = r
    
    print(f"Finished N={N}, trotter steps={r_max}", flush=True)
    return num_single_qubit_gates, num_two_qubit_gates, r_max

def get_H_spatial_search(n, lamb, gamma, encoding, dimension):
    assert dimension == 2
    if encoding == "unary" or encoding == "antiferromagnetic":
        if encoding == "antiferromagnetic":
            assert n % 2 == 1, "only works for odd number of qubits"
        J = np.zeros((dimension * n, dimension * n))
        h = np.zeros(dimension * n)

        for i in range(dimension):
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

        H_pen = lamb * (sum_J_zz(dimension * n, J) + sum_h_z(dimension * n, h))

        # Create the oracle Hamiltonian
        J = np.zeros((dimension * n, dimension * n))
        J[n-1, n] = 1/4

        h = np.zeros(dimension * n)
        h[n-1] = 1/4
        h[n] = -1/4

        H_oracle = sum_h_z(dimension * n, h) + sum_J_zz(dimension * n, J)

        # Correction term for Laplacian
        h_correction = np.zeros(dimension * n)
        for i in range(dimension):
            h_correction[i * n] = 1/2
            if encoding == "unary":
                h_correction[(i + 1) * n - 1] = -1/2
            else:
                h_correction[(i + 1) * n - 1] = -(-1) ** (n+1) / 2
        H_correction = (sum_h_z(dimension * n, h_correction))
        H_adj = sum_x(dimension * n)
        H_laplacian = H_adj + H_correction

        return -gamma * H_laplacian + H_oracle + H_pen
    elif encoding == "one-hot":

        # Create the oracle Hamiltonian
        J = np.zeros((dimension * n, dimension * n))
        J[n-1, n] = -1/4
        h = np.zeros(dimension * n)
        h[n-1] = 1/4
        h[n] = 1/4
        H_oracle = sum_J_zz(dimension * n, J) + sum_h_z(dimension * n, h)

        # Laplacian correction term
        h = np.zeros(dimension * n)
        for i in range(dimension):
            h[i * n] = 1/2
            h[(i+1) * n - 1] = 1/2

        # Adjacency matrix
        J = np.zeros((dimension * n, dimension * n))
        for i in range(dimension):
            for j in np.arange(i * n, (i + 1) * n - 1):
                J[j,j+1] = 1
        
        H_laplacian = (sum_J_xx(dimension * n, J) + sum_J_yy(dimension * n, J)) / 2 - sum_h_z(dimension * n, h)

        return -gamma * H_laplacian + H_oracle
    else:
        raise ValueError("Encoding not supported")
    
def get_H_pauli_op(n, lamb, gamma, encoding, dimension):
    H = []

    if encoding == "unary" or encoding == "antiferromagnetic":
        if encoding == "antiferromagnetic":
            assert n % 2 == 1, "only works for odd number of qubits"

        for i in range(dimension):
            op = dimension * n * ['I']
            op[i * n] = 'Z'
            H.append(SparsePauliOp(''.join(op), lamb))
            if encoding == "unary":
                op = dimension * n * ['I']
                op[(i + 1) * n - 1] = 'Z'
                H.append(SparsePauliOp(''.join(op), -lamb))
            else:
                op = dimension * n * ['I']
                op[(i + 1) * n - 1] = 'Z'
                H.append(SparsePauliOp(''.join(op), ((-1) ** n) * lamb))
            
            for j in np.arange(i * n, (i + 1) * n - 1):
                if encoding == "unary":
                    op = dimension * n * ['I']
                    op[j] = 'Z'
                    op[j + 1] = 'Z'
                    H.append(SparsePauliOp(''.join(op), - lamb))
                else:                    
                    op = dimension * n * ['I']
                    op[j] = 'Z'
                    op[j + 1] = 'Z'
                    H.append(SparsePauliOp(''.join(op), lamb))

        # Create the oracle Hamiltonian
        op = dimension * n * ['I']
        op[n-1] = 'Z'
        op[n] = 'Z'
        H.append(SparsePauliOp(''.join(op), 1/4))

        op = dimension * n * ['I']
        op[n-1] = 'Z'
        H.append(SparsePauliOp(''.join(op), 1/4))
        op = dimension * n * ['I']
        op[n] = 'Z'
        H.append(SparsePauliOp(''.join(op), -1/4))

        # Correction term for Laplacian
        for i in range(dimension):
            op = dimension * n * ['I']
            op[i * n] = 'Z'
            H.append(SparsePauliOp(''.join(op), - gamma / 2))
            if encoding == "unary":
                op = dimension * n * ['I']
                op[(i + 1) * n - 1] = 'Z'
                H.append(SparsePauliOp(''.join(op), gamma / 2))
            else:
                op = dimension * n * ['I']
                op[(i + 1) * n - 1] = 'Z'
                H.append(SparsePauliOp(''.join(op), gamma * ((-1) ** (n+1)) / 2))
        
        for i in range(dimension * n):
            op = dimension * n * ['I']
            op[i] = 'X'
            H.append(SparsePauliOp(''.join(op), - gamma))

        return sum(H).simplify()
    elif encoding == "one-hot":

        # Create the oracle Hamiltonian
        op = dimension * n * ['I']
        op[n-1] = 'Z'
        op[n] = 'Z'
        H.append(SparsePauliOp(''.join(op), - 1/4))

        h = np.zeros(dimension * n)
        op = dimension * n * ['I']
        op[n-1] = 'Z'
        H.append(SparsePauliOp(''.join(op), 1/4))
        op = dimension * n * ['I']
        op[n] = 'Z'
        H.append(SparsePauliOp(''.join(op), 1/4))

        # Laplacian correction term
        for i in range(dimension):
            op = dimension * n * ['I']
            op[i * n] = 'Z'
            H.append(SparsePauliOp(''.join(op), gamma / 2))

            op = dimension * n * ['I']
            op[(i+1) * n - 1] = 'Z'
            H.append(SparsePauliOp(''.join(op), gamma / 2))

        # Adjacency matrix
        for i in range(dimension):
            for j in np.arange(i * n, (i + 1) * n - 1):
                op = dimension * n * ['I']
                op[j] = 'X'
                op[j+1] = "X"
                H.append(SparsePauliOp(''.join(op), - gamma / 2))

                op = dimension * n * ['I']
                op[j] = 'Y'
                op[j+1] = "Y"
                H.append(SparsePauliOp(''.join(op), - gamma / 2))

        return sum(H).simplify()
    else:
        raise ValueError("Encoding not supported")

def get_spatial_search_circuit(N, lamb, gamma, t, r, encoding, dimension, trotter_method):
    n = num_qubits_per_dim(N, encoding)

    dt = t / r

    circuit = Circuit()
    if encoding == "unary":
    
        if trotter_method == "first_order":
            for _ in range(r):
                # x rotations
                for i in range(dimension * n):
                    circuit.rx(i, - 2 * gamma * dt)

                # penalty term
                for i in range(dimension):
                    circuit.rz(i * n, 2 * lamb * dt)
                    circuit.rz((i + 1) * n - 1, - 2 * lamb * dt)
                    
                    for j in np.arange(i * n, (i + 1) * n - 1):
                        circuit.zz(j, j+1, -2 * lamb * dt)

                # oracle term
                circuit.zz(n-1, n, dt / 2)
                circuit.rz(n-1, dt / 2)
                circuit.rz(n, -dt / 2)

                # laplacian correction term
                for i in range(dimension):
                    circuit.rz(i * n, -gamma * dt)
                    circuit.rz((i + 1) * n - 1, gamma * dt)
            
        elif trotter_method == "second_order":
            for _ in range(r):
                # x rotations
                for i in range(dimension * n):
                    circuit.rx(i, - gamma * dt)

                # penalty term
                for i in range(dimension):
                    circuit.rz(i * n, 2 * lamb * dt)
                    circuit.rz((i + 1) * n - 1, - 2 * lamb * dt)
                    
                    for j in np.arange(i * n, (i + 1) * n - 1):
                        circuit.zz(j, j+1, -2 * lamb * dt)

                # oracle term
                circuit.zz(n-1, n, dt / 2)
                circuit.rz(n-1, dt / 2)
                circuit.rz(n, -dt / 2)

                # laplacian correction term
                for i in range(dimension):
                    circuit.rz(i * n, -gamma * dt)
                    circuit.rz((i + 1) * n - 1, gamma * dt)
                    
                # x rotations
                for i in range(dimension * n):
                    circuit.rx(i, - gamma * dt)
        
        elif trotter_method == "randomized_first_order":
            np.random.seed(t * r)
            for _ in range(r):
                if np.random.rand() < 0.5:
                    # x rotations
                    for i in range(dimension * n):
                        circuit.rx(i, - 2 * gamma * dt)

                    # penalty term
                    for i in range(dimension):
                        circuit.rz(i * n, 2 * lamb * dt)
                        circuit.rz((i + 1) * n - 1, - 2 * lamb * dt)
                        
                        for j in np.arange(i * n, (i + 1) * n - 1):
                            circuit.zz(j, j+1, -2 * lamb * dt)

                    # oracle term
                    circuit.zz(n-1, n, dt / 2)
                    circuit.rz(n-1, dt / 2)
                    circuit.rz(n, -dt / 2)

                    # laplacian correction term
                    for i in range(dimension):
                        circuit.rz(i * n, -gamma * dt)
                        circuit.rz((i + 1) * n - 1, gamma * dt)
                else:
                    
                    # penalty term
                    for i in range(dimension):
                        circuit.rz(i * n, 2 * lamb * dt)
                        circuit.rz((i + 1) * n - 1, - 2 * lamb * dt)
                        
                        for j in np.arange(i * n, (i + 1) * n - 1):
                            circuit.zz(j, j+1, -2 * lamb * dt)

                    # oracle term
                    circuit.zz(n-1, n, dt / 2)
                    circuit.rz(n-1, dt / 2)
                    circuit.rz(n, -dt / 2)

                    # laplacian correction term
                    for i in range(dimension):
                        circuit.rz(i * n, -gamma * dt)
                        circuit.rz((i + 1) * n - 1, gamma * dt)

                    # x rotations
                    for i in range(dimension * n):
                        circuit.rx(i, - 2 * gamma * dt)
        else:
            raise ValueError("Trotter order must be 1 or 2")
    elif encoding == "one-hot":
        marked_vertex_index_1 = N-1
        marked_vertex_index_2 = 0
        if trotter_method == "first_order":
            for _ in range(r):
                # Laplacian term
                for i in range(dimension):
                    for j in np.arange(i * n, (i + 1) * n - 1, 2):
                        circuit.xx(j, j + 1, - gamma * dt)
                        circuit.yy(j, j + 1, - gamma * dt)
                    for j in np.arange(i * n + 1, (i + 1) * n - 1, 2):
                        circuit.xx(j, j + 1, - gamma * dt)
                        circuit.yy(j, j + 1, - gamma * dt)
                        
                # Laplacian correction term
                for i in range(dimension):
                    circuit.rz(i * n, gamma * dt)
                    circuit.rz((i + 1) * n - 1, gamma * dt)

                # Oracle term
                circuit.zz(marked_vertex_index_1, n + marked_vertex_index_2, - dt / 2)
                circuit.rz(marked_vertex_index_1, dt / 2)
                circuit.rz(n + marked_vertex_index_2, dt / 2)

        elif trotter_method == "second_order":
            for _ in range(r):
                # Laplacian term
                for i in range(dimension):
                    for j in np.arange(i * n, (i + 1) * n - 1, 2):
                        circuit.xx(j, j + 1, - 0.5 * gamma * dt)
                        circuit.yy(j, j + 1, - 0.5 * gamma * dt)
                    for j in np.arange(i * n + 1, (i + 1) * n - 1, 2):
                        circuit.xx(j, j + 1, - 0.5 * gamma * dt)
                        circuit.yy(j, j + 1, - 0.5 * gamma * dt)
                        
                # Laplacian correction term
                for i in range(dimension):
                    circuit.rz(i * n, gamma * dt)
                    circuit.rz((i + 1) * n - 1, gamma * dt)

                # Oracle term
                circuit.zz(marked_vertex_index_1, n + marked_vertex_index_2, - dt / 2)
                circuit.rz(marked_vertex_index_1, dt / 2)
                circuit.rz(n + marked_vertex_index_2, dt / 2)

                # Laplacian term
                for i in range(dimension):
                    for j in np.arange(i * n + 1, (i + 1) * n - 1, 2):
                        circuit.xx(j, j + 1, - 0.5 * gamma * dt)
                        circuit.yy(j, j + 1, - 0.5 * gamma * dt)
                    for j in np.arange(i * n, (i + 1) * n - 1, 2):
                        circuit.xx(j, j + 1, - 0.5 * gamma * dt)
                        circuit.yy(j, j + 1, - 0.5 * gamma * dt)
                        
        elif trotter_method == "randomized_first_order":
            
            np.random.seed(t * r)
            for _ in range(r):
                if np.random.rand() < 0.5:
                    # Laplacian term
                    for i in range(dimension):
                        for j in np.arange(i * n, (i + 1) * n - 1, 2):
                            circuit.xx(j, j + 1, - gamma * dt)
                            circuit.yy(j, j + 1, - gamma * dt)
                        for j in np.arange(i * n + 1, (i + 1) * n - 1, 2):
                            circuit.xx(j, j + 1, - gamma * dt)
                            circuit.yy(j, j + 1, - gamma * dt)
                            
                    # Laplacian correction term
                    for i in range(dimension):
                        circuit.rz(i * n, gamma * dt)
                        circuit.rz((i + 1) * n - 1, gamma * dt)

                    # Oracle term
                    circuit.zz(marked_vertex_index_1, n + marked_vertex_index_2, - dt / 2)
                    circuit.rz(marked_vertex_index_1, dt / 2)
                    circuit.rz(n + marked_vertex_index_2, dt / 2)

                else:
                    # Oracle term
                    circuit.zz(marked_vertex_index_1, n + marked_vertex_index_2, - dt / 2)
                    circuit.rz(marked_vertex_index_1, dt / 2)
                    circuit.rz(n + marked_vertex_index_2, dt / 2)

                    # Laplacian correction term
                    for i in range(dimension):
                        circuit.rz(i * n, gamma * dt)
                        circuit.rz((i + 1) * n - 1, gamma * dt)

                    # Laplacian term
                    for i in range(dimension):
                        for j in np.arange(i * n + 1, (i + 1) * n - 1, 2):
                            circuit.xx(j, j + 1, - gamma * dt)
                            circuit.yy(j, j + 1, - gamma * dt)
                        for j in np.arange(i * n, (i + 1) * n - 1, 2):
                            circuit.xx(j, j + 1, - gamma * dt)
                            circuit.yy(j, j + 1, - gamma * dt)
    else:
        raise ValueError("Encoding not supported")
    return circuit

if __name__ == "__main__":

    DATA_DIR = "resource_data"
    TASK_DIR = "spatial_search_2d"

    CURR_DIR = DATA_DIR
    check_and_make_dir(CURR_DIR)
    CURR_DIR = join(CURR_DIR, TASK_DIR)
    check_and_make_dir(CURR_DIR)
    
    print("Resource estimation for 2D spatial search.")
    num_jobs = 64
    print("Number of jobs:", num_jobs)
    num_samples = 1000

    dimension = 2
    error_tol = 5e-2
    fidelity_tol = 1e-2
    trotter_method = "second_order"

    print(f"Error tolerance: {error_tol : 0.2f}.")
    print(f"Fidelity tolerance: {fidelity_tol : 0.2f}.")
    print(f"Method: {trotter_method}")

    N_vals_binary = np.arange(3, 12)
    binary_trotter_steps = np.zeros(len(N_vals_binary))
    binary_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary))
    binary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary))
    
    N_vals_unary = np.arange(3, 7)
    unary_trotter_steps = np.zeros(len(N_vals_unary), dtype=int)
    unary_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary), dtype=int)
    unary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary), dtype=int)

    N_vals_unary_bound = np.arange(3, 12)
    unary_trotter_steps_bound = np.zeros(len(N_vals_unary_bound), dtype=int)
    unary_one_qubit_gate_count_per_trotter_step_bound = np.zeros(len(N_vals_unary_bound), dtype=int)
    unary_two_qubit_gate_count_per_trotter_step_bound = np.zeros(len(N_vals_unary_bound), dtype=int)

    N_vals_one_hot = np.arange(3, 7)
    one_hot_trotter_steps = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot), dtype=int)

    N_vals_one_hot_bound = np.arange(3, 12)
    one_hot_trotter_steps_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)
    one_hot_one_qubit_gate_count_per_trotter_step_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)


    print("\nRunning resource estimation for standard binary encoding", flush=True)
    for i, N in enumerate(N_vals_binary):
        start_time = time()
        binary_one_qubit_gate_count_per_trotter_step[i], binary_two_qubit_gate_count_per_trotter_step[i], binary_trotter_steps[i] = get_binary_resource_estimate(N, dimension, error_tol, trotter_method, num_samples, num_jobs)

        np.savez(join(CURR_DIR, f"std_binary_{trotter_method}_remove_edges.npz"),
                N_vals_binary=N_vals_binary[:i+1],
                binary_trotter_steps=binary_trotter_steps[:i+1],
                binary_one_qubit_gate_count_per_trotter_step=binary_one_qubit_gate_count_per_trotter_step[:i+1],
                binary_two_qubit_gate_count_per_trotter_step=binary_two_qubit_gate_count_per_trotter_step[:i+1])
        
        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

    # Unary encoding
    encoding = "unary"
    print(f"\nRunning resource estimation for {encoding} encoding", flush=True)
    device = LocalSimulator()
    
    for i, N in enumerate(N_vals_unary):

        start_time = time()
        print(f"Running N = {N}", flush=True)
        n = num_qubits_per_dim(N, encoding)
        codewords = get_codewords(N, dimension, encoding, periodic=False)

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
        H_spatial_search = - gamma * L + H_oracle
        T = get_T_2d(N, H_spatial_search)

        # lamb_min, lamb_max = 0, gamma * dimension * N * T
        # while subspace_fidelity(n, T, get_H_spatial_search(n, lamb_max, gamma, encoding, dimension), (-gamma * L + H_oracle), codewords) < 1 - fidelity_tol:
        #     lamb_max *= 2
        # while lamb_max - lamb_min > 1e-2:
        #     lamb = (lamb_min + lamb_max) / 2
        #     if subspace_fidelity(n, T, get_H_spatial_search(n, lamb, gamma, encoding, dimension), (-gamma * L + H_oracle), codewords) < 1 - fidelity_tol:
        #         lamb_min = lamb
        #     else:
        #         lamb_max = lamb
        # lamb = lamb_max
        # print(f"lamb = {lamb : 0.2f}")
        lamb = gamma * N * dimension * T

        H_ebd = get_H_spatial_search(n, lamb, gamma, encoding, dimension)

        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        while estimate_trotter_error(N, H_ebd, T, get_spatial_search_circuit(N, lamb, gamma, T, r_max, encoding, dimension, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
            r_max *= 2
        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error(N, H_ebd, T, get_spatial_search_circuit(N, lamb, gamma, T, r, encoding, dimension, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
                r_min = r
            else:
                r_max = r
        
        unary_trotter_steps[i] = r_max
        unary_one_qubit_gate_count_per_trotter_step[i], unary_two_qubit_gate_count_per_trotter_step[i] = get_gate_counts(get_spatial_search_circuit(N, lamb, gamma, T, 1, encoding, dimension, trotter_method))

        # Save data
        np.savez(join(CURR_DIR, f"unary_{trotter_method}.npz"),
                 N_vals_unary=N_vals_unary[:i+1],
                 unary_trotter_steps=unary_trotter_steps[:i+1],
                 unary_one_qubit_gate_count_per_trotter_step=unary_one_qubit_gate_count_per_trotter_step[:i+1],
                 unary_two_qubit_gate_count_per_trotter_step=unary_two_qubit_gate_count_per_trotter_step[:i+1])
        
        
        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)
    
    # Unary encoding
    print("\nRunning resource estimation for unary encoding with analytical bound", flush=True)
    encoding = "unary"

    for i, N in enumerate(N_vals_unary_bound):
        start_time = time()

        print(f"Running N = {N}", flush=True)
        n = num_qubits_per_dim(N, encoding)
        codewords = get_codewords(N, dimension, encoding, periodic=False)

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
        H_spatial_search = - gamma * L + H_oracle
        T = get_T_2d(N, H_spatial_search)

        # lamb_min, lamb_max = 0, gamma * dimension * N * T
        # while subspace_fidelity(n, T, get_H_spatial_search(n, lamb_max, gamma, encoding, dimension), (-gamma * L + H_oracle), codewords) < 1 - fidelity_tol:
        #     lamb_max *= 2
        # while lamb_max - lamb_min > 1e-2:
        #     lamb = (lamb_min + lamb_max) / 2
        #     if subspace_fidelity(n, T, get_H_spatial_search(n, lamb, gamma, encoding, dimension), (-gamma * L + H_oracle), codewords) < 1 - fidelity_tol:
        #         lamb_min = lamb
        #     else:
        #         lamb_max = lamb
        # lamb = lamb_max
        # print(f"lamb = {lamb : 0.2f}")
        lamb = gamma * N * dimension * T

        # Use bound to get Trotter number
        unary_trotter_steps_bound[i] = get_trotter_number(get_H_pauli_op(n, lamb, gamma, encoding, dimension), T, error_tol, trotter_method)
        unary_one_qubit_gate_count_per_trotter_step_bound[i], unary_two_qubit_gate_count_per_trotter_step_bound[i] = get_gate_counts(get_spatial_search_circuit(N, lamb, gamma, T, 1, encoding, dimension, trotter_method))

        # Save data
        np.savez(join(CURR_DIR, f"unary_{trotter_method}_bound.npz"),
                 N_vals_unary_bound=N_vals_unary_bound[:i+1],
                 unary_trotter_steps_bound=unary_trotter_steps_bound[:i+1],
                 unary_one_qubit_gate_count_per_trotter_step_bound=unary_one_qubit_gate_count_per_trotter_step_bound[:i+1],
                 unary_two_qubit_gate_count_per_trotter_step_bound=unary_two_qubit_gate_count_per_trotter_step_bound[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

    # One-hot encoding
    encoding = "one-hot"
    print(f"\nRunning resource estimation for {encoding} encoding", flush=True)
    device = LocalSimulator()

    for i, N in enumerate(N_vals_one_hot):

        start_time = time()
        print(f"Running N = {N}", flush=True)
        n = num_qubits_per_dim(N, encoding)
        codewords = get_codewords(N, dimension, encoding, periodic=False)

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
        H_spatial_search = - gamma * L + H_oracle
        T = get_T_2d(N, H_spatial_search)

        lamb = gamma * dimension * N * T
        
        H_ebd = get_H_spatial_search(n, lamb, gamma, encoding, dimension)

        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        while estimate_trotter_error(N, H_ebd, T, get_spatial_search_circuit(N, lamb, gamma, T, r_max, encoding, dimension, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
            r_max *= 2
        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error(N, H_ebd, T, get_spatial_search_circuit(N, lamb, gamma, T, r, encoding, dimension, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
                r_min = r
            else:
                r_max = r
        
        one_hot_trotter_steps[i] = r_max
        one_hot_one_qubit_gate_count_per_trotter_step[i], one_hot_two_qubit_gate_count_per_trotter_step[i] = get_gate_counts(get_spatial_search_circuit(N, lamb, gamma, T, 1, encoding, dimension, trotter_method))

        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}.npz"),
                 N_vals_one_hot=N_vals_one_hot[:i+1],
                 one_hot_trotter_steps=one_hot_trotter_steps[:i+1],
                 one_hot_one_qubit_gate_count_per_trotter_step=one_hot_one_qubit_gate_count_per_trotter_step[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step=one_hot_two_qubit_gate_count_per_trotter_step[:i+1])
        
        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

    # One hot encoding
    print("\nRunning resource estimation for one-hot encoding with analytical bound", flush=True)
    encoding = "one-hot"

    for i, N in enumerate(N_vals_one_hot_bound):
        start_time = time()

        print(f"Running N = {N}", flush=True)
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
        H_spatial_search = - gamma * L + H_oracle
        T = get_T_2d(N, H_spatial_search)

        lamb = gamma * dimension * N * T

        # Use bound to get Trotter number
        one_hot_trotter_steps_bound[i] = get_trotter_number(get_H_pauli_op(n, lamb, gamma, encoding, dimension), T, error_tol, trotter_method)
        one_hot_one_qubit_gate_count_per_trotter_step_bound[i], one_hot_two_qubit_gate_count_per_trotter_step_bound[i] = get_gate_counts(get_spatial_search_circuit(N, lamb, gamma, T, 1, encoding, dimension, trotter_method))

        # Save data
        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}_bound.npz"),
                 N_vals_one_hot_bound=N_vals_one_hot_bound[:i+1],
                 one_hot_trotter_steps_bound=one_hot_trotter_steps_bound[:i+1],
                 one_hot_one_qubit_gate_count_per_trotter_step_bound=one_hot_one_qubit_gate_count_per_trotter_step_bound[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step_bound=one_hot_two_qubit_gate_count_per_trotter_step_bound[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)