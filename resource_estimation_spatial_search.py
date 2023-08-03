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

def get_binary_resource_estimate(N, dimension, error_tol, trotter_method):
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

    N_padded = 2 ** int(np.ceil(np.log2(N)))

    L = get_laplacian_lattice(N_padded, d=dimension)
    binary_indices = []
    other_indices = []
    for i in range(N_padded ** dimension):
        col = i % N_padded
        row = i // N_padded
        if col < N and row < N:
            binary_indices.append(i)
        else:
            other_indices.append(i)
            

    H_spatial_search_padded = (-gamma * L).toarray()
    # Make sure the diagonal part is the same
    for i in range(len(binary_indices)):
        H_spatial_search_padded[binary_indices[i], binary_indices[i]] = H_spatial_search[i,i]
        
    # Remove all other edges
    for i in binary_indices:
        for j in other_indices:
            H_spatial_search_padded[i,j] = 0
            H_spatial_search_padded[j,i] = 0

    pauli_op = SparsePauliOp.from_operator(H_spatial_search_padded)
    
    # Compute number of gates per Trotter step
    if trotter_method == "first_order" or trotter_method == "randomized_first_order":
        circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
    elif trotter_method == "second_order":
        circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
    else:
        raise ValueError(f"{trotter_method} not supported")

    compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry'], optimization_level=3)

    ops = compiled_circuit.count_ops()

    num_single_qubit_gates, num_two_qubit_gates = get_gate_counts(ops)

    assert num_single_qubit_gates + num_two_qubit_gates == sum(ops.values())
    assert num_two_qubit_gates == compiled_circuit.num_nonlocal_gates()


    # Estimate number of Trotter steps required
    r_min, r_max = 1, 10
    while std_bin_trotter_error(H_spatial_search_padded, pauli_op, r_max, trotter_method) > error_tol:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if std_bin_trotter_error(H_spatial_search_padded, pauli_op, r, trotter_method) > error_tol:
            r_min = r
        else:
            r_max = r
    
    return num_two_qubit_gates, r_max

def get_H_spatial_search(n, lamb, gamma, encoding):
    dimension = 2
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
        for i in range(2):
            h_correction[i * n] = 1/2
            if encoding == "unary":
                h_correction[(i + 1) * n - 1] = -1/2
            else:
                h_correction[(i + 1) * n - 1] = -(-1) ** (n+1) / 2
        H_correction = - gamma * sum_h_z(dimension * n, h_correction)
        H_adj = - gamma * sum_x(dimension * n)

        return H_pen + H_oracle + H_correction + H_adj
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
        for i in range(2):
            h[i * n] = 1/2
            h[(i+1) * n - 1] = 1/2

        # Adjacency matrix
        J = np.zeros((dimension * n, dimension * n))
        for i in range(dimension):
            for j in np.arange(i * n, (i + 1) * n - 1):
                J[j,j+1] = 1
        
        H_laplacian = gamma * (sum_h_z(dimension * n, h) - (sum_J_xx(dimension * n, J) + sum_J_yy(dimension * n, J)) / 2)

        return H_laplacian + H_oracle
    else:
        raise ValueError("Encoding not supported")
    
def get_H_pauli_op(n, lamb, gamma, encoding):
    H = []

    dimension = 2
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
        for i in range(2):
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
        for i in range(2):
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

def get_spatial_search_circuit(N, lamb, gamma, r, encoding, trotter_method):
    n = num_qubits_per_dim(N, encoding)
    dimension = 2

    t = 1
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
    TASK_DIR = "spatial_search"

    CURR_DIR = DATA_DIR
    check_and_make_dir(CURR_DIR)
    CURR_DIR = join(CURR_DIR, TASK_DIR)
    check_and_make_dir(CURR_DIR)
    
    print("Resource estimation for spatial search.")
    num_jobs = 16
    print("Number of jobs:", num_jobs)

    dimension = 2
    error_tol = 1e-2
    T = 1
    # trotter_method = "first_order"
    trotter_method = "second_order"
    # trotter_method = "randomized_first_order"

    print(f"Error tolerance: {error_tol : 0.2f}.")
    print(f"Method: {trotter_method}")

    N_vals_binary = np.arange(3, 16)
    binary_trotter_steps = np.zeros(len(N_vals_binary))
    binary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary))
    
    N_vals_unary_bound = np.arange(3, 16)
    unary_trotter_steps_bound = np.zeros(len(N_vals_unary_bound), dtype=int)

    N_vals_unary = np.arange(3, 11)
    unary_trotter_steps = np.zeros(len(N_vals_unary), dtype=int)
    unary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary))

    N_vals_one_hot = np.arange(3, 11)
    one_hot_trotter_steps = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot))

    N_vals_one_hot_bound = np.arange(3, 16)
    one_hot_trotter_steps_bound = np.zeros(len(N_vals_one_hot_bound), dtype=int)


    print("Running resource estimation for standard binary encoding", flush=True)
    for i, N in enumerate(N_vals_binary):

        binary_two_qubit_gate_count_per_trotter_step[i], binary_trotter_steps[i] = get_binary_resource_estimate(N, dimension, error_tol, trotter_method)

        np.savez(join(CURR_DIR, f"std_binary_{trotter_method}.npz"),
                N_vals_binary=N_vals_binary,
                binary_trotter_steps=binary_trotter_steps,
                binary_two_qubit_gate_count_per_trotter_step=binary_two_qubit_gate_count_per_trotter_step)

    # Unary encoding
    encoding = "unary"
    print(f"Running resource estimation for {encoding} encoding", flush=True)
    device = LocalSimulator()
    num_samples = 100

    for i, N in enumerate(N_vals_unary):

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
        codewords = get_codewords_2d(n, encoding, periodic=False)

        lamb = dimension * n
        
        if trotter_method == "first_order" or trotter_method == "randomized_first_order":
            unary_two_qubit_gate_count_per_trotter_step[i] = dimension * (n - 1) + 1
        elif trotter_method == "second_order":
            unary_two_qubit_gate_count_per_trotter_step[i] = 2 * dimension * (n - 1) + 1
        else:
            raise ValueError(f"{trotter_method} not supported")
        
        H_ebd = get_H_spatial_search(n, lamb, gamma, encoding)

        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        while estimate_trotter_error(N, H_ebd, get_spatial_search_circuit(N, lamb, gamma, r_max, encoding, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
            r_max *= 2
        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error(N, H_ebd, get_spatial_search_circuit(N, lamb, gamma, r, encoding, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
                r_min = r
            else:
                r_max = r
        
        unary_trotter_steps[i] = r

        # Save data
        np.savez(join(CURR_DIR, f"unary_{trotter_method}.npz"),
                 N_vals_unary=N_vals_unary[:i+1],
                 unary_trotter_steps=unary_trotter_steps[:i+1],
                 unary_two_qubit_gate_count_per_trotter_step=unary_two_qubit_gate_count_per_trotter_step[:i+1])
        
        
        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)
    
    # Unary encoding
    print("Running resource estimation for unary encoding with analytical bound", flush=True)
    encoding = "unary"

    for i, N in enumerate(N_vals_unary_bound):
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

        lamb = dimension * n

        # Use bound to get Trotter number
        unary_trotter_steps_bound[i] = get_trotter_number(get_H_pauli_op(n, lamb, gamma, encoding), T, error_tol, trotter_method)

        # Save data
        np.savez(join(CURR_DIR, f"unary_{trotter_method}_bound.npz"),
                 N_vals_unary_bound=N_vals_unary_bound[:i+1],
                 unary_trotter_steps_bound=unary_trotter_steps_bound[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

    # One-hot encoding
    encoding = "one-hot"
    print(f"Running resource estimation for {encoding} encoding", flush=True)
    device = LocalSimulator()
    num_samples = 100

    for i, N in enumerate(N_vals_one_hot):

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
        codewords = get_codewords_2d(n, encoding, periodic=False)

        lamb = dimension * n
        
        if trotter_method == "first_order" or trotter_method == "randomized_first_order":
            one_hot_two_qubit_gate_count_per_trotter_step[i] = 2 * dimension * (n - 1) + 1
        elif trotter_method == "second_order":
            one_hot_two_qubit_gate_count_per_trotter_step[i] = 4 * dimension * (n - 1) + 1
        else:
            raise ValueError(f"{trotter_method} not supported")
        
        H_ebd = get_H_spatial_search(n, lamb, gamma, encoding)

        # Estimate number of Trotter steps required
        r_min, r_max = 1, 10
        while estimate_trotter_error(N, H_ebd, get_spatial_search_circuit(N, lamb, gamma, r_max, encoding, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
            r_max *= 2
        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if estimate_trotter_error(N, H_ebd, get_spatial_search_circuit(N, lamb, gamma, r, encoding, trotter_method), dimension, encoding, codewords, device, num_samples, num_jobs) > error_tol:
                r_min = r
            else:
                r_max = r
        
        one_hot_trotter_steps[i] = r

        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}.npz"),
                 N_vals_one_hot=N_vals_one_hot[:i+1],
                 one_hot_trotter_steps=one_hot_trotter_steps[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step=one_hot_two_qubit_gate_count_per_trotter_step[:i+1])
        
        
        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

    # One hot encoding
    print("Running resource estimation for one-hot encoding with analytical bound", flush=True)
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

        lamb = dimension * n

        # Use bound to get Trotter number
        one_hot_trotter_steps_bound[i] = get_trotter_number(get_H_pauli_op(n, lamb, gamma, encoding), T, error_tol, trotter_method)

        # Save data
        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}_bound.npz"),
                 N_vals_one_hot_bound=N_vals_one_hot_bound[:i+1],
                 one_hot_trotter_steps_bound=one_hot_trotter_steps_bound[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)