import numpy as np
from scipy.sparse import csc_matrix, kron, identity
from scipy.optimize import minimize
from functools import reduce
from os import mkdir
from os.path import exists

PAULI_X = csc_matrix(np.array([[0, 1], [1, 0]]))
PAULI_Y = csc_matrix(np.array([[0,-1j],[1j,0]]))
PAULI_Z = csc_matrix(np.array([[1, 0], [0, -1]]))
IDENTITY = csc_matrix(np.eye(2))
NUMBER = csc_matrix(np.array([[0, 0], [0, 1]]))

def tensor(op_list : list):
    return reduce(kron, op_list, 1).tocsc()

def uniform_superposition(n : int):
    return np.ones(2 ** n, dtype=np.complex64) / np.sqrt(2 ** n)

def sum_x(n : int):
    '''
    Returns `\sum_i \sigma_x^{(i)}` where `\sigma_x^{(i)}` 
    is the Pauli-x operator on the ith qubit.
    '''
    assert n > 0
    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), PAULI_X, identity(dims[i], format='csr')])
        
    return res

def sum_y(n : int):
    '''
    Returns `\sum_i \sigma_x^{(i)}` where `\sigma_x^{(i)}` 
    is the Pauli-x operator on the ith qubit.
    '''
    assert n > 0
    dims = [2 ** i for i in range(n)]
#     return np.sum([tensor([identity(dims[i], format='csr'), PAULI_Y, identity(dims[n-i-1], format='csr')]) for i in range(n)])
    
    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-1-i], format='csr'), PAULI_Y, identity(dims[i], format='csr')])
        
    return res

def sum_h_x(n : int, h : np.ndarray):
    '''
    Returns `\sum_i h_i \sigma_x^{(i)}` where `\sigma_x^{(i)}` 
    is the Pauli-y operator on the ith qubit.
    '''
    assert n > 0
    assert len(h) == n

    dims = [2 ** i for i in range(n)]
    

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), h[i] * PAULI_X, identity(dims[i], format='csr')])
        
    return res

def sum_h_y(n : int, h : np.ndarray):
    '''
    Returns `\sum_i h_i \sigma_y^{(i)}` where `\sigma_y^{(i)}` 
    is the Pauli-y operator on the ith qubit.
    '''
    assert n > 0
    assert len(h) == n

    dims = [2 ** i for i in range(n)]
    

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), h[i] * PAULI_Y, identity(dims[i], format='csr')])
        
    return res

def sum_delta_n(n : int, delta : np.ndarray):
    '''
    Returns `\sum_i \Delta_i \hat{n}^{(i)}` where `\hat{n}^{(i)}` 
    is the number operator on the ith qubit.
    '''
    assert n > 0
    assert len(delta) == n

    dims = [2 ** i for i in range(n)]
    
#     return np.sum([tensor([identity(dims[i], format='csr'), delta[i] * NUMBER, identity(dims[n-i-1], format='csr')]) for i in range(n)])
    
    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), delta[i] * NUMBER, identity(dims[i], format='csr')])
        
    return res

def sum_h_z(n : int, h : np.ndarray):
    '''
    Returns `\sum_i h_i \sigma_z^{(i)}` where `\sigma_z^{(i)}` 
    is the Pauli-z operator on the ith qubit.
    '''
    assert n > 0
    assert len(h) == n

    dims = [2 ** i for i in range(n)]
    
    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), h[i] * PAULI_Z, identity(dims[i], format='csr')])
        
    return res

def sum_V_nn(n : int, V : np.ndarray):
    '''
    Returns `\sum_{i>j} V_{i,j} \sigma_z^{(i)} \sigma_z^{(j)}` where `\sigma_z^{(i)}` 
    is the Pauli-z operator on the ith qubit.
    '''
    assert n > 0
    assert V.shape == (n,n)

    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(i):
            res += (V[i,j] + V[j,i]) * tensor([identity(dims[n-i-1], format='csr'), NUMBER, identity(dims[i-j-1], format='csr'), NUMBER, identity(dims[j], format='csr')])
            
    return res

def sum_J_zz(n : int, J : np.ndarray):
    '''
    Returns `\sum_{i,j} J_{i,j} \sigma_z^{(i)} \sigma_z^{(j)}` where `\sigma_z^{(i)}` 
    is the Pauli-z operator on the ith qubit.
    '''
    assert n > 0
    assert J.shape == (n,n)

    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(i):
            res += (J[i,j] + J[j,i]) * tensor([identity(dims[n-i-1], format='csr'), PAULI_Z, identity(dims[i-j-1], format='csr'), PAULI_Z, identity(dims[j], format='csr')])
            
    return res

def sum_J_xx(n : int, J : np.ndarray):
    '''
    Returns `\sum_{i,j} J_{i,j} \sigma_x^{(i)} \sigma_x^{(j)}` where `\sigma_x^{(i)}` 
    is the Pauli-x operator on the ith qubit.
    '''
    assert n > 0
    assert J.shape == (n,n)

    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(i):
            res += (J[i,j] + J[j,i]) * tensor([identity(dims[n-i-1], format='csr'), PAULI_X, identity(dims[i-j-1], format='csr'), PAULI_X, identity(dims[j], format='csr')])
            
    return res

def sum_J_yy(n : int, J : np.ndarray):
    '''
    Returns `\sum_{i,j} J_{i,j} \sigma_y^{(i)} \sigma_y^{(j)}` where `\sigma_y^{(i)}` 
    is the Pauli-y operator on the ith qubit.
    '''
    assert n > 0
    assert J.shape == (n,n)

    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(i):
            res += (J[i,j] + J[j,i]) * tensor([identity(dims[n-i-1], format='csr'), PAULI_Y, identity(dims[i-j-1], format='csr'), PAULI_Y, identity(dims[j], format='csr')])
            
    return res

def sum_J_zzz(n : int, J : np.ndarray):
    '''
    Returns `\sum_{i,j} J_{i,j} \sigma_z^{(i)} \sigma_z^{(j)}` where `\sigma_z^{(i)}` 
    is the Pauli-z operator on the ith qubit.
    '''
    assert n > 0
    assert J.shape == (n,n,n)

    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(i):
            for k in range(j):
                res += (J[i,j,k] + J[i,k,j] + J[j,i,k] + J[j,k,i] + J[k,i,j] + J[k,j,i]) * tensor([identity(dims[n-i-1], format='csr'), PAULI_Z, identity(dims[i-j-1], format='csr'), PAULI_Z, identity(dims[j-k-1], format='csr'), PAULI_Z, identity(dims[k], format='csr')])
            
    return res

def get_adjacency_lattice(n : int, d : int):
    L_1D = np.zeros((n,n))

    for i in range(n-1):
        L_1D[i, i+1] = 1
        L_1D[i+1, i] = 1
        
    dims = [n ** i for i in range(d)]
    return np.sum([tensor([np.identity(dims[d-i-1]), L_1D, np.identity(dims[i])]) for i in range(d)])

def get_laplacian_lattice(n : int, d : int):
    # returns the negative semidefinite Laplacian for a lattice
    L_1D = np.zeros((n,n))

    for i in range(n-1):
        L_1D[i, i+1] = 1
        L_1D[i+1, i] = 1
    
    for i in range(n):
        L_1D[i,i] = - np.sum(L_1D[i])
        
    dims = [n ** i for i in range(d)]
    return np.sum([tensor([np.identity(dims[d-i-1]), L_1D, np.identity(dims[i])]) for i in range(d)])


def get_adjacency_ring(n : int, d : int):
    L_1D = np.zeros((n,n))

    for i in range(n):
        L_1D[i, (i+1) % n] = 1
        L_1D[(i+1) % n, i] = 1
        
    dims = [n ** i for i in range(d)]
    return np.sum([tensor([np.identity(dims[d-i-1]), L_1D, np.identity(dims[i])]) for i in range(d)])

def get_adjacency_complete(N : int):
    return csc_matrix(np.ones((N, N)))

def scipy_get_optimal_gamma(A, H_potential, gamma_0):
    def get_spectral_gap(gamma_vals):
        if type(gamma_vals) == float:
            H = gamma_vals * A + H_potential
            evals, _ = np.linalg.eigh(H.toarray())
            return evals[-1] - evals[-2]
        else:
            H_vals = [gamma * A + H_potential for gamma in gamma_vals]
            evals_list = [np.linalg.eigh(H.toarray())[0] for H in H_vals]
            gap = np.zeros(len(evals_list))
            for i in range(len(evals_list)):
                evals = evals_list[i]
                gap[i] = evals[-1] - evals[-2]
            return gap
            
    res = minimize(get_spectral_gap, gamma_0, method='nelder-mead',
        options={'xatol': 1e-8, 'disp': False})

    gamma = res.x
    return gamma[0]


def check_and_make_dir(dir_name):
    if not exists(dir_name):
        mkdir(dir_name)

def bitstring_to_int(bitstring):
    '''Takes a bitstring array and returns the decimal value'''
    sum = 0
    for i in range(len(bitstring)):
        sum += bitstring[i] * 2 ** i
    
    return sum

def decode_unary(n, bitstring, periodic=False):
    if periodic == False:
        assert len(bitstring) == n
        flips = 0
        first_flip = 0
        for i in range(n-1):
            if bitstring[i] != bitstring[i+1]:
                flips += 1
                first_flip = i
        
        if flips >= 2 or (bitstring[0] == 0 and flips == 1):
            return -1
        else:
            if flips == 0:
                if bitstring[0] == 0:
                    return 0
                else:
                    return n
            elif flips == 1:
                return first_flip + 1
    else:
        raise NotImplementedError
        
def decode_antiferromagnetic(n, bitstring, periodic=False):
    if periodic == False:
        assert len(bitstring) == n

        flips = 0
        # first index k where bits at indices k and k+1 are the same
        first_ferromagnetic_index = None
        for k in range(n-1):
            if bitstring[k] != bitstring[k+1]:
                flips += 1
            elif first_ferromagnetic_index is None:
                first_ferromagnetic_index = k
        
        if flips < n - 2 or (bitstring[0] == 0 and flips == n - 2):
            return -1
        else:
            if flips == n - 1:
                if bitstring[0] == 0:
                    return 0
                else:
                    return n
            if flips == n - 2:
                return first_ferromagnetic_index + 1
    else:
        raise NotImplementedError

def decode_one_hot(n, bitstring):
    weight = np.sum(bitstring)
    if weight == 1:
        return np.argmax(bitstring)
    else:
        return -1
    
def get_occurrences(N : int, measurements, num_shots : int, dimension : int, encoding, periodic=False):
    occurrences = np.zeros(dimension * [N])
    bad_samples = 0
    for i in range(num_shots):
        bitstring = measurements[i]
        
        if encoding == "unary":
            if periodic == False:
                n = N - 1
                indices = np.zeros(dimension, dtype=int)
                for j in range(dimension):
                    indices[j] = decode_unary(n, bitstring[j * n : (j + 1) * n], periodic)
            else:
                n = N // 2
                indices = np.zeros(dimension, dtype=int)
                raise NotImplementedError
        elif encoding == "antiferromagnetic":
            if periodic == False:

                n = N - 1
                indices = np.zeros(dimension, dtype=int)
                
                for j in range(dimension):
                    
                    indices[j] = decode_antiferromagnetic(n, bitstring[j * n : (j + 1) * n], periodic)
            else:
                n = N // 2
                indices = np.zeros(dimension, dtype=int)
                raise NotImplementedError
            
        elif encoding == "one-hot":
            n = N
            indices = np.zeros(dimension, dtype=int)
            for j in range(dimension):
                indices[j] = decode_one_hot(n, bitstring[j * n : (j + 1) * n])
        if np.prod(indices >= 0):
            occurrences[tuple(indices)] += 1
        else:
            bad_samples += 1
                
    # print(f"Samples in codeword subspace: {int(np.sum(occurrences))} / {num_shots}")

    return occurrences

def get_codewords_1d(n : int, encoding, periodic):
    
    codewords = []

    if encoding == "unary" or encoding == "antiferromagnetic":
        if encoding == "unary":
            bitstring = 0
        elif encoding == "antiferromagnetic":
            bitstring = 0
            for k in range(n):
                if k % 2 == 1:
                    bitstring += 1 << k

        if periodic:
            for i in range(2 * n):
                codewords.append(bitstring)
                bitstring ^= 1 << (i % n)
        else:
            for i in range(n+1):
                codewords.append(bitstring)

                if i < n:
                    bitstring ^= 1 << i
    
    elif encoding == "one-hot":

        bitstring = 1

        for i in range(n):
            codewords.append(bitstring)

            if i < n - 1:
                bitstring ^= (1 << i)
                bitstring ^= (1 << (i+1))
    return codewords

def get_codewords(N : int, dimension: int, encoding, periodic=False):
    '''Returns codewords for a given encoding.'''

    n = num_qubits_per_dim(N, encoding)
    codewords_1d = get_codewords_1d(n, encoding, periodic)
    codewords = []

    indices = np.zeros(dimension, dtype=int)
    for i in range(N ** dimension):
        assert np.all(indices <= N - 1)

        codeword = 0
        for j in range(dimension):
            codeword += (2 ** (j * n)) * codewords_1d[indices[j]]
        codewords.append(codeword)
        
        if i < N ** dimension - 1:
            # Increment indices
            indices[-1] += 1
            for j in np.arange(dimension):
                if (indices[dimension - 1 - j] >= N):
                    indices[dimension - 1 - j - 1] += 1
                    indices[dimension - 1 - j] %= N

    return codewords

def num_qubits_per_dim(N, encoding):
    if encoding == "one-hot":
        return N
    elif encoding == "unary" or encoding == "antiferromagnetic":
        return N - 1
    else:
        raise ValueError("Encoding not supported. Valid encodings: unary, antiferromagnetic, one-hot")
    
def get_bitstrings_1d(N, encoding):
    bitstrings = []

    if encoding == "unary":
        bitstring = (N-1) * ["0"]
        for i in range(N):
            bitstrings.append("".join(bitstring))
            if i < N - 1:
                bitstring[i] = "1"

        return bitstrings
    
    elif encoding == "antiferromagnetic":
        bitstring = []
        for i in range(N-1):
            if i % 2 == 0:
                bitstring.append("0")
            else:
                bitstring.append("1")

        for i in range(N):
            bitstrings.append("".join(bitstring))
            if i < N - 1:
                if i % 2 == 0:
                    bitstring[i] = "1"
                else:
                    bitstring[i] = "0"

        return bitstrings
        
    elif encoding == "one-hot":
        bitstring = N * ["0"]
        for i in range(N):
            bitstring[i] = "1"
            if i > 0:
                bitstring[i-1] = "0"

            bitstrings.append("".join(bitstring))

        return bitstrings
    else:
        return ValueError("Encoding not supported.")

def get_bitstrings(N, dimension, encoding):
    bitstrings_1d = get_bitstrings_1d(N, encoding)
    N = len(bitstrings_1d)

    bitstrings = []
    indices = np.zeros(dimension, dtype=int)
    for i in range(N ** dimension):
        
        assert np.all(indices <= N - 1)
        bitstring = []
        for j in range(dimension):
            bitstring.append(bitstrings_1d[indices[j]])
        bitstrings.append("".join(bitstring))
        
        if i < N ** dimension - 1:
            # Increment indices
            indices[-1] += 1
            for j in np.arange(dimension):
                if (indices[dimension - 1 - j] >= N):
                    indices[dimension - 1 - j - 1] += 1
                    indices[dimension - 1 - j] %= N
                
    return bitstrings