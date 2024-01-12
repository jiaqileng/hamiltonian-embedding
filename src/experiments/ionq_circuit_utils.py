import numpy as np
from braket.circuits import Circuit
from qiskit import QuantumCircuit

import requests
import json
import sys
from os import getenv
from os.path import join, dirname
sys.path.append(join(dirname(__file__), ".."))
from utils import num_qubits_per_dim
from dotenv import load_dotenv

load_dotenv()
IONQ_API_KEY = getenv('IONQ_API_KEY')
if IONQ_API_KEY == None:
    raise Exception("No API key found.")

# Sends job and returns job_id
def send_job(job):
    headers = {
        "Authorization": f"apiKey {IONQ_API_KEY}",
        "Content-Type": "application/json"
    }
    req = requests.post("https://api.ionq.co/v0.3/jobs", json=job, headers=headers)
    try:
        job_id = json.loads(req.content)['id']
    except KeyError:
        print(req.content)
        raise KeyError(f"Error sending job. Error message: {req.content['''message''']}")
    return job_id

def get_ionq_single_job_result(job_id, codewords, use_error_mitigation=False):
    # Get job
    print("Getting job:", job_id)

    headers = {
        "Authorization": f"apiKey {IONQ_API_KEY}"
    }

    req = requests.get(f"https://api.ionq.co/v0.3/jobs/{job_id}", headers=headers)
    

    status = json.loads(req.content)['status']
    print(f"Job status: {status}")
    if status == "completed":
        
        headers = {
            "Authorization": f"apiKey {IONQ_API_KEY}"
        }
        if use_error_mitigation:
            params = {"sharpen": "true"} # plural voting aggregation
        else:
            params = {"sharpen": "false"}
        req = requests.get(f"https://api.ionq.co/v0.3/jobs/{job_id}/results", headers=headers, params=params)
        results = json.loads(req.content)
        if "error" in results.keys():
            raise AttributeError(results["message"])
        else:
            freq = np.zeros(len(codewords))
            for i in range(len(codewords)):
                if str(codewords[i]) in results.keys():
                    freq[i] = results[str(codewords[i])]
            return freq
    else:
        raise FileNotFoundError("Job not completed")
    
def get_results(job_ids_path, num_time_points, codewords, use_error_mitigation=False):
    with open(job_ids_path, "r") as f:
        job_ids = json.load(f)
        f.close()

    freq = np.zeros((num_time_points, len(codewords)))
    for i, job_id in enumerate(job_ids):

        freq[i] = get_ionq_single_job_result(job_id, codewords, use_error_mitigation)

    return freq

def cancel_job(job_id):
    print("Cancelling job:", job_id)

    headers = {
        "Authorization": f"apiKey {IONQ_API_KEY}"
    }
    r = requests.put(f"https://api.ionq.co/v0.3/jobs/{job_id}/status/cancel", headers=headers)
    return r


def get_hadamard(target):
    return {
        "gate": "h",
        "target": target
    }

def get_cnot(control, target):
    return {
        "gate": "cnot",
        "control": control,
        "target": target
    }

def get_rx(phase, target):
    # phase measured in radians
    return {
        "gate": "rx",
        "rotation": phase,
        "target": target
    }

def get_ry(phase, target):
    # phase measured in radians
    return {
        "gate": "ry",
        "rotation": phase,
        "target": target
    }

def get_rz(phase, target):
    # phase measured in radians
    return {
        "gate": "rz",
        "rotation": phase,
        "target": target
    }

def get_rxx(phase, targets):
    # phase measured in radians
    return {
        "gate": "xx",
        "rotation": phase,
        "targets": targets
    }

def get_ryy(phase, targets):
    # phase measured in radians
    return {
        "gate": "yy",
        "rotation": phase,
        "targets": targets
    }

def get_rzz(phase, targets):
    # phase measured in radians
    return {
        "gate": "zz",
        "rotation": phase,
        "targets": targets
    }

def get_rxy(phase, targets):
    # phase measured in radians
    return {
        "gate": "xy",
        "rotation": phase,
        "targets": targets
    }

def get_gpi(phase, target):
    # phase measured in turns
    return {
        "gate": "gpi",
        "phase": phase,
        "target": target
    }

def get_gpi2(phase, target):
    # phase measured in turns
    return {
        "gate": "gpi2",
        "phase": phase,
        "target": target
    }

def get_ms(phases, angle, targets):
    # assume angle is between 0 and 1
    if 0 <= angle <= 0.25:
        return {
            "gate": "ms",
            "phases": phases,
            "angle": angle,
            "targets": targets
        }
    elif 0.75 <= angle <= 1:
        return {
            "gate": "ms",
            "phases": [phases[0], (phases[1] + 0.5) % 1],
            "angle": 1-angle,
            "targets": targets
        }
    else:
        raise ValueError(f"Angle is {angle}, must be between 0 and 0.25 or 0.75 and 1 (use two gates instead)")
    
def get_hadamard_layer(n, dimension):
    instructions = []
    for i in range(n * dimension):
        instructions.append(get_hadamard(i))
    return instructions

def get_native_circuit(num_qubits, instructions):

    # phase stored in turns
    qubit_phase=[0] * num_qubits
    op_list=[]

    for op in instructions:
        if op["gate"] == "h":
            # Hadamard = GPi2(0.25) @ Z, where Z is Pauli-Z rotation
            qubit_phase[op["target"]] -= 0.5
            qubit_phase[op["target"]] %= 1
            op_list.append(get_gpi2((qubit_phase[op["target"]] + 0.25) % 1, op["target"]))
                
        elif op["gate"] == "cnot":
            # Hadamard on control
            qubit_phase[op["control"]] -= 0.5
            qubit_phase[op["control"]] %= 1
            op_list.append(get_gpi2((qubit_phase[op["control"]] + 0.25) % 1, op["control"]))
        
            # XX rotation
            op_list.append(get_ms([qubit_phase[op["control"]], qubit_phase[op["target"]]], 0.75, [op["control"], op["target"]]))

            # Hadamard on control
            qubit_phase[op["control"]] -= 0.5
            qubit_phase[op["control"]] %= 1
            op_list.append(get_gpi2((qubit_phase[op["control"]] + 0.25) % 1, op["control"]))

            # Rz on control
            qubit_phase[op["control"]] -= 0.25
            qubit_phase[op["control"]] %= 1

            # Rx on target
            op_list.append(get_gpi2((qubit_phase[op["target"]] + 0) % 1, op["target"]))

                
        elif op["gate"] == "rz":
            qubit_phase[op["target"]] -= op["rotation"] / (2 * np.pi)
            qubit_phase[op["target"]] %= 1

        elif op["gate"] == "ry":
            if abs(op["rotation"]) > 1e-5:
                if abs(op["rotation"] / (2 * np.pi) - 0.25) < 1e-6:
                    op_list.append(get_gpi2((qubit_phase[op["target"]] + 0.25) % 1, op["target"]))
                elif abs(op["rotation"] / (2 * np.pi) + 0.25) < 1e-6:
                    op_list.append(get_gpi2((qubit_phase[op["target"]] + 0.75) % 1, op["target"]))
                elif abs(op["rotation"] / (2 * np.pi) - 0.5) < 1e-6:
                    op_list.append(get_gpi((qubit_phase[op["target"]] + 0.25) % 1, op["target"]))
                elif abs(op["rotation"] / (2 * np.pi) + 0.5) < 1e-6:
                    op_list.append(get_gpi((qubit_phase[op["target"]] + 0.75) % 1, op["target"]))
                else:
                    # Basis change and do virtual Z rotation
                    op_list.append(get_gpi2((qubit_phase[op["target"]] + 0) % 1, op["target"]))
                    qubit_phase[op["target"]] -= op["rotation"] / (2 * np.pi)
                    qubit_phase[op["target"]] %= 1
                    op_list.append(get_gpi2((qubit_phase[op["target"]] + 0.5) % 1, op["target"]))

        elif op["gate"] == "rx":
            if abs(op["rotation"]) > 1e-5:
                if abs(op["rotation"] / (2 * np.pi) - 0.25) < 1e-6:
                    op_list.append(get_gpi2((qubit_phase[op["target"]] + 0) % 1, op["target"]))
                elif abs(op["rotation"] / (2 * np.pi) + 0.25) < 1e-6:
                    op_list.append(get_gpi2((qubit_phase[op["target"]] + 0.5) % 1, op["target"]))
                elif abs(op["rotation"] / (2 * np.pi) - 0.5) < 1e-6:
                    op_list.append(get_gpi((qubit_phase[op["target"]] + 0) % 1, op["target"]))
                elif abs(op["rotation"] / (2 * np.pi) + 0.5) < 1e-6:
                    op_list.append(get_gpi((qubit_phase[op["target"]] + 0.5) % 1, op["target"]))
                else:
                    op_list.append(get_gpi2((qubit_phase[op["target"]] + 0.75) % 1, op["target"]))
                    qubit_phase[op["target"]] -= op["rotation"] / (2 * np.pi) 
                    qubit_phase[op["target"]] %= 1
                    op_list.append(get_gpi2((qubit_phase[op["target"]] + 0.25) % 1, op["target"]))
            
        elif op["gate"] == "xx":
            if np.abs(op["rotation"]) > 1e-5:
                if (op["rotation"] / (2 * np.pi)) % 1 <= 0.25 or (op["rotation"] / (2 * np.pi)) % 1 >= 0.75:
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], qubit_phase[op["targets"][1]]], (op["rotation"] / (2 * np.pi)) % 1, op["targets"]))
                elif 0.25 <= (op["rotation"] / (2 * np.pi)) % 1 <= 0.5:
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], qubit_phase[op["targets"][1]]], (((op["rotation"] / (2 * np.pi)) % 1) / 2) % 1, op["targets"]))
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], qubit_phase[op["targets"][1]]], (((op["rotation"] / (2 * np.pi)) % 1) / 2) % 1, op["targets"]))
                elif 0.5 <= (op["rotation"] / (2 * np.pi)) % 1 <= 0.75:
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], qubit_phase[op["targets"][1]]], (((op["rotation"] / (2 * np.pi)) % 1) / 2 - 0.5) % 1, op["targets"]))
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], qubit_phase[op["targets"][1]]], (((op["rotation"] / (2 * np.pi)) % 1) / 2 - 0.5) % 1, op["targets"]))
                else:
                    raise ValueError(f"Rotation angle is {op['rotation']}, should be between 0 and 1")

        elif op["gate"] == "yy":
            if np.abs(op["rotation"]) > 1e-5:
                if (op["rotation"] / (2 * np.pi)) % 1 <= 0.25 or (op["rotation"] / (2 * np.pi)) % 1 >= 0.75:
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (op["rotation"] / (2 * np.pi)) % 1, op["targets"]))
                elif 0.25 <= (op["rotation"] / (2 * np.pi)) % 1 <= 0.5:
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2) % 1, op["targets"]))
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2) % 1, op["targets"]))
                elif 0.5 <= (op["rotation"] / (2 * np.pi)) % 1 <= 0.75:
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2 - 0.5) % 1, op["targets"]))
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2 - 0.5) % 1, op["targets"]))
                else:
                    raise ValueError(f"Rotation angle is {op['rotation']}, should be between 0 and 1")
            
        elif op["gate"] == "xy":
            if np.abs(op["rotation"]) > 1e-5:
                if (op["rotation"] / (2 * np.pi)) % 1 <= 0.25 or (op["rotation"] / (2 * np.pi)) % 1 >= 0.75:
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], (qubit_phase[op["targets"][1]] + 0.25) % 1], (op["rotation"] / (2 * np.pi)) % 1, op["targets"]))
                elif 0.25 <= (op["rotation"] / (2 * np.pi)) % 1 <= 0.5:
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2) % 1, op["targets"]))
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2) % 1, op["targets"]))
                elif 0.5 <= (op["rotation"] / (2 * np.pi)) % 1 <= 0.75:
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2 - 0.5) % 1, op["targets"]))
                    op_list.append(get_ms([qubit_phase[op["targets"][0]], (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2 - 0.5) % 1, op["targets"]))
                else:
                    raise ValueError(f"Rotation angle is {op['rotation']}, should be between 0 and 1")
                    
        elif op["gate"] == "zz":
            if np.abs(op["rotation"]) > 1e-5:

                # Rotate to YY basis 
                op_list.append(get_gpi2((qubit_phase[op["targets"][0]] + 0.5) % 1, op["targets"][0]))
                op_list.append(get_gpi2((qubit_phase[op["targets"][1]] + 0.5) % 1, op["targets"][1]))

                # Apply YY rotation
                if (op["rotation"] / (2 * np.pi)) % 1 <= 0.25 or (op["rotation"] / (2 * np.pi)) % 1 >= 0.75:
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (op["rotation"] / (2 * np.pi)) % 1, op["targets"]))
                elif 0.25 <= (op["rotation"] / (2 * np.pi)) % 1 <= 0.5:
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2) % 1, op["targets"]))
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2) % 1, op["targets"]))
                elif 0.5 <= (op["rotation"] / (2 * np.pi)) % 1 <= 0.75:
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2 - 0.5) % 1, op["targets"]))
                    op_list.append(get_ms([(qubit_phase[op["targets"][0]] + 0.25) % 1, (qubit_phase[op["targets"][1]] + 0.25) % 1], (((op["rotation"] / (2 * np.pi)) % 1) / 2 - 0.5) % 1, op["targets"]))

                # Rotate back
                op_list.append(get_gpi2(qubit_phase[op["targets"][0]], op["targets"][0]))
                op_list.append(get_gpi2(qubit_phase[op["targets"][1]], op["targets"][1]))

            # phase stored in radians
        elif op["gate"] == "gpi":
            op_list.append(get_gpi(qubit_phase[op["target"]] + op["phase"], op["target"]))

        elif op["gate"] == "gpi2":
            op_list.append(get_gpi2(qubit_phase[op["target"]] + op["phase"], op["target"]))

        elif op["gate"] == "ms":
            op_list.append(get_ms([qubit_phase[op["targets"][0]] + op["phases"][0], qubit_phase[op["targets"][1]] + op["phases"][1]], op["angle"], op["targets"]))

        else:
            raise TypeError(f"Gate is {op['''gate''']}, not Rx, Ry, Rz, XX, YY, ZZ, or native")

    return op_list, qubit_phase

def get_native_gate_counts(instructions):
    '''Returns gate count for circuit with IonQ native gates'''
    one_qubit_gate_count = 0
    two_qubit_gate_count = 0

    for op in instructions:
        if op["gate"] == "gpi":
            one_qubit_gate_count += 1
        elif op["gate"] == "gpi2":
            one_qubit_gate_count += 1
        elif op["gate"] == "ms":
            two_qubit_gate_count += 1
        else:
            raise TypeError(f"Gate is {op['''gate''']}, not native")
    return one_qubit_gate_count, two_qubit_gate_count

def get_qiskit_circuit(num_qubits, instructions):

    circuit = QuantumCircuit(num_qubits)

    for op in instructions:
        if op["gate"] == "h":
            circuit.h(op["target"])
            
        elif op["gate"] == "rz":
            circuit.rz(op["rotation"], op["target"])

        elif op["gate"] == "ry":
            if abs(op["rotation"]) > 1e-5:
                circuit.ry(op["rotation"], op["target"])

        elif op["gate"] == "rx":
            if abs(op["rotation"]) > 1e-5:
                circuit.rx(op["rotation"], op["target"])
        
        elif op["gate"] == "xx":
            if np.abs(op["rotation"]) > 1e-5:
                circuit.rxx(op["rotation"], op["targets"][0], op["targets"][1])

        elif op["gate"] == "yy":
                circuit.ryy(op["rotation"], op["targets"][0], op["targets"][1])
                
        elif op["gate"] == "zz":
            if np.abs(op["rotation"]) > 1e-5:
                circuit.rzz(op["rotation"], op["targets"][0], op["targets"][1])
        else:
            raise TypeError(f"Gate is {op['''gate''']}, not H, Rx, Ry, Rz, XX, YY, ZZ")


    return circuit

def get_circuit_from_qiskit(qiskit_circuit):
    instructions = []

    for item in qiskit_circuit.data:
        instruction, qubits = item[0], item[1]

        if instruction.name == "rz":
            instructions.append(get_rz(instruction.params[0], int(qiskit_circuit.find_bit(qubits[0]).index)))

        elif instruction.name == "ry":
            instructions.append(get_ry(instruction.params[0], int(qiskit_circuit.find_bit(qubits[0]).index)))

        elif instruction.name == "rx":
            instructions.append(get_rx(instruction.params[0], int(qiskit_circuit.find_bit(qubits[0]).index)))
        
        elif instruction.name == "rxx":
            instructions.append(get_rxx(instruction.params[0], [int(qiskit_circuit.find_bit(qubits[0]).index), int(qiskit_circuit.find_bit(qubits[1]).index)]))

        elif instruction.name == "ryy":
            instructions.append(get_ryy(instruction.params[0], [int(qiskit_circuit.find_bit(qubits[0]).index), int(qiskit_circuit.find_bit(qubits[1]).index)]))
        else:
            raise TypeError(f"Gate is {instruction.name}, not Rx, Ry, Rz, XX, YY")
    return instructions

def get_circuit_from_braket(braket_circuit):

    instructions = []
    for instruction in braket_circuit.instructions:
        if instruction.operator.name == "H":
            instructions.append(get_hadamard(int(instruction.target[0].real)))
        elif instruction.operator.name == "CNot":
            instructions.append(get_cnot(int(instruction.target[0].real), int(instruction.target[1].real)))
        elif instruction.operator.name == "X":
            instructions.append(get_rx(np.pi, int(instruction.target[0].real)))
        elif instruction.operator.name == "Rz":
            instructions.append(get_rz(instruction.operator.angle, int(instruction.target[0].real)))

        elif instruction.operator.name == "Ry":
            instructions.append(get_ry(instruction.operator.angle, int(instruction.target[0].real)))

        elif instruction.operator.name == "Rx":
            instructions.append(get_rx(instruction.operator.angle, int(instruction.target[0].real)))
        
        elif instruction.operator.name == "XX":
            instructions.append(get_rxx(instruction.operator.angle, [int(instruction.target[0].real), int(instruction.target[1].real)]))

        elif instruction.operator.name == "YY":
            instructions.append(get_ryy(instruction.operator.angle, [int(instruction.target[0].real), int(instruction.target[1].real)]))
            
        elif instruction.operator.name == "ZZ":
            instructions.append(get_rzz(instruction.operator.angle, [int(instruction.target[0].real), int(instruction.target[1].real)]))
        else:
            raise TypeError(f"Gate is {instruction.operator.name}, not Rx, Ry, Rz, XX, YY")
    return instructions

def get_braket_native_circuit(instructions):
    # Requires all gates to be native gates
    circuit = Circuit()

    for op in instructions:

        if op["gate"] == "gpi":
            circuit.gpi(op["target"], op["phase"] * 2 * np.pi)
            
        elif op["gate"] == "gpi2":
            circuit.gpi2(op["target"], op["phase"] * 2 * np.pi)

        elif op["gate"] == "ms":
            circuit.ms(op["targets"][0], op["targets"][1], op["phases"][0] * 2 * np.pi, op["phases"][1] * 2 * np.pi, op["angle"] * 2 * np.pi)
        else:
            raise TypeError(f"Gate is {op['''gate''']}, not GPi, GPi2, or MS")
    
    return circuit


def get_ionq_job_json(name, N, dimension, shots, device, encoding, instructions, use_native_gates=True, use_error_mitigation=False):
    # assert use_native_gates == True
    n = num_qubits_per_dim(N, encoding)
    job = {}
    job["lang"] = "json"
    job["name"] = name
    job["shots"] = shots
    job["target"] = device

    input = {}
    input["format"] = "ionq.circuit.v0"

    input["qubits"] = n * dimension
    
    if use_native_gates:
        input["gateset"] = "native"
        input["circuit"], _ = get_native_circuit(n * dimension, instructions)
    else:
        input["gateset"] = "qis"
        input["circuit"] = instructions

    job["input"] = input

    if use_error_mitigation:
        job["error_mitigation"] = {"debias": True}
    
    return job

# Old state preparation
def state_prep_braket(N, dimension, amplitudes, encoding):
    n = num_qubits_per_dim(N, encoding)
    circuit = Circuit()

    for i in range(dimension):

        if encoding == "unary" or encoding == "antiferromagnetic":

            assert len(amplitudes) == N

            circuit.ry(i * n, 2 * np.arccos(amplitudes[0]))

            for k in np.arange(0, n-1):
                a = amplitudes[k+1] / np.linalg.norm(amplitudes[k+1:], ord=2)
                # Y rotation controlled on previous qubit
                # Controlled Y rotation (basis change on control qubit)
                circuit.rx(i * n + k, -0.25 * (2 * np.pi))
                circuit.yy(i * n + k, i * n + k + 1, -np.arccos(a))
                circuit.rx(i * n + k, 0.25 * (2 * np.pi))
                circuit.ry(i * n + k + 1, np.arccos(a))

            # Just map from unary to antiferromagnetic encoding
            if encoding == "antiferromagnetic":
                for k in range(n):
                    if k % 2 == 1:
                        circuit.x(i * n + k)
            

        elif encoding == "one-hot":
            # Start from 1000...0
            circuit.x(i * n + 0)
            # Y rotation
            circuit.ry(i * n + 1, 2 * np.arccos(amplitudes[0]))
            # CNOT
            circuit.cnot(i * n + 1, i * n)
            
            for k in np.arange(1, N-1):
                a = amplitudes[k] / np.linalg.norm(amplitudes[k:], ord=2)
                # Y rotation controlled on previous qubit
                circuit.rx(i * n + k, -np.pi/2)
                circuit.yy(i * n + k, i * n + k + 1, -np.arccos(a))
                circuit.rx(i * n + k, np.pi/2)
                circuit.ry(i * n + k + 1, np.arccos(a))

                # CNOT
                circuit.cnot(i * n + k + 1, i * n + k)
        else:
            raise ValueError("Encoding not supported")
    
    return circuit

def state_prep_circuit(N, dimension, amplitudes, encoding):
    

    n = num_qubits_per_dim(N, encoding)
    instructions = []

    for i in range(dimension):

        if encoding == "unary" or encoding == "antiferromagnetic":
            assert np.all(np.isreal(amplitudes)) and np.all(amplitudes >= 0), "Only supports positive real valued amplitudes"
            assert len(amplitudes) == N

            instructions.append(get_ry(2 * np.arccos(amplitudes[0]), i * n))

            for k in np.arange(0, n-1):
                if np.linalg.norm(amplitudes[k+1:]) > 0:
                    a = amplitudes[k+1] / np.linalg.norm(amplitudes[k+1:], ord=2)
                    # Y rotation controlled on previous qubit
                    # Controlled Y rotation (basis change on control qubit)

                    instructions.append(get_rx(-0.25 * (2 * np.pi), i * n + k))
                    instructions.append(get_ryy(-np.arccos(a), [i * n + k, i * n + k + 1]))
                    instructions.append(get_rx(0.25 * (2 * np.pi), i * n + k))
                    instructions.append(get_ry(np.arccos(a), i * n + k + 1))

            # Just map from unary to antiferromagnetic encoding
            if encoding == "antiferromagnetic":
                for k in range(n):
                    if k % 2 == 1:
                        instructions.append(get_rx(np.pi, i * n + k))

        elif encoding == "one-hot":
            amplitudes_abs_val = np.abs(amplitudes)

            # Start from 000...001 (first qubit on the right)
            instructions.append(get_rx(np.pi, i * n))

            instructions += state_prep_one_hot_aux(n, i * n, amplitudes_abs_val)

            if not np.all(amplitudes >= 0):
                for k in range(N):
                    theta = np.angle(amplitudes[k])
                    instructions.append(get_rz(i * n + k, theta))

        else:
            raise ValueError("Encoding not supported")
    
    return instructions

def state_prep_one_hot_aux(n, starting_index, amplitudes):
    assert np.all(np.isreal(amplitudes)) and np.all(amplitudes >= 0)
    assert len(amplitudes) == n
    instructions = []

    if n > 1:
        amplitudes_left = amplitudes[:n//2]
        amplitudes_right = amplitudes[n//2:]
        a = np.linalg.norm(amplitudes_left)

        if np.linalg.norm(amplitudes_right) > 0:
            instructions.append(get_rxy(np.arccos(a), [starting_index, starting_index + n//2]))
            instructions.append(get_rxy(-np.arccos(a), [starting_index + n//2, starting_index]))

        if np.linalg.norm(amplitudes_left) > 0:
            instructions += state_prep_one_hot_aux(n//2, starting_index, amplitudes_left / np.linalg.norm(amplitudes_left))
        if np.linalg.norm(amplitudes_right) > 0:
            instructions += state_prep_one_hot_aux((n+1)//2, starting_index + n//2, amplitudes_right / np.linalg.norm(amplitudes_right))

    return instructions