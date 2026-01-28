import numpy as np
import pennylane as qml

def angle_encoding(x, wires, rotation="RY"):
    rot = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}.get(rotation)
    if rot is None:
        raise ValueError("rotation must be one of: 'RX', 'RY', 'RZ'")

    for i, w in enumerate(wires):
        angle = x[i % len(x)]
        rot(angle, wires=w)


def entangle(wires, pattern="chain"):
   
    if len(wires) < 2:
        return

    if pattern == "chain":
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    elif pattern == "all":
        for i in range(len(wires)):
            for j in range(i + 1, len(wires)):
                qml.CNOT(wires=[wires[i], wires[j]])
    else:
        raise ValueError("pattern must be 'chain' or 'all'")


def trainable_layer(W, wires, entanglement="chain"):
    for i, w in enumerate(wires):
        qml.RX(W[i, 0], wires=w)
        qml.RY(W[i, 1], wires=w)
        qml.RZ(W[i, 2], wires=w)

    entangle(wires, pattern=entanglement)


def make_variational_classifier(
    n_qubits=2,
    depth=2,
    reupload=1,
    encoding_rotation="RY",
    entanglement="chain",
    measure_wire=0,
):

    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    if reupload == 1:
        weight_shape = (depth, n_qubits, 3)
    else:
        weight_shape = (reupload, depth, n_qubits, 3)

    @qml.qnode(dev)
    def circuit(x, weights):

        if reupload == 1:
            # Encode
            angle_encoding(x, wires, rotation=encoding_rotation)
            for d in range(depth):
                trainable_layer(weights[d], wires, entanglement=entanglement)

        else:
            # Re-uploading
            for r in range(reupload):
                angle_encoding(x, wires, rotation=encoding_rotation)
                for d in range(depth):
                    trainable_layer(weights[r, d], wires, entanglement=entanglement)
        return qml.expval(qml.PauliZ(measure_wire))

    return circuit, weight_shape



def draw_circuit_example(
    n_qubits=2,
    depth=2,
    reupload=1,
    encoding_rotation="RY",
    entanglement="chain",
):
   
    circuit, weight_shape = make_variational_classifier(
        n_qubits=n_qubits,
        depth=depth,
        reupload=reupload,
        encoding_rotation=encoding_rotation,
        entanglement=entanglement,
    )

    
    x = np.array([0.2, 1.0])
    rng = np.random.default_rng(0)
    weights = 0.01 * rng.standard_normal(size=weight_shape)

    drawer = qml.draw(circuit)
    return drawer(x, weights)
