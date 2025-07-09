import pennylane as qml
import torch
import math

def quantum_sum_block(poly_vals):
    """
    Aggregates across P polynomial outputs for one feature using quantum summation.
    Input: poly_vals (tensor): shape (P,)
    Output: scalar tensor
    """
    P = len(poly_vals)
    n_ctrl = math.ceil(math.log2(P))
    wires = list(range(n_ctrl + 1))  # control + 1 target
    dev = qml.device("default.qubit", wires=len(wires))

    uniform = torch.zeros(2**n_ctrl)
    uniform[:P] = 1 / math.sqrt(P)
    uniform = uniform / torch.norm(uniform)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit():
        qml.StatePrep(uniform, wires=wires[:-1])

        for i in range(P):
            ctrl_bin = [int(b) for b in f"{i:0{n_ctrl}b}"]
            qml.ctrl(qml.RY, control=wires[:-1], control_values=ctrl_bin)(2 * poly_vals[i], wires=wires[-1])

        qml.adjoint(qml.StatePrep(uniform, wires=wires[:-1]))

        return qml.expval(qml.PauliZ(wires[-1]))

    return circuit()
