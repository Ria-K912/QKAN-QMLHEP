import torch
import torch.nn as nn
from qsvt_sinepoly import QSVT
from LCU import quantum_lcu_block 
from quantum_summation import quantum_sum_block
from SplineKANlayer import KANLayer 

class QuantumKANRegressor(nn.Module):
    def __init__(self, num_features, degree=5):
        super().__init__()
        self.num_features = num_features
        self.degree = degree

        self.pqc = QSVT(wires=1, degree=degree, depth=2) 
        self.lcu_weights = nn.Parameter(torch.rand(num_features, degree)) 
        self.kan = KANLayer(in_features=num_features, out_features=1)

    def forward(self, X):
        all_outputs = []

        for xi in X:  # xi shape: (num_features,)
            qsvt_all = [self.pqc(xi[j]) for j in range(self.num_features)]  
            qsvt_all = torch.stack(qsvt_all)  # shape: (F, P)
            lcu_weighted = [quantum_lcu_block(qsvt_all[f], self.lcu_weights[f]) for f in range(self.num_features)]
            summed = [quantum_sum_block(torch.stack([val])) for val in lcu_weighted]
            all_outputs.append(torch.stack(summed))  # shape: (F,)

        return self.kan(torch.stack(all_outputs))  # shape: (B, 1)
