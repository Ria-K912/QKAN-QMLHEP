import torch
import torch.nn as nn
from qsvt_sinepoly import QSVT
from LCU import quantum_lcu_block 
from quantum_summation import quantum_sum_block
from SplineKANlayer import KANLayer                 

class QuantumKANClassifier(nn.Module):
    def __init__(self, num_features, degree=5):
        super().__init__()
        self.num_features = num_features
        self.degree = degree

        self.qsvt = QSVT(wires=1, degree=degree, depth=2) 
        self.lcu_weights = nn.Parameter(torch.rand(num_features, degree))  # (F, P)
        self.kan = KANLayer(in_features=num_features, out_features=2)  # Binary classification

    def forward(self, X):
        """
        Input: X of shape (B, F)
        Output: logits of shape (B, 2)
        """
        B = X.size(0)
        feature_outputs = []

        for i in range(B):
            xi = X[i]  
            qsvt_vecs = [self.qsvt(xi[f]) for f in range(self.num_features)]  
            lcu_vals = [quantum_lcu_block(qsvt_vecs[f], self.lcu_weights[f]) for f in range(self.num_features)]
            summed_vals = [quantum_sum_block(torch.stack([val])) for val in lcu_vals] 
            feature_outputs.append(torch.stack(summed_vals))  # shape: (F,)

        features = torch.stack(feature_outputs)  # shape: (B, F)
        return self.kan(features)  # shape: (B, 2)
