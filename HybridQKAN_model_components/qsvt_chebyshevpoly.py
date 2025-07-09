from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pennylane as qml
import numpy as np

target_polys = [
    [0, 1],
    [-1, 0, 2],
    [0, -3, 0, 4],
    [1, 0, -8, 0, 8]
]
iris = load_iris()
X = np.tile(iris.data, (250, 1))[:1000]
y = np.tile(iris.target, 250)[:1000]

qsvt_outputs = {}
for poly_idx, target_poly in enumerate(target_polys):
    for i in range(1000):
        x_raw = X[i]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_norm = scaler.fit_transform(x_raw.reshape(-1, 1)).flatten()
        A = np.diag(x_norm)
        wire_order = list(range(3))
        try:
            U_A = qml.matrix(qml.qsvt, wire_order=wire_order)(
                A, target_poly, encoding_wires=wire_order, block_encoding="embedding"
            )
            qsvt_result = np.real(np.diagonal(U_A))[:4]
            qsvt_outputs[(i, poly_idx)] = qsvt_result
        except Exception:
            qsvt_outputs[(i, poly_idx)] = None