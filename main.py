from ml.linearRegressor import linReg
import numpy as np

X = np.random.rand(10000,10)
m = np.array([[4],
                [8],
                [2.6],
                [100],
                [-1],
                [7.2],
                [2.5],
                [0.01],
                [6.2],
                [6.2]])

y = np.matmul(X, m) - 0.001

l = linReg()
l.fit(X, y)

print(f"slope: {l.slope}")
print(f"intersection: {l.intersect}")