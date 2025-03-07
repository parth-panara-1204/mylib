from ml.linearRegressor import linReg
import numpy as np

X = np.random.rand(10000,1)
m = np.array([[4],
                [8],
                [2.6],
                [100],
                [-1],
                [7.2],
                [2.5],
                [0.001],
                [6.2],
                [6.2]])

y = np.matmul(X, m) - 0.001
# y = X*400 - 0.0001

l = linReg()
l.fit(X, y, 0.5, 1000)

print(f"slope: {l.slope}")
print(f"intersection: {l.intersect}")