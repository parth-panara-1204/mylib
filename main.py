from ml.linearRegressor import linReg
import numpy as np
from time import perf_counter

X = np.random.rand(5000,10)
m = np.array([[4],
                [8],
                [2.6],
                [100],
                [-1],
                [7.2],
                [2.5],
                [1.5],
                [6.2],
                [6.2]])

y = np.matmul(X, m) - 100

start = perf_counter()
l = linReg()
l.fit(X, y, lr=0.15, epochs=5, grad='NAG', batch=80)
end = perf_counter()

print(f"slope: {l.slope}")
print(f"intersection: {l.intersect}")
print(f"time taken: {end - start}")