from ml.linearRegressor import linReg
from ml.logisticRegressor import logicreg
import numpy as np
from time import perf_counter
import pandas as pd

# linear regression
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
l.fit(X, y, lr=0.15, epochs=5, grad='miniBatchMomentum', batch=80)
end = perf_counter()

print(f"slope: {l.slope}")
print(f"intersection: {l.intersect}")
print(f"time taken: {end - start}")


# logistic regression
df = pd.read_csv('iris.csv')

X = df.drop('species', axis=1).to_numpy()
y = df['species']
y = np.array([1 if i == 'setosa' else 0 for i in y])

y = np.reshape(y, (len(y), 1))

X_train, X_test = np.vstack(( X[:30], X[50:100] )), np.vstack(( X[20:50], X[100:150] ))
y_train, y_test = np.vstack(( y[:30], y[50:100] )), np.vstack(( y[20:50], y[100:150] ))

model = logicreg()
model.fit(X_train, y_train, 0.5)

y_pred = model.predict(X_test)

print(y_pred)