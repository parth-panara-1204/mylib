import numpy as np

class linReg():
    def __init__(self):
        self.msg = 'hello from linreg'
        self.slope = 0.0
        self.intersect = 0.0

    def fit(self, X, y, lr=0.5, epochs=1000):
        self.slope = np.zeros((X.shape[1], 1))
        self.intersect = 0.0

        self._gradDec(X, y, lr, epochs)

    def _gradDec(self, X, y, lr, epochs):
        n = X.shape[0]
        
        i = 0
        while i < epochs:
            y_pred = np.matmul(X, self.slope) + self.intersect

            # for least square cost
            slopeGrad = 1/n * np.matmul( X.T, (y_pred - y))
            intersectGrad = 1/n * np.sum(y_pred - y)

            self.slope = self.slope - lr * slopeGrad
            self.intersect = self.intersect - lr * intersectGrad

            y_pred = np.matmul(X, self.slope) + self.intersect
            i += 1