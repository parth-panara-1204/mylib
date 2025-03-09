import numpy as np

class logicreg():
    def __init__(self, slope=None, intersect=None, threshold=None):
        self.msg = 'hello from logivreg'
        self.slope = slope
        self.intersect = intersect
        self.threshold = threshold

    def fit(self, X, y, threshold):
        self.slope = np.zeros((X.shape[1], 1))
        self.intersect = 0.0
        self.threshold = threshold

        self._gradDec(X, y, lr=0.01, epochs=100)

    def _gradDec(self, X, y, lr, epochs):
        for i in range(epochs):
            y_pred = np.matmul(X, self.slope) + self.intersect
            y_pred = 1 / (1 + np.exp(-y_pred))

            loss = -np.mean(y*np.log(y_pred + 0.00001) + (1-y)*np.log(1-y_pred + 0.00001))
            
            slopeGrad = 1/X.shape[0] * np.matmul(X.T, (y_pred - y))
            slopeIntersect = 1/X.shape[0] * np.sum(y_pred - y)

            self.slope = self.slope - lr*slopeGrad
            self.intersect = self.intersect - lr*slopeIntersect

    def predict(self, X):
        pred = np.matmul(X, self.slope) + self.intersect
        pred = 1 / (1 + np.exp(-pred))

        return [1  if i > self.threshold else 0 for i in pred.T[0]]