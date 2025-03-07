import numpy as np

class linReg():
    def __init__(self):
        self.msg = 'hello from linreg'
        self.slope = 0.0
        self.intersect = 0.0

    def fit(self, X, y, lr=0.01, epochs=1000):
        self.slope = np.zeros((X.shape[1], 1))

        y_pred = np.matmul(X, self.slope) + self.intersect
        # print(y_pred)
        self.gradDec(self, y_pred, X, y, lr, epochs)

    @staticmethod
    def gradDec(self, y_pred, X, y, lr, epochs):
        n = X.shape[0]
        
        i = 0
        while i < epochs:
            cost = 1/n * np.matmul((1/2 * (y_pred - y)**2).T, np.ones((n,1))) 

            # for least square cost
            slopeGrad = 1/n * np.matmul( X.T, (y_pred - y))
            intersectGrad = 1/n *np.matmul( (y_pred - y).T, np.ones((n, 1)) )

            self.slope = self.slope - lr * slopeGrad
            self.intersect = self.intersect - lr * intersectGrad

            y_pred = np.matmul(X, self.slope) + self.intersect
            i += 1