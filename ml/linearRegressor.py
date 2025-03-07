import numpy as np

class linReg():
    def __init__(self):
        self.msg = 'hello from linreg'
        self.slope = 0.0
        self.intersect = 0.0

    def fit(self, X, y, lr=0.5, epochs=1000, grad='miniBatchMomentum', batch=32, beta=0.9):
        self.slope = np.zeros((X.shape[1], 1))
        self.intersect = 0.0

        self._gradDec(X, y, lr, epochs, grad, batch, beta)

    def _gradDec(self, X, y, lr, epochs, grad, batch, beta):
        n = X.shape[0]
        
        # BATCH GRADIENT
        if grad == 'batch':
            i = 0

            while i < epochs:
                y_pred = np.matmul(X, self.slope) + self.intersect

                # for least square cost
                slopeGrad = 1/n * np.matmul( X.T, (y_pred - y))
                intersectGrad = 1/n * np.sum(y_pred - y)

                self.slope = self.slope - lr * slopeGrad
                self.intersect = self.intersect - lr * intersectGrad

                i += 1

            y_pred = np.matmul(X, self.slope) + self.intersect
            print(np.mean((y_pred - y)**2))

        # MINI-BATCH GRADIENT
        elif grad == 'miniBatch':
            batches = n // batch

            for i in range(epochs):
                for j in range(batches):
                    y_pred = np.matmul(X[j*batch:(j+1)*batch], self.slope) + self.intersect

                    # for least square cost
                    slopeGrad = 1/batch * np.matmul( X[j*batch:(j+1)*batch].T, (y_pred - y[j*batch:(j+1)*batch]))
                    intersectGrad = 1/batch * np.sum(y_pred - y[j*batch:(j+1)*batch])

                    self.slope = self.slope - lr * slopeGrad
                    self.intersect = self.intersect - lr * intersectGrad

        # BATCH GRADIENT WITH MOMENTUM
        elif grad == 'batchMomentum':
            i = 0
            v1, v2 = 0, 0
            while i < epochs:
                y_pred = np.matmul(X, self.slope) + self.intersect

                # for least square cost
                slopeGrad = 1/n * np.matmul( X.T, (y_pred - y))
                intersectGrad = 1/n * np.sum(y_pred - y)

                v1 = (1-lr)*v1 + lr*slopeGrad
                v2 = (1-lr)*v2 + lr*intersectGrad

                self.slope = self.slope - v1
                self.intersect = self.intersect - v2

                i += 1

        # MINI-BATCH GRADIENT WITH MOMENTUM
        elif grad == 'miniBatchMomentum':
            batches = n // batch

            for i in range(epochs):
                v1, v2 = 0, 0
                for j in range(batches):
                    y_pred = np.matmul(X[j*batch:(j+1)*batch], self.slope) + self.intersect

                    # for least square cost
                    slopeGrad = 1/batch * np.matmul( X[j*batch:(j+1)*batch].T, (y_pred - y[j*batch:(j+1)*batch]))
                    intersectGrad = 1/batch * np.sum(y_pred - y[j*batch:(j+1)*batch])

                    v1 = (1-lr)*v1 + lr*slopeGrad
                    v2 = (1-lr)*v2 + lr*intersectGrad

                    self.slope = self.slope - v1
                    self.intersect = self.intersect - v2

            y_pred = np.matmul(X, self.slope) + self.intersect
            print(np.mean((y_pred - y)**2))


        # NESTEROV's ACCELERATED GRADIENT
        elif grad == 'NAG':
            batches = n // batch

            for i in range(epochs):
                v1, v2 = 0, 0
                for j in range(batches):
                    la_slope = self.slope - beta*v1
                    la_intersect = self.intersect - beta*v2

                    y_pred = np.matmul(X[j*batch:(j+1)*batch], la_slope) + la_intersect

                    # for least square cost
                    slopeGrad = 1/batch * np.matmul( X[j*batch:(j+1)*batch].T, (y_pred - y[j*batch:(j+1)*batch]))
                    intersectGrad = 1/batch * np.sum(y_pred - y[j*batch:(j+1)*batch])

                    v1 = beta*v1 + lr*slopeGrad
                    v2 = beta*v2 + lr*intersectGrad

                    self.slope = self.slope - v1
                    self.intersect = self.intersect - v2
