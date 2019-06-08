import numpy as np


class logistic(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learn_rate=0.01, num_iters=5000):
        num_train, num_feature = X.shape
        # init the weight
        self.W = 0.001 * np.random.randn(num_feature, 1).reshape((-1, 1))
        loss = []

        for i in range(num_iters):
            error, dW = self.compute_loss(X, y)
            self.W += -learn_rate * dW

            loss.append(error)
            if i % 200 == 0:
                print
                'i=%d,error=%f' % (i, error)
        return loss

    def compute_loss(self, X, y):
        num_train = X.shape[0]
        h = self.output(X)
        loss = -np.sum((y * np.log(h) + (1 - y) * np.log((1 - h))))
        loss = loss / num_train

        dW = X.T.dot((h - y)) / num_train

        return loss, dW

    def output(self, X):
        g = np.dot(X, self.W)
        return self.sigmod(g)

    def sigmod(self, X):
        return 1 / (1 + np.exp(-X))

    def predict(self, X_test):
        h = self.output(X_test)
        y_pred = np.where(h >= 0.5, 1, 0)
        return y_pred