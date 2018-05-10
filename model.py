import numpy as np


class CreditModel:
    def __init__(self, regularization = 1, numIterations = 100):
        """
        Instantiates the model object, creating class variables if needed.
        """
        self.regularization = regularization
        self.numIterations = numIterations

    def calcfg(self, w, X_train, y_train):

        yXw = y_train * X_train.dot(w)

        regTermF = (self.regularization / 2) * (w.T @ w)

        regTermG = self.regularization * w

        # Calculate the loss
        f = np.sum(np.log(1. + np.exp(-yXw))) + regTermF

        # Calculate the gradient value
        res = - y_train / (1. + np.exp(yXw))
        g = X_train.T.dot(res) + regTermG

        return f, g

    def fit(self, X_train, y_train):
        """
        Fits the model based on the given `X_train` and `y_train`.

        You should somehow manipulate and store this data to your model class
        so that you can make predictions on new testing data later on.
        """
        n, d = X_train.shape
        self.w = np.ones(d)

        f, g = self.calcfg(self.w, X_train, y_train)
        iterationsSoFar = 1

        alpha = 1.
        while True:

            w_new = self.w - alpha * g

            f_new, g_new = self.calcfg(w_new, X_train, y_train)

            iterationsSoFar += 1

            # Update parameters/function/gradient
            self.w = w_new
            g = g_new

            if iterationsSoFar >= self.numIterations:
                break


    def predict(self, X_test):
        """
        Returns `y_hat`, a prediction for a given `X_test` after fitting.

        You should make use of the data that you stored/computed in the
        fitting phase to make your prediction on this new testing data.
        """

        # y_rand = np.random.randint(2, size=len(X_test))
        # y_rand[y_rand == 0] = -1
        # return y_rand

        return np.sign(X_test @ self.w).astype(int)
