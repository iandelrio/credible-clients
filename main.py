import numpy as np
from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

from model import CreditModel


def main():
    # Load data from disk and split into training and validation sets.
    data = np.loadtxt('data/credit-data.csv', dtype=np.int, delimiter=',', skiprows=1)
    X, y = data[:, 1:-1], data[:, -1]

    # print(y)
    # Transform all 0s to -1s in y to make logistic regression loss function and prediction easier
    y[y == 0] = -1

    # print(X_train.shape[0]) # number of training examples
    # print(X_.shape[1]) # number of features
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Fit the model against the training data.
    model = CreditModel()

    y_hat_cv = cross_val_predict(model, X_train, y_train, cv=10)
    print(y_hat_cv)

    model.fit(X_train, y_train)

    # Predict against test data and ensure `y_hat` returns ints.
    y_hat = model.predict(X_test)
    y_hat = np.rint(np.squeeze(y_hat)).astype(int)
    assert len(y_hat) == len(X_test)

    # Train scikit-learn's L2-regularized logistic regression and predict on X_test
    model2 = LogisticRegression(fit_intercept=False)
    model2.fit(X_train, y_train)
    y_hat2 = model2.predict(X_test)

    # Compare our output with scikit-learn's model's output
    print(np.array_equal(y_hat, y_hat2))


    # Print out accuracy/precision/recall scores for our output and scikit-learn's model
    print("Our accuracy:        {:06.3f}%".format(100 * accuracy_score(y_test, y_hat)))
    print("sklearn's accuracy:  {:06.3f}%".format(100 * accuracy_score(y_test, y_hat2)))
    print("\nOur precision:       {:06.3f}%".format(100 * precision_score(y_test, y_hat)))
    print("sklearn's precision: {:06.3f}%".format(100 * precision_score(y_test, y_hat2)))
    print("\nOur recall:          {:06.3f}%".format(100 * recall_score(y_test, y_hat)))
    print("sklearn's recall:    {:06.3f}%".format(100 * recall_score(y_test, y_hat2)))


if __name__ == '__main__':
    main()
