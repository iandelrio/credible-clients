# credible-clients

This repository contains a small pre-task for potential ML team
members for [UBC Launch Pad](https://www.ubclaunchpad.com).


## Overview

The dataset bundled in this repository contains information about credit card
bill payments, courtesy of the [UCI Machine Learning Repository][UCI]. Your task
is to train a model on this data to predict whether or not a customer will default
on their next bill payment.

Most of the work should be done in [**`model.py`**](model.py). It contains a
barebones model class; your job is to implement the `fit` and `predict` methods,
in whatever way you want (feel free to import any libraries you wish). You can
look at [**`main.py`**](main.py) to see how these methods will be called. Don't
worry about getting "good" results (this dataset is _very tough_ to predict on)
— treat this as an exploratory task!

To run this code, you'll need Python and three libraries: [NumPy], [SciPy],
and [`scikit-learn`]. After invoking **`python main.py`** from your shell of
choice, you should see the model accuracy printed: approximately 50% if you
haven't changed anything, since the provided model predicts completely randomly.

## Instructions

Here are the things you should do:

1. Fork this repo, so we can see your code!
2. Install the required libraries using `pip install -r requirements.txt` (if needed).
3. Ensure you see the model's accuracy/precision/recall scores printed when running `python main.py`.
4. Replace the placeholder code in [`model.py`](model.py) with your own model.
5. Fill in the "write-up" section below in your forked copy of the README.
6. Submit a publicly viewable link to your forked repo in the application form

_Good luck, and have fun with this_! :rocket:


## Write-up

Give a brief summary of the approach you took, and why! Include your model's
accuracy/precision/recall scores as well!

Disclaimer:
Since the instructions specified that most of the work should be done in model.py, I abstained from doing
any pre-processing and feature engineering in main.py, such as one-hot-encoding to get categorical data from
discrete numerical data, and standardizing features.
I also abstained from doing hyperparameter optimization, which I would have done using GridSearchCV.
I also abstained from using cross validation (cross_val_predict) to decrease approximation error,
  and thus to avoid overfitting.

Model
- Since this is a binary classification problem, we will implement logistic regression with L2-regularization
    - The paper “Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?” by
      Fernandez-Delgado et al. [2014] concludes that Random Forests and SVMs are the best
      out-of-the-box classifiers
    - However, we chose Logistic Regression because
        1. it gives us smoother predictions than Random Forests, and
        2. its loss function is easier to minimize than SVM's (logistic loss is smooth,
           whereas hinge loss is not)
    - We will use L2-regularization to regularize our weights
        - We chose L2-regularization over L1-regularization because features selection is unnecessary with
          such a small amount of features (23 features for 22,500 training examples)
        - We will use the Akaike information criterion for regularization, i.e. regularization term = 1, for
          the sake of simplicity.

Optimization
- Optimize using gradient descent
    - Less than 100,000 training samples, so according to [scikit-learn's "Choosing the right estimator" guide](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html),
      no need to do stochastic gradient descent
    - We will do gradient descent with a constant step-size of 1, for simplicity's sake.

Once we have our model
- Compare my model's output with scikit-learn's L2-regularized logistic regression

Model's accuracy: 78.3%
Model's precision: 0.00%
Model's recall: 0.00%


## Data Format

`X_train` and `X_test` contain data of the following form:

| Column(s) | Data |
| :-------: | ---- |
| 0         | Amount of credit given, in dollars |
| 1         | Gender (_1 = male, 2 = female_) |
| 2         | Education (_1 = graduate school; 2 = university; 3 = high school; 4 = others_) |
| 3         | Marital status (_1 = married; 2 = single; 3 = others_) |
| 4         | Age, in years |
| 5–10      | History of past payments over 6 months (_-1 = on-time; 1 = one month late; …_) |
| 11–16     | Amount of previous bill over 6 months, in dollars |
| 17–22     | Amount of previous payment over 6 months, in dollars |

`y_train` and `y_test` contain a `1` if the customer defaulted on their next
payment, and a `0` otherwise.


[UCI]: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
[NumPy]: http://www.numpy.org
[SciPy]: https://www.scipy.org
[`scikit-learn`]: http://scikit-learn.org
