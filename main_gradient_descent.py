from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from patsy import dmatrices
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# read the data
data = pd.read_csv('./HR_comma_sep.csv')

y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_monthly_hours+time_spend_company+'
                 'Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
X = np.asmatrix(X)
y = np.ravel(y)

print("X.shape, y.shape:", X.shape, y.shape)

# Normalize: some feature range is big, some is small. Normalize make them [0, 1]
for i in range(1, X.shape[1]):
    xmin = X[:, i].min()
    xmax = X[:, i].max()
    X[:, i] = (X[:, i] - xmin) / (xmax - xmin)

print(X[:10, :])

# gradient descent
np.random.seed(1)
alpha = 1.
beta = np.random.randn(X.shape[1])
loss_pre = 0.

for T in range(200):
    prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()
    prob_y = list(zip(prob, y))
    # loss function and error rate
    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for (p, y) in prob_y]) / len(y)

    if abs(loss - loss_pre) < 1e-4:
        break
    loss_pre = loss

    error_rate = 0.
    for i in range(len(y)):
        if (prob[i] > 0.5 and y[i] == 0) or (prob[i] < 0.5 and y[i] == 1):
            error_rate += 1
    error_rate /= len(y)

    if T % 5 == 0:
        print('T={}, loss={}, errpr_rate={}'.format(T, loss, error_rate))

    # loss function: gradient about beta
    deriv = np.zeros(X.shape[1])
    for i in range(len(y)):
        deriv += np.array(X[i, :]).ravel() * (prob[i] - y[i])
    deriv /= len(y)

    beta -= alpha * deriv