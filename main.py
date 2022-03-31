import numpy as np
from pprint import pprint
import pandas as pd
from matplotlib import pyplot as plt
from patsy.highlevel import dmatrices
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# read the data
data = pd.read_csv('./HR_comma_sep.csv')
# print first 10 rows
print(data.head(10))
print(data.dtypes)
print(data.shape)

# relationship between turnover and salary --> related!
q = pd.crosstab(data['salary'], data['left'])
print(q)
pd.crosstab(data['salary'], data['left']).plot(kind='bar')
plt.show()

print(q.sum(1))
q.div(q.sum(1), axis=0).plot(kind='bar', stacked=True)
plt.show()

# based on above diagram, it shows that turnover is related to salary

# relationship between turnover and satisfaction --> related!
data[data.left == 0].satisfaction_level.hist()
plt.show()
data[data.left == 1].satisfaction_level.hist()
plt.show()

# relationship between turnover and last_evaluation --> related
data[data.left == 0].last_evaluation.hist()
plt.show()
data[data.left == 1].last_evaluation.hist()
plt.show()

# relationship between turnover and number_project --> related!
print("=== number_project ===")
q = pd.crosstab(data['number_project'], data['left'])
print(q)
pd.crosstab(data['number_project'], data['left']).plot(kind='bar')
plt.show()

# relationship between turnover and average_monthly_hours --> related!
print("=== average_monthly_hours ===")
q = pd.crosstab(data['average_monthly_hours'], data['left'])
print(q)
pd.crosstab(data['average_monthly_hours'], data['left']).plot(kind='bar')
plt.show()

# relationship between turnover and time_spend_company --> related!
print("=== time_spend_company ===")
q = pd.crosstab(data['time_spend_company'], data['left'])
print(q)
pd.crosstab(data['time_spend_company'], data['left']).plot(kind='bar')
plt.show()

# relationship between turnover and Work_accident --> related!
print("=== Work_accident ===")
q = pd.crosstab(data['Work_accident'], data['left'])
print(q)
pd.crosstab(data['Work_accident'], data['left']).plot(kind='bar')
plt.show()

# relationship between turnover and promotion_last_5years --> related!
print("=== promotion_last_5years ===")
q = pd.crosstab(data['promotion_last_5years'], data['left'])
print(q)
pd.crosstab(data['promotion_last_5years'], data['left']).plot(kind='bar')
plt.show()

print("=== salary ===")
print(data['salary'].value_counts())

print("=== sales ===")
print(data['sales'].value_counts())

# model training
model=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling='1', l1_ratio=None,
                   n_jobs=None, max_iter=10000, multi_class='auto', penalty='l2', random_state=None, solver='lbfgs',
                   tol=0.0001, verbose=0, warm_start=False)

# col name ~ feature1 + feature2 + feature3
# C(salary) --> category: low, mid, high salary
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_monthly_hours+time_spend_company+'
                 'Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')

X = X.rename(columns={
    'C(sales)[T.RandD]': 'Department: Random',
    'C(sales)[T.accounting]': 'Department: Accounting',
    'C(sales)[T.hr]': 'Department: HR',
    'C(sales)[T.management]': 'Department: Management',
    'C(sales)[T.marketing]': 'Department: Marketing',
    'C(sales)[T.product_mng]': 'Department: Product_Management',
    'C(sales)[T.sales]': 'Department: Sales',
    'C(sales)[T.support]': 'Department： Support',
    'C(sales)[T.technical]': 'Department: Technical',
    'C(salary)[T.low]': 'Salary: Low',
    'C(salary)[T.medium]': 'Salary: Medium'
})
y = np.ravel(y)  # change y to np 1 D array
print(y)
pprint(X.head)
print("y.shape: ", y.shape)
print("X.shape: ", X.shape)

model.fit(X, y)
print(pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_)))))
print(model.score(X, y))


# training data : test data = 7 : 3
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression(max_iter=10000)
model2.fit(Xtrain, ytrain)
pred = model2.predict(Xtest)
print("pred2: ", pred)
pred_proba = model2.predict_proba(Xtest)
print("pred2 proba: ", pred_proba)

metrics.accuracy_score(ytest, pred)

print("=== classification report ===")
print(metrics.classification_report(ytest, pred))

print("=== cross validation ===")
print(cross_val_score(LogisticRegression(max_iter=10000), X, y, scoring='accuracy', cv=10))

print("=== confusion matrix ===")
print(metrics.confusion_matrix(ytest, pred))
