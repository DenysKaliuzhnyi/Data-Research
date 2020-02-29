import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
sns.set()


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


fname = 'datasets/uData.csv'
data = pd.read_csv(fname,
                   header=2,
                   na_values=['?', 'не открываются файлы', 'нет роботы', 'нет работы', '#VALUE!'],
                   decimal=',')
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
data.columns = ['surname', 'name', 'total', 'critical_thinking', 'math', 'python1', 'python2', 'python3', 'python_total']
print(data.describe(), end='\n\n')

me = data.query("surname == 'Калюжний'")
print(me, end='\n\n')
########################################################################################################################


# custom pairplot
num_data = data[data.columns[2:]]
n = num_data.shape[1]
fig, ax = plt.subplots(nrows=n, ncols=n, sharex='col', sharey='row', figsize=(16, 16), dpi=400)
for i in range(n):
    ax[i, 0].set_ylabel(num_data.columns[i])
    ax[-1, i].set_xlabel(num_data.columns[i])
    for j in range(i, n):
        color = np.where(data.surname == 'Калюжний', 'red', 'grey')
        size = np.where(data.surname == 'Калюжний', 50, 10)
        ax[i, j].scatter(num_data.iloc[:, j],
                         num_data.iloc[:, i],
                         c=color,
                         s=size)
        ax[j, i].scatter(num_data.iloc[:, i],
                         num_data.iloc[:, j],
                         c=color,
                         s=size)
fig.tight_layout()
fig.savefig(f"conclusions\custom_pairplot.png")


# pairplot
sns.pairplot(data, kind="scatter")
plt.savefig('conclusions\pairplot.png')


# good at math but python is bad
print(data.query("python_total <= 50 & math > 0").sort_values(by='math', ascending=False))


# linear model info
mod = sm.OLS(data.total, sm.add_constant(data.python_total))
fit = mod.fit()
print(fit.summary())


# custom linear model
model = LinearRegression()
fit = model.fit(data.python_total[:, np.newaxis], data.total)
determination = model.score(data.python_total[:, np.newaxis], data.total)
intercept = fit.intercept_
coef = fit.coef_[0]
f = lambda y: intercept + coef*y
f = np.vectorize(f)
predictions = f(data.python_total)
rng = np.linspace(np.min(data.python_total), np.max(data.python_total), data.python_total.size)
regrline = f(rng)
fig, ax = plt.subplots()
ax.set_xlabel('python total')
ax.set_ylabel('total')
ax.set_title(
    f"Linear Regression\n"
    f"intercept={np.around(intercept, 2)}, "
    f"coef={np.around(coef, 2)}, "
    f"determination={np.around(determination, 2)}")
ax.scatter(data.python_total, data.total)
ax.plot(rng, regrline, c='red')
fig.savefig(f"conclusions\\custom_regression.png")


# linear model
reg = sns.lmplot('python_total', 'total', data=data)
plt.savefig(f"conclusions\\regression.png")


# names
fig, ax = plt.subplots(figsize=(16, 8))
names = data.groupby('name').size().sort_values(ascending=False)
names.plot.bar(ax=ax)
fig.tight_layout()
plt.savefig(f"conclusions\\names.png")


# python
fig, ax = plt.subplots(figsize=(24, 8))
data[['python1', 'python2', 'python3']].plot.bar(stacked=True, ax=ax)
fig.tight_layout()
plt.savefig(f"conclusions\\python.png")
