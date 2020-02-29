import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
sns.set()
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import pandas as pd
from scipy import stats


path = os.getcwd()
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


group_names = ['pm1', 'pm2', 'pm3', 'pm4', 'pm5', 'pm6',
               'sa1', 'sa2', 'sa3', 'sa4', 'sa5', 'sa6',
               'inf1', 'inf2', 'inf3', 'inf4', 'inf5', 'inf6',
               'pi1', 'pi2', 'pi3', 'pi4', 'pi5', 'pi6']


data1 = pd.read_csv('datasets\\rating1.csv')
data2 = pd.read_csv('datasets\\rating2.csv')

data = pd.merge(data1[['name', 'group', 'total']],
                data2[['name', 'total']],
                on='name',
                suffixes=['1', '2'],
                validate='one_to_one')
data.eval('diff = total2 - total1', inplace=True)


data['year'] = data['group'].str[-1].astype(int)
data['group'] = data['group'].str[:-1]
data = data.merge(pd.get_dummies(data['group']), left_index=True, right_index=True)
data = data[['name', 'inf', 'pi', 'pm',  'sa', 'year', 'total1', 'total2', 'diff']]
print(data)


feature_names = ['inf', 'pi', 'pm',  'sa', 'year', 'total1']
X = data[feature_names].values
y = data['total2'].values


pipe = Pipeline([('features', PolynomialFeatures()),
                 ('model', LinearRegression())])
hyperparams = [
    dict(model=[LinearRegression(n_jobs=-1)],
         features__degree=np.arange(1, 7),
         features__interaction_only=[True, False],
         features__include_bias=[True, False],
         model__fit_intercept=[True, False]),
    dict(model=[Lasso(random_state=0)],
         features__degree=np.arange(1, 7),
         features__interaction_only=[True, False],
         features__include_bias=[True, False],
         model__fit_intercept=[True, False],
         model__alpha=np.arange(1, 5))
    # dict(
    #     model=[RandomForestRegressor(random_state=0, n_jobs=-1)],
    #     model__n_estimators=[13, 15, 18],
    #     model__max_depth=np.arange(1, 3),
    #     model__max_features=np.arange(2, 5),
    # )
]

grid = GridSearchCV(pipe,
                    param_grid=hyperparams,
                    cv=KFold(n_splits=10, shuffle=True, random_state=0),
                    n_jobs=-1,)

grid.fit(X, y)

model = grid.best_estimator_

params = pd.Series(model.named_steps['model'].coef_,
                   index=model.named_steps['features'].get_feature_names(feature_names)).round()
print(params)
print(grid.best_score_, grid.best_params_)



#
# """total comparison"""
# g = sns.relplot(x='total1', y='total2', data=data, linewidth=1)
# g.map(sns.lineplot, x=[60, 100], y=[60, 100], color='m', lw=2)
# g.set_axis_labels('previous', 'current')
# g.fig.tight_layout()
# plt.savefig('comparison_images\\total.png')
# # pearson = stats.pearsonr(data['total1'], data['total2'])
# # spearman = stats.spearmanr(data['total1'], data['total2'])
# # kendall = stats.kendalltau(data['total1'], data['total2'])
# # print(pearson, spearman, kendall, sep='\n')

#
# """group comparison"""
# g = sns.relplot(x='total1',
#                 y='total2',
#                 col='group',
#                 col_wrap=4,
#                 s=130,
#                 linewidth=2,
#                 data=data.iloc[:-1])
# g.map(sns.lineplot, x=[60, 100], y=[60, 100], color='m', lw=2)
# g.set_axis_labels('previous', 'current')
# g.fig.tight_layout()
# plt.savefig('comparison_images\\by_group.png')
#
#
# """residuals"""
# g = sns.FacetGrid(data, height=6)
# g.map(sns.distplot, 'diff', rug=True)
# g.fig.tight_layout()
# plt.savefig('comparison_images\\residuals.png')


# """jointplot"""
# g = sns.jointplot(x='total1',
#                   y='total2',
#                   data=data,
#                   joint_kws=dict(edgecolor='w'))
# g.set_axis_labels('previous', 'current')
# plt.savefig('comparison_images\\jointplot.png')


# """hex jointplot"""
# with sns.axes_style("white"):
#     g = sns.jointplot(x='total1',
#                       y='total2',
#                       data=data,
#                       kind='hex',
#                       space=0)
#     g.set_axis_labels('previous', 'current')
#     plt.savefig("comparison_images\\jointplot_hex.png")
#
#
# """kde jointplot"""
# with sns.axes_style("white"):
#     g = sns.jointplot("total1",
#                       "total2",
#                       data=data,
#                       kind="kde",
#                       space=0,
#                       color="g")
#     g.set_axis_labels('previous', 'current')
#     g.ax_joint.collections[0].set_alpha(0)
#     plt.savefig("comparison_images\\jointplot_kde.png")


plt.show()