import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
sns.set()
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas as pd
path = os.getcwd()

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

group_names = ['pm1', 'pm2', 'pm3', 'pm4', 'pm5', 'pm6',
               'sa1', 'sa2', 'sa3', 'sa4', 'sa5', 'sa6',
               'inf1', 'inf2', 'inf3', 'inf4', 'inf5', 'inf6',
               'pi1', 'pi2', 'pi3', 'pi4', 'pi5', 'pi6']

data = pd.read_csv('datasets\\rating1.csv')


"""absolute frequency hist"""
fig, ax = plt.subplots()
sns.distplot(data['total'],
             kde=False,
             bins=8,
             hist_kws=dict(range=(60, 100), alpha=1),
             ax=ax)
ax.set(xticks=np.arange(60, 100+1, 10), xlim=(60, 100))
fig.savefig("images1\\total_absolute.png")


"""relative frequency hist"""
fig, ax = plt.subplots()
sns.distplot(data['total'],
             kde=False,
             norm_hist=True,
             bins=8,
             hist_kws=dict(range=(60, 100), alpha=1),
             ax=ax)
ax.set(xticks=np.arange(60, 100+1, 10), xlim=(60, 100))
fig.savefig("images1\\total_relative.png")


"""density"""
fig, ax = plt.subplots()
sns.distplot(data['total'],
             kde=True,
             hist=False,
             rug=True,
             bins=8,
             kde_kws=dict(shade=True),
             hist_kws=dict(range=(60, 100), alpha=1),
             ax=ax)
ax.set(xticks=np.arange(60, 100+1, 10), xlim=(60, 100))
fig.savefig("images1\\total_density.png")


"""absolute frequency hist for each group"""
fig, ax = plt.subplots(4, 6, figsize=(18, 12), sharey='row', sharex='col')
for (cell, group_name) in zip(ax.flat, group_names):
    group_total = data.loc[data['group'] == group_name, 'total']
    sns.distplot(group_total,
                 kde=False,
                 bins=8,
                 hist_kws=dict(range=(60, 100), alpha=1),
                 axlabel=False,
                 ax=cell)
    cell.set(title=group_name,
             xticks=np.arange(60, 100+1, 10),
             yticks=np.arange(0, 25+1, 5),
             ylim=(0, 25),
             xlim=(60, 100))
    leg = cell.legend([group_total.shape[0]], loc=2)
    for item in leg.legendHandles:
        item.set_visible(False)
plt.tight_layout()
fig.savefig("images1\group_absolute.png")


"""relative frequency hist for each group"""
fig, ax = plt.subplots(4, 6, figsize=(18, 12), sharey='row', sharex='col')
for (cell, group_name) in zip(ax.flat, group_names):
    group_total = data.loc[data['group'] == group_name, 'total']
    sns.distplot(group_total,
                 kde=False,
                 norm_hist=True,
                 bins=8,
                 hist_kws=dict(range=(60, 100), alpha=1),
                 axlabel=False,
                 ax=cell)
    cell.set(title=group_name,
             xticks=np.arange(60, 100+1, 10),
             yticks=np.arange(0, 33+1, 9)/300,
             ylim=(0, 33/300),
             xlim=(60, 100))
    leg = cell.legend([group_total.shape[0]], loc=2)
    for item in leg.legendHandles:
        item.set_visible(False)
plt.tight_layout()
fig.savefig("images1\group_relative.png")


"""density for each group"""
fig, ax = plt.subplots(4, 6, figsize=(18, 12), sharey='row', sharex='col')
for (cell, group_name) in zip(ax.flat, group_names):
    group_total = data.loc[data['group'] == group_name, 'total']
    sns.distplot(group_total,
                 kde=True,
                 hist=False,
                 rug=True,
                 bins=8,
                 kde_kws=dict(shade=True),
                 hist_kws=dict(range=(60, 100), alpha=1),
                 axlabel=False,
                 ax=cell)
    cell.set(title=group_name,
             xticks=np.arange(60, 100+1, 10),
             yticks=np.arange(0, 36+1, 9)/300,
             ylim=(0, 36/300),
             xlim=(60, 100))
    leg = cell.legend([group_total.shape[0]], loc=2)
    for item in leg.legendHandles:
        item.set_visible(False)
plt.tight_layout()
fig.savefig("images1\group_density.png")


"""extra points"""
active = data.query("urban + sport + science > 0")
active_proj = active[["urban", "sport", "science"]].reset_index(drop=True)
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
tcc = (active_proj != 0).sum().rename('total count contribution')
tcc.plot.pie(ax=ax[0, 0])
tcc.sort_values().plot.barh(color=[plt.rcParams['axes.prop_cycle'].by_key()['color'][1], 'g', 'b'], ax=ax[0, 1])
ax[0, 1].set_xlabel(tcc.name)
ax[0, 1].set(xlim=(0, 70))

tvc = active_proj.sum().rename('total value contribution')
tvc.plot.pie(ax=ax[1, 0])
tvc.sort_values().plot.barh(color=[plt.rcParams['axes.prop_cycle'].by_key()['color'][1], 'g', 'b'], ax=ax[1, 1])
ax[1, 1].set_xlabel(tvc.name)
ax[1, 1].set(xlim=(0, 70))
plt.tight_layout()
fig.savefig("images1\extra_points.png")


# fig.text(0.515, 0.04, 'rating', va='center', ha='center', fontsize=18)
# fig.text(0.04, 0.5, 'count of people', va='center', ha='center', rotation='vertical', fontsize=18)


"""extra points weight"""
fig, ax = plt.subplots()
sns.relplot(x='rating',
            y='total',
            hue=data[["urban", "sport", "science"]].sum(axis=1),
            data=data,
            palette="ch:r=-.5,l=.75",
            ax=ax)
fig.savefig("images1\\extra_weight.png")


"""violin plot density"""
sns.catplot(x="group", y="total", kind="box", data=data, height=12, aspect=2)
plt.savefig("images1\\box.png", dpi=500)


"""box plot density"""
sns.catplot(x="group", y="total", kind="violin", inner="stick", data=data, height=12, aspect=2)
plt.savefig("images1\\violin.png", dpi=700)
