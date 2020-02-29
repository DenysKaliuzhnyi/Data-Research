# Калюжный Денис Владимирович

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sns.set()
path = os.getcwd()
fname = "titanic.csv"
dir_load = f"{path}\dataset\{fname}"
data = pd.read_csv(dir_load, index_col="PassengerId")


# 3.1 #################################################################################################

# alived_female = data[(data['Survived'] == 1) & (data['Sex'] == 'female')]
# print(alived_female)


# female_survived = data.query("Survived == 1 and Sex == 'female'").Name
# find = female_survived.str.find('Mrs.')
# find[find==-1] = female_survived.str.find('Miss.')
# find.drop(find[find==-1].index, inplace=True)
# print(female_survived.str.find('Mrs.', find))

# 3.2 #################################################################################################

fig, ax = plt.subplots()
surv = data.query("Survived == 1").Pclass.value_counts(sort=False)
died = data.query("Survived ==  0").Pclass.value_counts(sort=False)
pd.DataFrame(dict(survived=surv, died=died), index=[1, 2, 3]).plot.bar(rot=False, stacked=True, cmap='RdYlGn_r', ax=ax)
ax.set(xlabel='Pclass', ylabel='Count of people')
fig.tight_layout()
fig.savefig('conclusions\\task3.png')

