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
fname = "iris.csv"
dir_load = f"{path}\dataset\{fname}"
data = pd.read_csv(dir_load, index_col='Id')
print(data.describe())


# 2.1 #################################################################################################
# проведем однофакторный дисперсионный анализ для SepalLengthCm и SepalWidthCm
# по категоризированной переменной Species. Выделим 2 таблицы

# для SepalLengthCm
# sepal_length_frame = pd.DataFrame(
#     np.hstack(
#         np.array(
#             np.split(data['SepalLengthCm'], 3)
#         )[:, :, np.newaxis]
#     ),
#     columns=data['Species'].unique()
# )
#
# fig, ax = plt.subplots()
# intervals = stats.t.interval(0.95,
#                              sepal_length_frame.shape[0]-1,
#                              loc=np.mean(sepal_length_frame),
#                              scale=stats.sem(sepal_length_frame))
# # 95% доверительные интервалы для среднего (золотой цвет)
# ax.errorbar(sepal_length_frame.mean(),
#             sepal_length_frame.columns,
#             xerr=(intervals[1] - intervals[0]) / 2,
#             fmt='o',
#             ecolor='gold',
#             elinewidth=30,
#             markersize=0)
# # линейная диаграмма рассеяния
# sns.stripplot(data=sepal_length_frame, orient='h')
# fig.legend([f"mean = {np.around(np.mean(sepal_length_frame['Iris-setosa']), 2)}",
#             f"mean = {np.around(np.mean(sepal_length_frame['Iris-versicolor']), 2)}",
#             f"mean = {np.around(np.mean(sepal_length_frame['Iris-virginica']), 2)}"])
# ax.set_xlabel(f"{data.columns[0]}")
# ax.set_title("gold is 95% confidence interval for mean")
# fig.tight_layout()
#
#
# # для SepalWidthCm
# sepal_width_frame = pd.DataFrame(
#     np.hstack(
#         np.array(
#             np.split(data['SepalWidthCm'], 3)
#         )[:, :, np.newaxis]
#     ),
#     columns=data['Species'].unique()
# )
#
# fig, ax = plt.subplots()
# intervals = stats.t.interval(0.95,
#                              sepal_width_frame.shape[0]-1,
#                              loc=np.mean(sepal_width_frame),
#                              scale=stats.sem(sepal_width_frame))
# # 95% доверительные интервалы для среднего (золотой цвет)
# ax.errorbar(sepal_width_frame.mean(),
#             sepal_width_frame.columns,
#             xerr=(intervals[1] - intervals[0]) / 2,
#             fmt='o',
#             ecolor='gold',
#             elinewidth=30,
#             markersize=0)
# # линейная диаграмма рассеяния
# sns.stripplot(data=sepal_width_frame, orient='h')
# fig.legend([f"mean = {np.around(np.mean(sepal_width_frame['Iris-setosa']), 2)}",
#             f"mean = {np.around(np.mean(sepal_width_frame['Iris-versicolor']), 2)}",
#             f"mean = {np.around(np.mean(sepal_width_frame['Iris-virginica']), 2)}"])
# ax.set_xlabel(f"{data.columns[1]}")
# ax.set_title("gold is 95% confidence interval for mean")
# fig.tight_layout()
#
#
# # 2.2 #################################################################################################
# # построим boxplot для отображения основных характеристик (медиана, квартили, мин, макс, выбросы)
#
# # для SepalLengthCm
# fig, ax = plt.subplots()
# ax.boxplot(np.flip(sepal_length_frame.values), vert=False)
# ax.set_xlabel(data.columns[0])
# ax.set_yticklabels(sepal_length_frame.columns[::-1])
# fig.tight_layout()
#
#
# # для SepalWidthCm
# fig, ax = plt.subplots()
# ax.boxplot(np.flip(sepal_width_frame.values), vert=False)
# ax.set_xlabel(data.columns[1])
# ax.set_yticklabels(sepal_width_frame.columns[::-1])
# fig.tight_layout()


################################################################################################################


data_reshape = pd.concat(
    [
        data[['SepalLengthCm', 'Species']]
            .rename(columns={'SepalLengthCm': 'Sepal'})
            .assign(type='SepalLengthCm'),

        data[['SepalWidthCm', 'Species']]
            .rename(columns={'SepalWidthCm': 'Sepal'})
            .assign(type='SepalWidthCm')
    ],
    axis=0,
    ignore_index=True
)

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
print(data_reshape)
sns.swarmplot(x='Sepal',
              y='Species',
              hue='type',
              data=data_reshape,
              dodge=True,
              linewidth=1.2,
              ax=ax)

sns.boxplot(x="Sepal",
            y="Species",
            hue='type',
            data=data_reshape,
            whis=np.inf,
            boxprops=dict(alpha=.6),
            ax=ax)

# same result:
# sns.catplot(x='Sepal',
#             y='Species',
#             hue='type',
#             data=data_reshape,
#             kind='swarm',
#             linewidth=1,
#             dodge=True,
#             ax=ax)
#
# sns.catplot(x='Sepal',
#             y='Species',
#             hue='type',
#             data=data_reshape,
#             kind='box',
#             whis=np.inf,
#             boxprops=dict(alpha=.6),
#             ax=ax)

fig.tight_layout()

fig.savefig('conclusions\\task2.png')

