import pandas as pd
import numpy as np


fname = 'datasets\\рейтинг.xlsx'
data = pd.read_excel(fname, skiprows=1, usecols=[1, 2, 3, 4, 5, 6], header=None)
data.columns = ['name', 'rating', 'urban', 'sport', 'science', 'total']
data = pd.concat([data, pd.Series([np.NaN], index=[data.index[-1]+1])])
isna = data['total'].isna()
delim = data[isna].index
group_names = ['pm1', 'pm2', 'pm3', 'pm4', 'pm5', 'pm6',
               'sa1', 'sa2', 'sa3', 'sa4', 'sa5', 'sa6',
               'inf1', 'inf2', 'inf3', 'inf4', 'inf5', 'inf6',
               'pi1', 'pi2', 'pi3', 'pi4', 'pi5', 'pi6']
data['group'] = np.NaN
i = 0
for (f, t) in zip(delim[:-1], delim[1:]):
    if t - f > 2:
        if group_names[i] == 'inf6':
            i += 1
        data.loc[f: t, 'group'] = group_names[i]
        i += 1
data.dropna(subset=['total'], inplace=True)
data.fillna(0, inplace=True)
data.reset_index(drop=True, inplace=True)
data = data[['group', 'name', 'rating', 'urban', 'sport', 'science', 'total']]

data.to_csv('datasets\\rating2.csv', index=False)

