import pandas as pd
import numpy as np


fname = 'datasets\\rating.xlsx'
data = pd.read_excel(fname, skiprows=12, usecols=[1, 2, 3, 4, 5, 7])
data.columns = ['name', 'rating', 'urban', 'sport', 'science', 'total']
total_data = data.dropna(how='all')[:-3].fillna(0)

isna = data['name'].isna()
delim = data[isna].index
group_names = ['pm1', 'pm2', 'pm3', 'pm4', 'pm5', 'pm6',
               'sa1', 'sa2', 'sa3', 'sa4', 'sa5', 'sa6',
               'inf1', 'inf2', 'inf3', 'inf4', 'inf5', 'inf6',
               'pi1', 'pi2', 'pi3', 'pi4', 'pi6', '_']
data.drop(data.iloc[-9:].index, inplace=True)
data.drop(0, inplace=True)
data['group'] = np.NaN
for (f, t) in zip(delim[:-1], delim[1:]):
    if t - f > 2:
        data.loc[f: t, 'group'] = group_names.pop(0)
data.dropna(subset=['name'], inplace=True)
data.fillna(0, inplace=True)
data.reset_index(drop=True, inplace=True)
data = data[['group', 'name', 'rating', 'urban', 'sport', 'science', 'total']]

data.to_csv('datasets\\rating1.csv', index=False)



