import numpy as np
import pandas as pd
import os
from pandas import datetime

import matplotlib.pyplot as plt


def ReadDatatime(File):
    data=pd.read_csv(File)
    data.loc[:, 'timestamp']=data['timestamp'].apply(lambda x:datetime.utcfromtimestamp(x))
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    return data

def RedaPredata(file):
    data=pd.read_csv(file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    return data

def SplitKPIList(data):
    KPI_LIST=[]
    KPI_ID = list(set(data['KPI ID']))
    for i in KPI_ID:
        KPI_LIST.append(data[data['KPI ID']==i])
    return KPI_LIST,KPI_ID


def plot_ts_label(index,KPI_LIST,KPI_ID):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    pd.DataFrame(KPI_LIST[index])['value'].plot(ax=axes[0]);axes[0].set_title(KPI_ID[index] + 'value')
    pd.DataFrame(KPI_LIST[index])['label'].plot(ax=axes[1]);axes[1].set_title(KPI_ID[index] + 'label')
    plt.show()


def to_int(x):
    return int(x)


def bool_to_int(x):
    if x:
        return 1
    else:
        return 0


def expand_label(y, wide):
    s = pd.Series(y)
    df = pd.DataFrame(s, columns=['s'])
    df['result'] = df['s']
    for i in range(1, wide + 1):
        df['shift'] = df['s'].shift(i)
        df['shift'].fillna(0, inplace=True)
        df['shift'].map(to_int)
        df['result'] = df['result'] | df['shift']
        df['shift'] = df['s'].shift(-i)
        df['shift'].fillna(0, inplace=True)
        df['shift'].map(to_int)
        df['result'] = df['result'] | df['shift']
    return df['result'].map(bool_to_int)

# def ReadTimeStamp(File):
#     data=pd.read_csv(File)

# rng = pd.date_range('1/1/2014', periods=10, freq='5min')
# s = pd.Series([1, 0, 1, 1, 1, 0, 0, 1, 0, 1], index=rng)
# s2 = pd.Series([0, 0, 1, 0, 1, 1, 0, 0, 0, 1], index=rng)
# # df = pd.DataFrame(s, columns=['val1'])
# df = pd.concat([s,s2], axis=1, ignore_index=True)
# df.index.name ="dt"
#
# df['or'] = df[0] | df[1]
# print(df)
                #[1, 0, 0, 0, 1, 0, 0, 1, 0, 1]
# s = pd.Series([0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
# r = expand_label(s, 2)
# print(r)