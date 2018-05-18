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


# def ReadTimeStamp(File):
#     data=pd.read_csv(File)

