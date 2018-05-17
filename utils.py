import pandas as pd
import numpy as np
from pprint import pprint
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
import os
from pandas import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import arange, sin, pi, random

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

def ReadTimeStamp(File):
    data=pd.read_csv(File)