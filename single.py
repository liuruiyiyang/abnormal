import numpy as np
import pandas as pd

from utils import  ReadDatatime, SplitKPIList, plot_ts_label, expand_label
from read_data import get_process_feature, get_process_f
import matplotlib.pyplot as plt
import os

# reproducibility
seed = 321

def split_data(x, y, split):
    x_train_split = x[:int(len(x) * split)]
    x_test_split = x[int(len(x) * split):]
    y_train_split = y[:int(len(y) * split)]
    y_test_split = y[int(len(y) * split):]
    return x_train_split, x_test_split, y_train_split, y_test_split


def padding_y(y_label, w):
    prediction_ = y_label[:w]
    predict_ = pd.DataFrame(np.array(prediction_))
    prediction_ = y_label
    predict_ = predict_.append(pd.DataFrame(np.array(prediction_)), ignore_index=True)
    return predict_


def padding_shift_y(y_label, w):
    prediction_ = y_label[:int(w / 2)]
    predict_ = pd.DataFrame(np.array(prediction_))
    prediction_ = y_label
    predict_ = predict_.append(pd.DataFrame(np.array(prediction_)), ignore_index=True)
    prediction_ = y_label[(len(y_label) - int(w / 2)):]
    predict_ = predict_.append(pd.DataFrame(np.array(prediction_)), ignore_index=True)
    return predict_


def get_result(ts_KPI_ID_test, ts_timestamp, predict):
    ts_KPI_ID_test = ts_KPI_ID_test.reset_index(drop=True)
    ts_timestamp = pd.DataFrame(ts_timestamp).reset_index(drop=True)
    ts_timestamp.insert(loc=0, column='KPI ID', value=ts_KPI_ID_test)
    ts_timestamp.insert(loc=2, column='predict', value=predict)
    # print(ts_timestamp)
    return ts_timestamp


def save(result_path, ts_result):
    if os.path.isfile(result_path):
        with open(result_path, 'a') as file:
            ts_result.to_csv(file, header=False, index=False)
    else:
        ts_result.to_csv(result_path, header=True, index=False)


def is_near_zero(number):
    if number < 0.1:
        return 1
    else:
        return 0


def is_zero(number):
    if number == 0:
        return 1
    else:
        return 0


def boolean2int(number):
    if number:
        return 1
    else:
        return 0


def get_manual_feature(ts_df):
    raw_df = pd.DataFrame(ts_df)
    raw_value = raw_df['value']
    # print("raw_df['value']\n", raw_value)
    is_zero_feature = raw_df['value'].map(is_zero)
    # print("is_zero_feature\n", is_zero_feature)
    is_near_zero_feature = raw_df['value'].map(is_near_zero)
    # print("is_near_zero_feature\n", is_near_zero_feature)
    diff_feature = raw_df['value'].diff().fillna(0)
    # abs_diff_feature = abs(diff_feature)

    mean_diff = diff_feature.mean()
    std_diff = diff_feature.std()
    is_diff_over_3sigma = diff_feature.apply(lambda x: abs(x - mean_diff) > (10 * std_diff))
    is_diff_over_3sigma = is_diff_over_3sigma.map(boolean2int)
    # print("mean_diff:")
    # print(mean_diff)
    # print("std_diff:")
    # print(std_diff)
    # std_diff = abs_diff_feature.std()
    # print("std_diff:")
    # print(std_diff)
    # print("diff_feature\n", diff_feature)
    mean = raw_df['value'].mean()
    diff_minus_mean_feature = diff_feature.apply(lambda x: (x - mean))
    value_minus_mean_feature = raw_df['value'].apply(lambda x: (x - mean))
    abs_value_minus_mean_feature = raw_df['value'].apply(lambda x: abs(x - mean))
    is_over_mean_feature = raw_df['value'].apply(lambda x: ((abs(x - mean) / mean) > 0.3))
    is_over_mean_feature = is_over_mean_feature.map(boolean2int)
    is_over_mean_feature_1 = raw_df['value'].apply(lambda x: ((abs(x - mean) / mean) > 0.5))
    is_over_mean_feature_1 = is_over_mean_feature_1.map(boolean2int)
    is_over_mean_feature_2 = raw_df['value'].apply(lambda x: ((abs(x - mean) / mean) > 1))
    is_over_mean_feature_2 = is_over_mean_feature_2.map(boolean2int)
    is_over_mean_feature_3 = raw_df['value'].apply(lambda x: ((abs(x - mean) / mean) > 2))
    is_over_mean_feature_3 = is_over_mean_feature_3.map(boolean2int)
    is_over_mean_feature_4 = raw_df['value'].apply(lambda x: ((abs(x - mean) / mean) > 5))
    is_over_mean_feature_4 = is_over_mean_feature_4.map(boolean2int)
    is_over_mean_feature_5 = raw_df['value'].apply(lambda x: ((abs(x - mean) / mean) > 0.75))
    is_over_mean_feature_5 = is_over_mean_feature_5.map(boolean2int)
    diff_timestamp_feature = raw_df['timestamp'].diff()
    diff_timestamp_feature = pd.DataFrame(diff_timestamp_feature).bfill()

    manual_features = pd.DataFrame()
    manual_features['raw_value'] = raw_value
    manual_features['is_zero_feature'] = is_zero_feature
    manual_features['is_near_zero_feature'] = is_near_zero_feature
    manual_features['diff_feature'] = diff_feature
    manual_features['diff_minus_mean_feature'] = diff_minus_mean_feature
    manual_features['value_minus_mean_feature'] = value_minus_mean_feature
    manual_features['abs_value_minus_mean_feature'] = abs_value_minus_mean_feature
    manual_features['is_over_mean_feature'] = is_over_mean_feature
    manual_features['diff_timestamp_feature'] = diff_timestamp_feature
    manual_features['is_over_mean_feature_1'] = is_over_mean_feature_1
    manual_features['is_over_mean_feature_2'] = is_over_mean_feature_2
    manual_features['is_over_mean_feature_3'] = is_over_mean_feature_3
    manual_features['is_over_mean_feature_4'] = is_over_mean_feature_4
    manual_features['is_over_mean_feature_5'] = is_over_mean_feature_5
    manual_features['is_diff_over_3sigma'] = is_diff_over_3sigma
    manual_features = manual_features.reset_index(drop=True)
    return manual_features


def lg_1600(x):
    if x > 1700 or x < 1000:
        return 1
    else:
        return 0


def lw_300(x):
    if x > 3600 or x < 300:
        return 1
    else:
        return 0


def a20(x):
    if x > 0.2:
        return 1
    else:
        return 0


def bef(x):
    if x > 3960 or x < 350:
        return 1
    else:
        return 0


def c89(x):
    if x > 13000 or x < 1500:
        return 1
    else:
        return 0


def bd9(x):
    if x > 54 or x < 6.9:
        return 1
    else:
        return 0


def ee5(x):
    if x > 0.4 or x < 0.018:
        return 1
    else:
        return 0


def e99(x):
    if x > 3 or x < 0.366:
        return 1
    else:
        return 0


def fbb(x):
    if x > 15:
        return 1
    else:
        return 0


def e25(x):
    if x > 2630 or x < 120:
        return 1
    else:
        return 0


def e8a(x):
    if x > 3 or x < -3:
        return 1
    else:
        return 0


def f45(x):
    if x > 0.2 or x < 0:
        return 1
    else:
        return 0


def cf3(x):
    if x > 0.4 or x < 0.02:
        return 1
    else:
        return 0


def ec2(x):
    if x < -1.67:
        return 1
    else:
        return 0


def a9a(x):
    if x > 2.0 or x < -1.55:
        return 1
    else:
        return 0


def ae3(x):
    if x > 1500000000.0 or x < 700000000:
        return 1
    else:
        return 0


def dd7(x):
    if x > 2500 or x < 220:
        return 1
    else:
        return 0


def bae(x):
    if x > 0.2 or x < 0.01:
        return 1
    else:
        return 0


def a5b(x):
    if x > 3:
        return 1
    else:
        return 0


def a40(x):
    if x > 3680 or x < 420:
        return 1
    else:
        return 0


def aff(x):
    if x > 3900 or x < 390:
        return 1
    else:
        return 0


def b3b(x):
    if x > 4 or x < -4:
        return 1
    else:
        return 0


def cff(x):
    if x > 2400 or x < 200:
        return 1
    else:
        return 0


def da4(x):
    if x > 6.5 or x < 0.7:
        return 1
    else:
        return 0


def e07(x):
    if x > 13000 or x < 2000:
        return 1
    else:
        return 0


def to_int(x):
    return int(x)


def get_single_feature(raw_df, KPI_ID_name):
    df = pd.DataFrame(raw_df.copy())
    df = df.reset_index(drop=True)
    # test_manual_feature = get_manual_feature(KPI_LIST_test[index])
    # test_manual_feature['diff_shift']
    # median_time = test_manual_feature['diff_timestamp_feature'].median()

    print("KPI_LIST_test[index] length:", len(KPI_LIST_test[index]))
    # print("test_manual_feature length:", len(test_manual_feature))
    # test_processd_feature = get_process_feature(is_test=True, KPI_ID_name=KPI_ID_name, window=window)
    # df['pro_diff'] = test_processd_feature
    # df['pro_diff'] = df['pro_diff'] * 100

    del df['timestamp']
    del df['KPI ID']

    # print(df)
    if KPI_ID_name == '1c35dbf57f55f5e4':
        y_df = df['value'].map(lg_1600)
    elif KPI_ID_name == '7c189dd36f048a6c':
        y_df = df['value'].map(lw_300)
    elif KPI_ID_name == '8a20c229e9860d0c':
        y_df = df['value'].map(a20)
    elif KPI_ID_name == '8bef9af9a922e0b3':
        y_df = df['value'].map(bef)
    elif KPI_ID_name == '8c892e5525f3e491':
        y_df = df['value'].map(c89)
    elif KPI_ID_name == '9bd90500bfd11edb':
        y_df = df['value'].map(bd9)
    elif KPI_ID_name == '9ee5879409dccef9':
        y_df = df['value'].map(ee5)
    elif KPI_ID_name == '02e99bd4f6cfb33f':
        y_df = df['value'].map(e99)
    elif KPI_ID_name == '18fbb1d5a5dc099d':
        y_df = df['value'].map(fbb)
    elif KPI_ID_name == '40e25005ff8992bd':
        y_df = df['value'].map(e25)
    elif KPI_ID_name == '54e8a140f6237526':
        y_df = df['value'].map(e8a)
    elif KPI_ID_name == '76f4550c43334374':
        y_df = df['value'].map(f45)
    elif KPI_ID_name == '88cf3a776ba00e7c':
        y_df = df['value'].map(cf3)
    elif KPI_ID_name == '046ec29ddf80d62e':
        y_df = df['value'].map(ec2)
    elif KPI_ID_name == '07927a9a18fa19ae':
        y_df = df['value'].map(a9a)
    elif KPI_ID_name == '09513ae3e75778a3':
        y_df = df['value'].map(ae3)
    elif KPI_ID_name == '71595dd7171f4540':
        y_df = df['value'].map(dd7)
    elif KPI_ID_name == '769894baefea4e9e':
        y_df = df['value'].map(bae)
    elif KPI_ID_name == 'a5bf5d65261d859a':
        y_df = df['value'].map(a5b)
    elif KPI_ID_name == 'a40b1df87e3f1c87':
        y_df = df['value'].map(a40)
    elif KPI_ID_name == 'affb01ca2b4f0b45':
        y_df = df['value'].map(aff)
    elif KPI_ID_name == 'b3b2e6d1a791d63a':
        y_df = df['value'].map(b3b)
    elif KPI_ID_name == 'c58bfcbacb2822d1':
        test_manual_feature = get_manual_feature(KPI_LIST_test[index])
        y_df = test_manual_feature['is_diff_over_3sigma']
    elif KPI_ID_name == 'cff6d3c01e6a6bfa':
        y_df = df['value'].map(cff)
    elif KPI_ID_name == 'da403e4e3f87c9e0':
        y_df = df['value'].map(da4)
    elif KPI_ID_name == 'e0770391decc44ce':
        y_df = df['value'].map(e07)

    # print("y_df before shift:")
    # print(y_df)

    df['y'] = y_df
    if is_shift:
        df['y'] = df['y'].shift(shift)
        df['y'].bfill(inplace=True)
        df['y'] = df['y'].map(to_int)

    df['y'] = expand_label(df['y'].values, wide=wide)

    return df['y']


window = 8
wide = 1
is_shift = False
shift = 2
# score_threshold = 0.997
# KPI_ID_name = '76f4550c43334374' 8a20c229e9860d0c
train_data_path = 'resources/train.csv'
test_data_path = 'resources/test.csv'
augment_data_path = 'resources/augment_data/'
test_augment_data_path = 'resources/test_augment_data/'
full_result_path = 'resources/result/prediction.csv'
xgb_result_path = 'resources/result_xgb/diff_expand1_prediction.csv'
output_path = 'resources/label_prediction_plot'
manual_features_path = 'resources/manual_feature'
test_data_raw = pd.read_csv(test_data_path)
train_data_raw = pd.read_csv(train_data_path)

KPI_LIST, KPI_ID = SplitKPIList(train_data_raw)
KPI_LIST_test, KPI_ID_test = SplitKPIList(test_data_raw)

# print(KPI_LIST)
# print('KPI_ID:', KPI_ID)
# KPI_ID:
# ['71595dd7171f4540',
# '88cf3a776ba00e7c',
# 'affb01ca2b4f0b45',
# '769894baefea4e9e',
# '02e99bd4f6cfb33f',
# '54e8a140f6237526',
# 'b3b2e6d1a791d63a',
# '07927a9a18fa19ae',
# 'a40b1df87e3f1c87',
# 'c58bfcbacb2822d1',
# '1c35dbf57f55f5e4',
# 'cff6d3c01e6a6bfa',
# '7c189dd36f048a6c',
# '76f4550c43334374',
# '18fbb1d5a5dc099d',
# '9bd90500bfd11edb',
# '8a20c229e9860d0c',
# 'e0770391decc44ce',
# '09513ae3e75778a3',
# '046ec29ddf80d62e',
# '8c892e5525f3e491',
# '8bef9af9a922e0b3',
# '40e25005ff8992bd',
# 'da403e4e3f87c9e0',
# 'a5bf5d65261d859a',
# '9ee5879409dccef9']

KPI_ID_e = ['02e99bd4f6cfb33f', '046ec29ddf80d62e', '07927a9a18fa19ae', '09513ae3e75778a3', '18fbb1d5a5dc099d', '1c35dbf57f55f5e4', '40e25005ff8992bd', '54e8a140f6237526', '71595dd7171f4540', '769894baefea4e9e', '76f4550c43334374', '7c189dd36f048a6c', '88cf3a776ba00e7c', '8a20c229e9860d0c', '8bef9af9a922e0b3', '8c892e5525f3e491', '9bd90500bfd11edb', '9ee5879409dccef9', 'a40b1df87e3f1c87', 'a5bf5d65261d859a', 'affb01ca2b4f0b45', 'b3b2e6d1a791d63a', 'c58bfcbacb2822d1', 'cff6d3c01e6a6bfa', 'da403e4e3f87c9e0', 'e0770391decc44ce']

# KPI_ID_e = ['c58bfcbacb2822d1'] # , '76f4550c43334374'

for KPI_ID_name in KPI_ID_e:

    # prediction ##################################----------------------------

    index = KPI_ID_test.index(KPI_ID_name)
    xgb_predict = get_single_feature(raw_df=KPI_LIST_test[index], KPI_ID_name=KPI_ID_name)
    ts_KPI_ID_test = KPI_LIST_test[index].pop('KPI ID')
    ts_timestamp = KPI_LIST_test[index]['timestamp']
    print("ts_KPI_ID_test:\n", ts_KPI_ID_test.iloc[[0]])
    xgb_ts_result = get_result(ts_KPI_ID_test, ts_timestamp, xgb_predict)

    save(result_path=xgb_result_path, ts_result=xgb_ts_result)

print("finish !!!")


