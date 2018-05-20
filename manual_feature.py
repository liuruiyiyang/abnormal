import numpy as np
import pandas as pd
import heapq
import xgboost as xgb
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score, recall_score,f1_score,confusion_matrix
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from utils import  ReadDatatime, SplitKPIList, plot_ts_label
from lstm import get_lstm_diff
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
    manual_features = manual_features.reset_index(drop=True)
    return manual_features




window = 6
# score_threshold = 0.997
# KPI_ID_name = '76f4550c43334374' 8a20c229e9860d0c
train_data_path = 'resources/train.csv'
test_data_path = 'resources/test.csv'
augment_data_path = 'resources/augment_data/'
test_augment_data_path = 'resources/test_augment_data/'
full_result_path = 'resources/result/prediction.csv'
xgb_result_path = 'resources/result_xgb/man_xgb_prediction.csv'
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

KPI_ID_e = ['07927a9a18fa19ae', '76f4550c43334374']

for KPI_ID_name in KPI_ID_e:

    print("current KPI ID:", KPI_ID_name)

    index = KPI_ID.index(KPI_ID_name)
    print("current KPI ID index:", index, "/", len(KPI_ID))
    filtered_feature_name_list = []

    y = KPI_LIST[index].pop('label')
    y_shift = y
    y = y.values[window:]
    y = pd.Series(y)

    # print("y length:", len(y_shift))
    # y_shift = y_shift.values[int(window/2):len(y_shift)-int(window/2)]
    y_shift = y_shift.values
    # print("y_shift length:", len(y_shift))
    y_shift = pd.Series(y_shift)

    # y = y.reset_index(drop=True)
    # print(y)
    # calculate manual features

    manual_feature_df = get_manual_feature(KPI_LIST[index])
    lstm_diff_train, lstm_diff_test = get_lstm_diff(KPI_ID_name)
    manual_feature_df['lstm_diff'] = lstm_diff_train

    print("manual_feature_df with lstm added:\n", manual_feature_df)

    print("lstm_diff_test:", lstm_diff_test)

    # print(range(window))
    # manual_features.drop(range(window), inplace=True)
    # print("manual_features:", manual_features)

    # for i in range(window, len(KPI_LIST[index])):
    #     ts_point_value = KPI_LIST[index][i]['value']
    #
    #     ts_slice = pd.DataFrame(ts_slice)
    #     ts_slice['KPI ID'] = i - window
    #     augment_data = augment_data.append(ts_slice, ignore_index=True)
    #     print(i, "in", len(KPI_LIST[index]), float('%.2f' % (i / len(KPI_LIST[index]) * 100)), "%")
    # augment_data.to_csv(
    #     augment_data_path + "augment_data_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv",
    #     index=False
    # )

    ts_KPI_ID = KPI_LIST[index].pop('KPI ID')
    print("ts_KPI_ID:\n", ts_KPI_ID.iloc[[0]])

#    pd.set_option('mode.use_inf_as_na', True)
    # sc = MinMaxScaler()
    # ts_X_train_df = pd.read_csv("resources/ts_feature_"+"window_"+str(window)+"_KPI_"+KPI_ID_name+".csv")

    # del ts_X_train_df['id']
    # ts_X_train = ts_X_train_df.values

    X_train = manual_feature_df.values
    y_train_shift = y_shift.values

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_shift, test_size=0.33, stratify=y_train_shift, random_state=seed)

    X_train = manual_feature_df.values
    y_train = y_train_shift

    print("X_train shape:")
    print(X_train.shape)
    print("y_train length:")
    print(len(y_train))

    # X_train_ = X_train
    # y_train_ = y_train

    # y_train_shift_ = y_train_shift

    print(len(X_train),len(y_train))
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'silent': 1,
        'eta': 5,
        'learning_rate': 0.19,
        'n_estimators': 100000
    }

    num_rounds = 50

    train_labels = dtrain.get_label()

    ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)
    # print("ratio:", ratio)
    params['scale_pos_weight'] = ratio

    bst = xgb.train(params, dtrain, num_rounds)
    y_test_preds = (bst.predict(dtest) > 0.5).astype('int')
    print(confusion_matrix(y_test, y_test_preds))

    print('Accuracy: {0:.4f}'.format(accuracy_score(y_test, y_test_preds)))
    print('Precision: {0:.4f}'.format(precision_score(y_test, y_test_preds)))
    print('Recall: {0:.4f}'.format(recall_score(y_test, y_test_preds)))
    print('f1: {0:.4f}'.format(f1_score(y_test, y_test_preds)))
    print('roc_auc_score: {0:.4f}'.format(roc_auc_score(y_test, y_test_preds)))



    # prediction ##################################----------------------------

    index = KPI_ID_test.index(KPI_ID_name)

    test_manual_feature = get_manual_feature(KPI_LIST_test[index])

    ts_KPI_ID_test = KPI_LIST_test[index].pop('KPI ID')
    ts_timestamp = KPI_LIST_test[index]['timestamp']
    print("ts_KPI_ID_test:\n", ts_KPI_ID_test.iloc[[0]])

    X_train = test_manual_feature.values
    print("X_train.shape:", X_train.shape)

    dtest = xgb.DMatrix(X_train)
    y_test_preds = (bst.predict(dtest) > 0.5).astype('int')

    xgb_predict = y_test_preds
    # xgb_predict = padding_shift_y(y_test_preds, window)
    # shift_predict = padding_shift_y(cl_shift.predict(X_train), window)
    # split_predict = padding_y(cl_split.predict(X_train), window)
    # full_ts_result = get_result(ts_KPI_ID_test, ts_timestamp, full_predict)
    xgb_ts_result = get_result(ts_KPI_ID_test, ts_timestamp, xgb_predict)
    print("xgb_ts_result:", ts_KPI_ID_test.iloc[[0]])

    save(result_path=xgb_result_path, ts_result=xgb_ts_result)

print("finish !!!")


