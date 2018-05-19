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
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from sklearn.tree import DecisionTreeClassifier
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import matplotlib
import matplotlib.pyplot as plt
import os

# reproducibility
seed = 123

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
    print(ts_timestamp)
    return ts_timestamp


def save(result_path, ts_result):
    if os.path.isfile(result_path):
        with open(result_path, 'a') as file:
            ts_result.to_csv(file, header=False, index=False)
    else:
        ts_result.to_csv(result_path, header=True, index=False)


window = 6
# score_threshold = 0.997
# KPI_ID_name = '76f4550c43334374' 8a20c229e9860d0c
train_data_path = 'resources/train.csv'
test_data_path = 'resources/test.csv'
augment_data_path = 'resources/augment_data/'
test_augment_data_path = 'resources/test_augment_data/'
full_result_path = 'resources/result/prediction.csv'
xgb_result_path = 'resources/result_xgb/xgb_prediction.csv'
output_path = 'resources/label_prediction_plot'
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

    print("y length:", len(y_shift))
    y_shift = y_shift.values[int(window/2):len(y_shift)-int(window/2)]
    print("y_shift length:", len(y_shift))
    y_shift = pd.Series(y_shift)
    # y = y.reset_index(drop=True)
    # print(y)

    ts_KPI_ID = KPI_LIST[index].pop('KPI ID')
    print("ts_KPI_ID:\n", ts_KPI_ID.iloc[[0]])

    pd.set_option('mode.use_inf_as_na', True)
    sc = MinMaxScaler()

    if os.path.isfile("resources/ts_feature_"+"window_"+str(window)+"_KPI_"+KPI_ID_name+".csv"):
        X_train_df = pd.read_csv("resources/ts_feature_"+"window_"+str(window)+"_KPI_"+KPI_ID_name+".csv")
        head_df = pd.read_csv("resources/ts_feature_with_head_"+"window_"+str(window)+"_KPI_"+KPI_ID_name+".csv")
        del head_df['id']
        filtered_feature_name_list = head_df.columns.values.tolist()
        # print("filtered_feature_name_list:", filtered_feature_name_list)
        print("filtered_feature_name_list length:", len(filtered_feature_name_list))
        del X_train_df['id']
        X_train = X_train_df.values
        sc.fit(X_train)
        X_train = sc.transform(X_train)
    else:
        augment_data = pd.DataFrame()
        if os.path.isfile(augment_data_path + "augment_data_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv"):
            augment_data = pd.read_csv(augment_data_path + "augment_data_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv")
        else:
            for i in range(window, len(KPI_LIST[index])):
                ts_slice = KPI_LIST[index][i - window:i]
                ts_slice = pd.DataFrame(ts_slice)
                ts_slice['KPI ID'] = i - window
                augment_data = augment_data.append(ts_slice, ignore_index=True)
                print(i, "in", len(KPI_LIST[index]), float('%.2f' % (i / len(KPI_LIST[index]) * 100)), "%")
            augment_data.to_csv(
                augment_data_path + "augment_data_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv",
                index=False
            )

        print("augment_data:\n", augment_data)
        ts_feature = extract_relevant_features(augment_data, y, column_id="KPI ID", column_sort="timestamp")
        ts_feature.dropna(axis=1, inplace=True)
        ts_feature = pd.DataFrame(ts_feature)
        feature_name_list_df = pd.DataFrame(ts_feature.iloc[[0]])
        feature_name_list_df.to_csv("resources/ts_feature_with_head_"+"window_"+str(window)+"_KPI_"+KPI_ID_name+".csv")
        filtered_feature_name_list = ts_feature.columns.values.tolist()
        # print("filtered_feature_name_list:", filtered_feature_name_list)
        print("filtered_feature_name_list length:", len(filtered_feature_name_list))
        # ts_feature.to_csv("resources/ts_feature_with_head_"+"window_"+str(window)+"_KPI_"+KPI_ID_name+".csv")
        ts_feature.to_csv("resources/ts_feature_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv")
        X_train = ts_feature.values
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        # X_train_df = pd.DataFrame(X_train)
        # X_train_df.to_csv("resources/ts_feature_" + "window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv")

    # y_train = y.values
    y_train_shift = y_shift.values

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_shift, test_size=0.33, stratify=y_train_shift, random_state=seed)

    print("X_train shape:")
    print(X_train.shape)
    print("y_train length:")
    print(len(y_train))

    # X_train_ = X_train
    # y_train_ = y_train

    # y_train_shift_ = y_train_shift

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 50,
        'silent': 1,
        'eta': 5,
        'learning_rate': 0.1,
        'n_estimators': 100000
    }

    num_rounds = 50

    train_labels = dtrain.get_label()

    ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)
    params['scale_pos_weight'] = ratio

    bst = xgb.train(params, dtrain, num_rounds)
    y_test_preds = (bst.predict(dtest) > 0.5).astype('int')
    print(confusion_matrix(y_test, y_test_preds))

    print('Accuracy: {0:.4f}'.format(accuracy_score(y_test, y_test_preds)))
    print('Precision: {0:.4f}'.format(precision_score(y_test, y_test_preds)))
    print('Recall: {0:.4f}'.format(recall_score(y_test, y_test_preds)))
    print('f1: {0:.4f}'.format(f1_score(y_test, y_test_preds)))
    print('roc_auc_score: {0:.4f}'.format(roc_auc_score(y_test, y_test_preds)))

    # while score < score_threshold:
    #     X_train, X_test, y_train, y_test = train_test_split(X_train_, y_train_, test_size=.3)
    #     cl_split = DecisionTreeClassifier()
    #     cl_split.fit(X_train, y_train)
    #     prediction = cl_split.predict(X_test)
    #     score = cl_split.score(X_test, y_test
   #     print(cl_split.score(X_test, y_test))
    #     print(classification_report(y_test, prediction))



    # prediction = cl_split.predict(X_train_)
    # print(cl_split.score(X_train_, y_train_))
    # print(classification_report(y_train_, prediction))

    # fig, axes = plt.subplots(nrows=2, ncols=1)
    # pd.DataFrame(y_train_).plot(ax=axes[0]);axes[0].set_title('label')
    # pd.DataFrame(prediction).plot(ax=axes[1]);axes[1].set_title('predict')
    # plt.savefig(os.path.join(output_path, KPI_ID[index] + '_label_prediction.png'))

    # prediction ##################################----------------------------

    index = KPI_ID_test.index(KPI_ID_name)
    ts_KPI_ID_test = KPI_LIST_test[index].pop('KPI ID')
    ts_timestamp = KPI_LIST_test[index]['timestamp']
    print("ts_KPI_ID_test:\n", ts_KPI_ID_test.iloc[[0]])

    if os.path.isfile("resources/test_feature/ts_feature_"+"window_"+str(window)+"_KPI_"+KPI_ID_name+".csv"):
        X_train_df = pd.read_csv("resources/test_feature/ts_feature_"+"window_"+str(window)+"_KPI_"+KPI_ID_name+".csv")
        del X_train_df['id']
        X_train = X_train_df.values
        X_train = sc.transform(X_train)
    else:
        augment_data = pd.DataFrame()
        if os.path.isfile(test_augment_data_path + "test_augment_data_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv"):
            augment_data = pd.read_csv(test_augment_data_path + "test_augment_data_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv")
        else:
            for i in range(window, len(KPI_LIST_test[index])):
                ts_slice = KPI_LIST_test[index][i - window:i]
                ts_slice = pd.DataFrame(ts_slice)
                ts_slice['KPI ID'] = i - window
                augment_data = augment_data.append(ts_slice, ignore_index=True)
                print(i, "in", len(KPI_LIST_test[index]), float('%.2f' % (i / len(KPI_LIST_test[index]) * 100)), "%")
            augment_data.to_csv(
                test_augment_data_path + "test_augment_data_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv",
                index=False
            )
        print("augment_test_data:\n", augment_data)
        ts_feature = extract_features(augment_data, column_id="KPI ID", column_sort="timestamp")
        ts_feature = pd.DataFrame(ts_feature)
        ts_feature = ts_feature[filtered_feature_name_list]
        ts_feature.fillna(0, inplace=True)
        # ts_feature.to_csv("resources/ts_feature_"+"window_"+str(window)+"_KPI_"+KPI_ID_name+".csv")
        ts_feature.to_csv("resources/test_feature/ts_feature_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv")
        X_train = ts_feature.values
        X_train = sc.transform(X_train)
        # X_train_df = pd.DataFrame(X_train)
        # X_train_df.to_csv("resources/test_feature/ts_feature_window_" + str(window) + "_KPI_" + KPI_ID_name + ".csv")

    print("X_train.shape:", X_train.shape)

    dtest = xgb.DMatrix(X_train)
    y_test_preds = (bst.predict(dtest) > 0.5).astype('int')

    xgb_predict = padding_shift_y(y_test_preds, window)
    # shift_predict = padding_shift_y(cl_shift.predict(X_train), window)
    # split_predict = padding_y(cl_split.predict(X_train), window)
    # full_ts_result = get_result(ts_KPI_ID_test, ts_timestamp, full_predict)
    xgb_ts_result = get_result(ts_KPI_ID_test, ts_timestamp, xgb_predict)
    print("xgb_ts_result:", ts_KPI_ID_test.iloc[[0]])
    print(xgb_ts_result)
    # split_ts_result = get_result(ts_KPI_ID_test, ts_timestamp, split_predict)

    save(result_path=xgb_result_path, ts_result=xgb_ts_result)

    # fig2, axes = plt.subplots(nrows=2, ncols=1)
    # pd.DataFrame(full_predict).plot(ax=axes[0]);axes[0].set_title('full predict')
    # pd.DataFrame(split_predict).plot(ax=axes[1]);axes[1].set_title('split predict')
    # plt.savefig(os.path.join(output_path, KPI_ID[index] + '_full_split_prediction.png'))

print("finish !!!")


