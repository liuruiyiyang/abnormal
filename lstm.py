import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import matplotlib.pyplot as plt
import warnings
from keras.optimizers import Adam
from utils import SplitKPIList

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings
shift_window = 1
# 是否加载已经训练好的模型
# IS_LOAD_MODEL = False
nodes=10
batch_size=1
timestep=1
epochs=2

output_path = './BasicLSTM_output'
nb_epoch=1



def scale_data(data, label):
    X_scaler = MinMaxScaler(feature_range=(0, 1))

    X = data.values

    scaled_X = X_scaler.fit_transform(X)

    y_scaler = MinMaxScaler(feature_range=(0, 1))

    y_train = label.values

    scaled_y = y_scaler.fit_transform(y_train)

    return X_scaler, y_scaler, scaled_X, scaled_y

def fit_lstm(X, y, model_file):
    n_sample = X.shape[0]  # 样本个数
    n_feat_dim = X.shape[1]  # 特征维度
    print("X n_feat_dim=" + str(n_feat_dim))
    print("X n_sample=" + str(n_sample))
    n_label_dim = y.shape[1]

    # shape: (样本个数, time step, 特征维度)
    X = X.reshape(int(n_sample/timestep), timestep, n_feat_dim)

    y = y.reshape(int(n_sample/timestep), n_label_dim)

    print('y:\n')
    print(y)

    layer_2_units = 40
    # 构建模型
    model = Sequential()

    model.add(LSTM(input_dim=n_feat_dim, output_dim=layer_2_units, return_sequences=True))
    model.add(LSTM(
        layer_2_units,
        return_sequences=False))
    model.add(Dense(n_label_dim))
    opt = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.add(Activation("sigmoid"))
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['acc'])

    print('开始训练...')
    for i in range(nb_epoch):
        print('已迭代{}次（共{}次） '.format(i + 1, nb_epoch))
        model.fit(X, y, epochs=epochs, batch_size=batch_size,  validation_split=0.02)

        model.reset_states()

    # 在所有训练样本上运行一次，构建cell状态
    model.predict(X, batch_size=batch_size)

    # 保存模型
    model.save(model_file)

    return model


def conv2Noneflag(number):
    if number == 1:
        return np.nan
    else:
        return 0


def forecast_lstm(model, X):
    n_sample = X.shape[0]
    n_feat_dim = X.shape[1]
    X = X.reshape(-1,  n_feat_dim)
    y_pred = model.predict(X, batch_size=batch_size)
    print('model.predict(X, batch_size=batch_size):\n')
    print(model.predict(X, batch_size=batch_size))
    print('y_pred:\n')
    print(y_pred)
    print('y_pred-----end-------\n')
    return y_pred


def get_lstm_diff(KPI_ID_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

# if __name__ == '__main__':

    train_data_path = 'resources/train.csv'
    test_data_path = 'resources/test.csv'
    augment_data_path = 'resources/augment_data/'
    test_augment_data_path = 'resources/test_augment_data/'
    full_result_path = 'resources/result/prediction.csv'
    split_result_path = 'resources/result_split/prediction.csv'
    # output_path = 'resources/label_prediction_plot'
    test_data_raw = pd.read_csv(test_data_path)
    train_data_raw = pd.read_csv(train_data_path)

    KPI_LIST, KPI_ID = SplitKPIList(train_data_raw)
    KPI_LIST_test, KPI_ID_test = SplitKPIList(test_data_raw)

    # KPI_ID_name = '07927a9a18fa19ae' # 07927a9a18fa19ae 76f4550c43334374 a5bf5d65261d859a

    index = KPI_ID.index(KPI_ID_name)
    train_X_raw = pd.DataFrame(KPI_LIST[index])
    X_train_reserve = train_X_raw.copy()
    X_train_reserve = pd.DataFrame(X_train_reserve)
    # y_reserve = pd.DataFrame()
    y_reserve = X_train_reserve['value']

    y_reserve = pd.DataFrame(y_reserve)

    y_reserve = y_reserve.shift(-1)
    y_reserve = pd.DataFrame(y_reserve)

    y_reserve.ffill(inplace=True)
    y_reserve = pd.DataFrame(y_reserve)
    # print("y_reserve:", y_reserve)
    y_reserve = y_reserve.values
    X_reserve = X_train_reserve['value'].values
    test_index = KPI_ID_test.index(KPI_ID_name)
    test_X_raw = pd.DataFrame(KPI_LIST_test[test_index])
    # X_raw = pd.DataFrame(KPI_LIST[index])
    X_test = test_X_raw.copy()
    del X_test['KPI ID']
    del X_test['timestamp']
    label = KPI_LIST[index].copy()['label']
    KPI_LIST[index]['label'] = KPI_LIST[index]['label'].map(conv2Noneflag)
    KPI_LIST[index] = KPI_LIST[index].dropna(axis=0)
    del KPI_LIST[index]['KPI ID']
    del KPI_LIST[index]['label']
    del KPI_LIST[index]['timestamp']
    df = pd.DataFrame(KPI_LIST[index])
    # print("df:\n", df)
    df_y = df.shift(-1)
    df_y.ffill(inplace=True)
    # print("df_y:\n", df_y)

    y_test = X_test.shift(-1)
    y_test.ffill(inplace=True)

    test_data = X_test.values

    test_label = y_test.values

    train_X_scaler, train_y_scaler, train_scaled_X, train_scaled_y = scale_data(df, df_y)

    test_scaled_X = train_X_scaler.transform(test_data)

    test_scaled_y = train_y_scaler.transform(test_label)

    pre_item = 'value'
    model_file = './BasicLSTM_output/'+KPI_ID_name+'_lstm_model.h5'
    # 加载LSTM模型
    if os.path.exists(model_file):
        lstm_model = load_model(model_file)
    else:
        # 训练LSTM模型
        lstm_model = fit_lstm(train_scaled_X, train_scaled_y, model_file)
        # print('{}模型文件不存在'.format(model_file))
        # exit(0)

    test_diff_df = get_diff(test_data, test_scaled_y, test_scaled_X, lstm_model, train_y_scaler, KPI_ID_name+'test')

    # X_reserve = X_reserve.reshape(-1, 1)
    train_scaled_X = train_X_scaler.transform(X_reserve)

    train_scaled_y = train_y_scaler.transform(y_reserve)
    train_diff_df = get_diff(X_train_reserve.values, train_scaled_y, train_scaled_X, lstm_model, train_y_scaler, KPI_ID_name+'_train')
    return train_diff_df, test_diff_df
    # print("pred_daily_df['diff'] :", pred_daily_df['lsmt_diff'] )

    # pred_daily_df.to_csv(os.path.join(output_path, 'kdd_pred_daily_df.csv'))
    # pred_daily_df.plot()
    # plt.savefig(os.path.join(output_path, KPI_ID_name + '_lstm_pred_test_value.png'))

    # plt.show()

def get_diff(test_data, test_scaled_y, test_scaled_X, lstm_model, train_y_scaler, name):
    test_data = pd.DataFrame(test_data)
    test_dates = test_data.index.tolist()
    pred_daily_df = pd.DataFrame(columns=['True Value', 'Pred Value'], index=test_dates)
    pred_daily_df['True Value'] = pd.DataFrame(test_scaled_y)
    # test_scaled_X = test_scaled_X.reshape(int(test_scaled_X.shape[0] / timestep), timestep, test_scaled_X.shape[1])
    # print("test_scaled_X\n", test_scaled_X)
    n_sample = test_scaled_X.shape[0]  # 样本个数
    n_feat_dim = 1
    test_scaled_X = test_scaled_X.reshape(int(n_sample / timestep), timestep, n_feat_dim)
    # print("after reshape test_scaled_X\n", test_scaled_X)
    pred_value = lstm_model.predict(test_scaled_X)
    rescaled_y_pred = train_y_scaler.inverse_transform(pred_value)

    # print("rescaled_y_pred:")
    # print(rescaled_y_pred[:, 0])
    # print(len(rescaled_y_pred[:, 0]))
    for i in range(len(rescaled_y_pred[:, 0])):
        pred_daily_df.loc[i, 'Pred Value'] = rescaled_y_pred[i, 0]

    # pred_daily_df.plot()

    pred_daily_df['test_diff'] = pred_daily_df['Pred Value'] - pred_daily_df['True Value']

    # pred_daily_df.to_csv(os.path.join(output_path, 'kdd_pred_daily_df.csv'))
    # pred_daily_df.plot()
    # plt.savefig(os.path.join(output_path, name + '_lstm_pred_test_value.png'))
    #
    # plt.show()

    return pred_daily_df['test_diff']