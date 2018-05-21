import pandas as pd


def get_process_f(is_test, KPI_ID_name, window):
    # KPI_ID_name = '18fbb1d5a5dc099d'
    # window = 8
    # file_name = 'resources/train.csv'
    # raw_data = pd.read_csv(file_name)

    # print(raw_data)

    data_name = ['02e99bd4f6cfb33f', '046ec29ddf80d62e', '07927a9a18fa19ae', '09513ae3e75778a3', '18fbb1d5a5dc099d', '1c35dbf57f55f5e4', '40e25005ff8992bd', '54e8a140f6237526', '71595dd7171f4540', '769894baefea4e9e', '76f4550c43334374', '7c189dd36f048a6c', '88cf3a776ba00e7c', '8a20c229e9860d0c', '8bef9af9a922e0b3', '8c892e5525f3e491', '9bd90500bfd11edb', '9ee5879409dccef9', 'a40b1df87e3f1c87', 'a5bf5d65261d859a', 'affb01ca2b4f0b45', 'b3b2e6d1a791d63a', 'c58bfcbacb2822d1', 'cff6d3c01e6a6bfa', 'da403e4e3f87c9e0', 'e0770391decc44ce']

    index = data_name.index(KPI_ID_name)
    print("index=", index)

    if is_test:
        input_file_path = 'process_data/' + str(index + 1) + 'testresult.txt'
    else:
        input_file_path = 'process_data/' + str(index + 1) + 'result.txt'
    result = pd.read_table(input_file_path, delim_whitespace=True, header=None)
    result = result.T
    result['process_diff'] = abs(result[0]-result[1])

    head_data = result.iloc[range(49-int(window / 2))]
    head_data = pd.DataFrame(head_data)
    result = head_data.append(result, ignore_index=True)
    result = result.reset_index(drop=True)
    # print(result)
    result.drop(range(len(result) - int(window / 2), len(result)), inplace=True)
    result = result.reset_index(drop=True)
    return result['process_diff']

# i = 1
# for name, group in raw_data.groupby('KPI ID'):
#
#     group.name = name
#     if i == index + 1:
#         print("group:")
#         print(group)
#         print("name:")
#         print(name)
#     data_name.append(name)
#     i = i + 1
#
#     data.append(group)
# print("data_name:", data_name)
