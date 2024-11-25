import sys
sys.path.append('../')

import os
import pandas as pd
import numpy as np
import gc
from logparser import Spell, Drain
from tqdm import tqdm
import json
from logdeep.dataset.session import sliding_window

tqdm.pandas()
pd.options.mode.chained_assignment = None  # default='warn'
log_templates_file = r"E:\logbert-main\output\no_shuffle_tbird_325\Thunderbird_20M.log_templates.csv"


def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by=["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx, event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    with open(output_dir + "tbird_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def count_anomaly(log_path):
    total_size = 0
    normal_size = 0
    with open(log_path, errors='ignore') as f:
        for line in f:
            total_size += 1
            if line.split('')[0] == '-':
                normal_size += 1
    print("total size {}, abnormal size {}".format(total_size, total_size - normal_size))


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+',  # hexadecimal
        r'\d+\.\d+\.\d+\.\d+',
        r'(?<=Warning: we failed to resolve data source name )[\w\s]+',
        r'\d+'
    ]
    keep_para = False
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes

        # Drain is modified
        parser = Drain.LogParser(log_format,
                                 indir=input_dir,
                                 outdir=output_dir,
                                 depth=depth,
                                 st=st,
                                 rex=regex,
                                 keep_para=keep_para, maxChild=1000)
        parser.parse(log_file)

    elif parser_type == "spell":
        tau = 0.35
        parser = Spell.LogParser(indir=data_dir,
                                 outdir=output_dir,
                                 log_format=log_format,
                                 tau=tau,
                                 rex=regex,
                                 keep_para=keep_para)
        parser.parse(log_file)



def sample_raw_data(data_file, output_file, sample_window_size, sample_step_size):
    # sample 1M by sliding window, abnormal rate is over 2%
    sample_data = []
    labels = []
    idx = 0

    # spirit dataset can start from the 2Mth line, as there are many abnormal lines gathering in the first 2M
    with open(data_file, 'r', errors='ignore') as f:
        for line in f:
            labels.append(line.split()[0] != '-')
            sample_data.append(line)

            if len(labels) == sample_window_size:
                abnormal_rate = sum(np.array(labels)) / len(labels)
                print(f"{idx + 1} lines, abnormal rate {abnormal_rate}")
                break

            idx += 1
            if idx % sample_step_size == 0:
                print(f"Process {round(idx/sample_window_size * 100, 4)} % raw data", end='\r')

    with open(output_file, "w") as f:
        f.writelines(sample_data)

    print("Sampling done")


if __name__ == "__main__":
    data_dir = os.path.expanduser(r"E:\logbert-main\datasets\tbird/")
    # output_dir = "../output/tbird/"
    output_dir = "../output/no_shuffle_tbird_325/"
    raw_log_file = "Thunderbird.log"
    sample_log_file = "Thunderbird_20M.log"
    sample_window_size = 2*10**7
    sample_step_size = 10**4
    window_name = ''
    log_file = sample_log_file

    parser_type = 'drain'
    #mins
    window_size = 1
    step_size = 0.5
    train_ratio = 6000

    ########
    # count anomaly
    ########
    # count_anomaly(data_dir + log_file)
    # sys.exit()

    #########
    # sample raw data
    #########
    sample_raw_data(data_dir+raw_log_file, data_dir+sample_log_file, sample_window_size, sample_step_size )


    ##########
    # Parser #
    #########
    if not os.path.exists(r"E:\logbert-main\output\tbird\Thunderbird_20M.log_structured.csv"):
        parse_log(data_dir, output_dir, log_file, parser_type)

    mapping()

    ##################
    # Transformation #
    ##################
    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')

    # data preprocess
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

    df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'], format='%Y-%m-%d %H:%M:%S')
    df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
    df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)

    # sampling with sliding window
    deeplog_df = sliding_window(df[["timestamp", "Label", "EventId", "deltaT"]],
                                para={"window_size": float(window_size)*60, "step_size": float(step_size) * 60}
                                )
    output_dir += window_name

    # 所有数据df的长度
    # deeplog_df_len = len(deeplog_df)
    # # 将deeplog_df按照5:2.5:2.5划分成训练集、验证集和测试集
    #
    # train_deeplog_df_len = int(deeplog_df_len * 0.5)
    # train_deeplog_df = deeplog_df[:train_deeplog_df_len]
    #
    # # 将valid_test_deeplog_df按照5:5划分成验证集和测试集
    # valid_and_test_deeplog_df = deeplog_df[train_deeplog_df_len:]
    # valid_and_test_deeplog_df_len = len(valid_and_test_deeplog_df)
    #
    # # 验证集
    # valid_deeplog_df_len = int(valid_and_test_deeplog_df_len * 0.5)
    # valid_deeplog_df = valid_and_test_deeplog_df[:valid_deeplog_df_len]
    #
    # # 测试集
    # test_deeplog_df = valid_and_test_deeplog_df[valid_deeplog_df_len:]
    # test_deeplog_df_len = len(test_deeplog_df)

    # 将deeplog_df按照3:2:5划分成训练集、验证集和测试集
    deeplog_df_len = len(deeplog_df)
    data_len = deeplog_df_len
    train_len = int(data_len * 0.3)  # 训练集长度
    valid_len = int(data_len * 0.2)  # 验证集长度
    test_len = data_len - train_len - valid_len  # 测试集长度

    train_deeplog_df = deeplog_df.iloc[:train_len]  # 训练集
    valid_deeplog_df = deeplog_df.iloc[train_len:train_len + valid_len]  # 验证集
    test_deeplog_df = deeplog_df.iloc[train_len + valid_len:]  # 测试集

    #########
    # Train #
    #########
    train_deeplog_df_len = int(len(train_deeplog_df))
    train_normal_df = train_deeplog_df[train_deeplog_df["Label"] == 0]
    train_abnormal_df = train_deeplog_df[train_deeplog_df["Label"] == 1]
    # no shuffle
    train_normal_len = int(len(train_normal_df))
    train_abnormal_len = int(len(train_abnormal_df))
    # deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId", "deltaT"])
    deeplog_file_generator(os.path.join(output_dir, 'train'), train_deeplog_df, ["EventId"])
    deeplog_file_generator(os.path.join(output_dir, 'train_normal'), train_normal_df, ["EventId"])
    deeplog_file_generator(os.path.join(output_dir, 'train_abnormal'), train_abnormal_df, ["EventId"])

    print("训练集长度：{}".format(train_deeplog_df_len))
    print("训练集中正常的长度 {}".format(train_normal_len))
    print("训练集中异常的长度 {}".format(train_abnormal_len))


    #########
    # Valid #
    #########
    valid_deeplog_df_len = int(len(valid_deeplog_df))
    valid_df = valid_deeplog_df[valid_deeplog_df["Label"] == 0]
    # no shuffle
    valid_len = int(len(valid_df))
    # deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId", "deltaT"])
    deeplog_file_generator(os.path.join(output_dir, 'valid'), valid_df, ["EventId"])

    print("valid size {}".format(valid_len))
    print("验证集长度:{}".format(valid_deeplog_df_len))
    print("验证集中正常的长度: {}".format(valid_len))

    ###############
    # Test Normal #
    ###############
    test_normal_df = test_deeplog_df[test_deeplog_df["Label"] == 0]
    test_normal_len = int(len(test_normal_df))
    deeplog_file_generator(os.path.join(output_dir, 'test_normal'), test_normal_df, ["EventId"])
    print("test normal size {}".format(test_normal_len))

    del test_normal_df
    del train_deeplog_df
    gc.collect()

    #################
    # Test Abnormal #
    #################
    test_abnormal_df = test_deeplog_df[test_deeplog_df["Label"] == 1]
    test_abnormal_len = int(len(test_abnormal_df))
    deeplog_file_generator(os.path.join(output_dir, 'test_abnormal'), test_abnormal_df, ["EventId"])
    print('test abnormal size {}'.format(test_abnormal_len))

    #########
    # Train #   #原始未变动版本
    #########
    # df_normal = deeplog_df[deeplog_df["Label"] == 0]
    # df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True) #shuffle
    # normal_len = len(df_normal)
    # train_len = int(train_ratio) if train_ratio >= 1 else int(normal_len * train_ratio)
    #
    # train = df_normal[:train_len]
    # deeplog_file_generator(os.path.join(output_dir, 'train'), train, ["EventId"])
    # print("training size {}".format(train_len))


    ###############
    # Test Normal #     #原始未变动版本
    ###############
    # test_normal = df_normal[train_len:]
    # deeplog_file_generator(os.path.join(output_dir, 'test_normal'), test_normal, ["EventId"])
    # print("test normal size {}".format(normal_len - train_len))


    #################
    # Test Abnormal #       #原始未变动版本
    #################
    # df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    # deeplog_file_generator(os.path.join(output_dir, 'test_abnormal'), df_abnormal, ["EventId"])
    # print('test abnormal size {}'.format(len(df_abnormal)))

