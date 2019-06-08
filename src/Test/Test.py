# coding=utf-8
# 统计训练集的 mean 和　std 信息
from sklearn.preprocessing import StandardScaler
import numpy as np


def algorithm():
    np.random.seed(123)
    print('use sklearn')
    # 注：shape of data: [n_samples, n_features]
    data = np.random.randn(5, 4)
    scaler = StandardScaler()
    scaler.fit(data)
    trans_data = scaler.transform(data)
    print('original data: ')
    print(data)
    print('transformed data: ')
    print(trans_data)
    print('scaler info: scaler.mean_: {}, scaler.var_: {}'.format(scaler.mean_, scaler.var_))
    print('\n')

    print('use numpy by self')
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    var = std * std
    print('mean: {}, std: {}, var: {}'.format(mean, std, var))
    # numpy 的广播功能
    another_trans_data = data - mean
    # 注：是除以标准差
    another_trans_data = another_trans_data / std
    print('another_trans_data: ')
    print(another_trans_data)


if __name__ == '__main__':
    algorithm()
