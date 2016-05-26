# -*- coding: utf-8 -*-
'''
課題1から課題4までの実装
Python 3.5.1
OSX 10.10.5 Yosemite
'''

from operator import mul, add, sub
from math import sqrt
import random
import sys

def outer(x, y):
    '''
    ベクトルxとベクトルyの外積
    '''
    return [[xi * yj for yj in y] for xi in x]


def affine(x, W, b):
    '''
    アファイン変換  Wx + b
    '''
    for wRow in W:
        assert len(wRow) == len(x)
    assert len(W) == len(b)

    h = [sum(map(mul, wRow, x)) for wRow in W]
    return list(map(add, h, b))


def subtract_M2_from_M1(matrix1, matrix2):
    '''
    2つの行列の引き算  matrix1 - matrix2
    '''
    for row1, row2 in zip(matrix1, matrix2):
        assert len(row1) == len(row2)
    assert len(matrix1) == len(matrix2)

    return [list(map(sub, M1row, M2row)) for M1row, M2row in zip(matrix1, matrix2)]


def multiply_matrix_by_a(matrix, a):
    '''
    行列matrixのすべての要素をa倍  a * matrix
    '''
    return [[a * element  for element in row] for row in matrix]


def auto_encoder(file_path, hidden_dim_num=5, epoch_num=1000):
    '''
    Auto Encoder
    隠れ層1層
    file_path        データセットファイルへのパス
    hidden_dim_num   隠れ層の次元
    epoch_num        イテレーション回数

    return           最終的なパラメータ W1, W2, b1, b2のディクショナリ
    '''
    data = data_from_file(file_path)
    # normalized_data = normalization(data['data'])
    elements_num = data['N']
    params = initialize_params_randomly(hidden_dim_num, data['D'])

    for i in range(epoch_num):
        sys.stderr.write('\r\033[K' + str(i))
        sys.stderr.flush()
        for x in data['data']:
            params['x'] = x
            params_new = learn_one_iteration(**params)
            params.update(params_new)

    del params['eta']
    total_loss = 0.0
    for x in data['data']:
        params['x'] = x
        params_forward = forward(**params)
        total_loss += params_forward['loss']

    average_loss = total_loss / elements_num
    params['loss'] = average_loss
    return params


def learn_one_iteration(x, W1, W2, b1, b2, eta):
    '''
    学習の1イテレーション
    return          学習したパラメータ
    '''
    args = {'x': x, 'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    args_forward = args.copy()
    args_bp = args.copy()
    args['eta'] = eta
    del args['x']

    forward_result = forward(**args_forward)
    #loss = forward_result['loss']
    del forward_result['loss']
    args_bp.update(forward_result)

    bp_result = back_propagation(**args_bp)
    args.update(bp_result)

    param_result = param_update(**args)
    #param_result['loss'] = loss
    return param_result


def forward(x, W1, W2, b1, b2):
    '''
    順伝播
    '''
    h = affine(x, W1, b1)
    y = affine(h, W2, b2)
    assert len(x) == len(y)
    loss = 1.0/2.0 * sum(map(lambda e: e**2, map(sub, x, y)))
    return {'h': h, 'y': y, 'loss': loss}


def back_propagation(x, h, y, W1, W2, b1, b2):
    '''
    逆伝播
    '''
    gy = list(map(sub, y, x))
    gW2 = outer(gy, h)
    W2T = list(map(list, zip(*W2)))
    gh = [sum(map(mul, wRow, gy)) for wRow in W2T]
    gW1 = outer(gh, x)
    return {'gW1': gW1, 'gW2': gW2, 'gb1': gh, 'gb2': gy}


def param_update(W1, W2, b1, b2, gW1, gW2, gb1, gb2, eta):
    '''
    パラメータの更新
    '''
    etaTimesW1 = multiply_matrix_by_a(gW1, eta)
    W1new = subtract_M2_from_M1(W1, etaTimesW1)
    etaTimesW2 = multiply_matrix_by_a(gW2, eta)
    W2new = subtract_M2_from_M1(W2, etaTimesW2)
    b1new = list(map(sub, b1, list(map(lambda x: eta * x, gb1))))
    b2new = list(map(sub, b2, list(map(lambda x: eta * x, gb2))))
    return {'W1': W1new, 'W2': W2new, 'b1': b1new, 'b2': b2new}


def data_from_file(file_path):
    '''
    file_pathから訓練データを取得
    '''
    data = []
    for line in open(file_path, 'r'):
        data.append(list(map(float, line.split())))
    ans = {'N': int(data[0][0]), 'D': int(data[0][1]), 'data':data[1:]}
    return ans


def initialize_params_randomly(hidden_dim_num, data_dim_num):
    '''
    パラメータの初期化
    2つのバイアスの要素すべて0，2つの行列の要素は-0.05~0.05でランダム
    学習率etaは0.01
    '''
    W1 = []
    W2 = []
    b1 = [0.0] * hidden_dim_num
    b2 = [0.0] * data_dim_num
    for i in range(hidden_dim_num):
        W1row = []
        for j in range(data_dim_num):
            W1row.append(random.uniform(-0.05, 0.05))
        W1.append(W1row)

    for i in range(data_dim_num):
        W2row = []
        for j in range(hidden_dim_num):
            W2row.append(random.uniform(-0.05, 0.05))
        W2.append(W2row)

    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2, 'eta': 0.01}


def normalization(dataset):
    '''
    データを正規化
    (x - min) / (max - min)
    '''
    maxV = []
    minV = []
    datasetT = list(map(list, zip(*dataset)))

    for data in datasetT:
        maxV.append(max(data))
        minV.append(min(data))

    return [list(map(lambda x, max, min: (x - min) / (max - min), data, maxV, minV)) for data in dataset]


def print_params2file(params, file_path):
    '''
    file_pathに対してparamsをスペース区切りで保存
    '''
    f = open(file_path,'w')

    for Wrow in params['W1']:
        f.write(' '.join(list(map(lambda x: str(x), Wrow))) + '\n')
    for Wrow in params['W2']:
        f.write(' '.join(list(map(lambda x: str(x), Wrow))) + '\n')
    f.write(' '.join(list(map(lambda x: str(x), params['b1']))) + '\n')
    f.write(' '.join(list(map(lambda x: str(x), params['b2']))) + '\n')
    f.close()


if __name__ == '__main__':
    params = auto_encoder('./data/dataset.dat')
    print(params['loss'])
    print_params2file(params, './results/result.txt')
