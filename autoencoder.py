# -*- coding: utf-8 -*-
'''
AutoEncoderクラス
Python 3.5.1
OSX 10.10.5 Yosemite
'''

import numpy as np
from operator import mul, add, sub
from math import sqrt
import random
import sys

class AutoEncoder:
    def __init__(self, data_file_path, hidden_dim_num=5, epoch_num=1000):
        self.data = data_from_file(data_file_path)
        self.hidden_dim_num = hidden_dim_num
        self.epoch = epoch_num

    def get_params(self):
        return {'W1': self.W1.tolist(), 'W2': self.W2.tolist(), 'b1': self.b1.tolist(), 'b2': self.b2.tolist()}

    def auto_encoder(self):
        '''
        Auto Encoder
        隠れ層1層

        return           最終的なパラメータ W1, W2, b1, b2のディクショナリ
        '''
        elements_num = self.data['N']
        self.initialize_params_randomly()

        for i in range(self.epoch):
            sys.stderr.write('\r\033[K' + str(i))
            sys.stderr.flush()
            for x in self.data['data']:
                self.x = x
                self.learn_one_iteration()
        print("")

        total_loss = 0.0
        for x in self.data['data']:
            self.x = x
            self.forward()
            total_loss += self.loss

        self.average_loss = total_loss / elements_num
        print(self.average_loss)

    def learn_one_iteration(self):
        '''
        学習の1イテレーション
        return          学習したパラメータ
        '''
        self.forward()
        self.back_propagation()
        self.param_update()

    def forward(self):
        '''
        順伝播
        '''
        self.h = self.W1.dot(self.x) + self.b1
        self.y = self.W2.dot(self.h) + self.b2
        self.loss = np.sum((self.x - self.y) ** 2) / 2.0

    def back_propagation(self):
        '''
        逆伝播
        '''
        self.gy  = self.y - self.x
        self.gb2 = self.gy
        self.gW2 = np.outer(self.gy, self.h)
        self.gh  = self.W2.transpose().dot(self.gy)
        self.gb1 = self.gh
        self.gW1 = np.outer(self.gh, self.x)

    def param_update(self):
        '''
        パラメータの更新
        '''
        self.W2 -= self.eta * self.gW2
        self.b2 -= self.eta * self.gb2
        # print(self.W1)
        # print(self.gW1)

        self.W1 -= self.eta * self.gW1
        self.b1 -= self.eta * self.gb1

    def initialize_params_randomly(self):
        '''
        パラメータの初期化
        2つのバイアスの要素すべて0，2つの行列の要素は-0.05~0.05でランダム
        学習率etaは0.01
        '''
        data_dim_num = self.data['D']
        W1 = []
        W2 = []
        b1 = [0.0] * self.hidden_dim_num
        b2 = [0.0] * data_dim_num
        for i in range(self.hidden_dim_num):
            W1row = []
            for j in range(data_dim_num):
                W1row.append(random.uniform(-0.05, 0.05))
            W1.append(W1row)

        for i in range(data_dim_num):
            W2row = []
            for j in range(self.hidden_dim_num):
                W2row.append(random.uniform(-0.05, 0.05))
            W2.append(W2row)

        self.W1 = np.array(W1)
        self.W2 = np.array(W2)
        self.b1 = np.array(b1)
        self.b2 = np.array(b2)
        self.eta = 0.01
        return


def data_from_file(file_path):
    '''
    file_pathから訓練データを取得
    '''
    data = []
    for line in open(file_path, 'r'):
        data.append(list(map(float, line.split())))
    ans = {'N': int(data[0][0]), 'D': int(data[0][1]), 'data':np.array(data[1:])}
    return ans


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
    ae = AutoEncoder('./data/dataset.dat')
    ae.auto_encoder()
    print_params2file(ae.get_params(), './results/result_test.txt')
