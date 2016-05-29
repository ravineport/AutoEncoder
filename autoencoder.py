# -*- coding: utf-8 -*-
'''
AutoEncoderクラス
Python 3.5.1
OSX 10.10.5 Yosemite
'''

import numpy as np
import random
import sys
import data
import time

class AutoEncoder:
    def __init__(self, data_file_path, hidden_dim_num=5, epoch_num=1000):
        self.data = data.data_from_file(data_file_path)
        self.hidden_dim_num = hidden_dim_num
        self.epoch = epoch_num

    def get_params(self):
        return {'W1': self.W1.tolist(), 'W2': self.W2.tolist(), 'b1': self.b1.tolist(), 'b2': self.b2.tolist()}

    def get_best_params(self):
        return {'W1': self.bestW1.tolist(), 'W2': self.bestW2.tolist(), 'b1': self.bestb1.tolist(), 'b2': self.bestb2.tolist()}

    def divide_data(self, k):
        return np.vsplit(self.data['data'], k)

    def updata_best_params(self):
        self.bestW1 = self.W1
        self.bestW2 = self.W2
        self.bestb1 = self.b1
        self.bestb2 = self.b2

    def calc_loss_and_params(self, k=10):
        '''
        Auto Encoder
        隠れ層1層
        k分割交差検定

        return           最終的なパラメータ W1, W2, b1, b2のディクショナリ
        '''
        test_data_num = self.data['N'] / k
        self.data['data'] = self.divide_data(k)
        self.average_loss = [0] * k
        self.elapsed_time = [0] * k

        for i in range(k):
            self.initialize_params_randomly()
            test_data = self.data['data'][i]
            train_data = np.vstack(np.delete(self.data['data'], i, 0))

            start = time.time()
            for j in range(self.epoch):
                sys.stderr.write('\r\033[K' + str(i) + " " +str(j))
                sys.stderr.flush()
                np.random.shuffle(train_data)
                for x in train_data:
                    self.x = x
                    self.learn_one_iteration()

            elapsed_time = time.time() - start
            self.elapsed_time[i] = elapsed_time
            print("")

            total_loss = 0.0
            for x in test_data:
                self.x = x
                self.forward()
                total_loss += self.loss

            self.average_loss[i] = total_loss / test_data_num
            if i == 0 or self.average_loss[i] < self.average_loss[i-1]:
                self.updata_best_params()

        print("elapsed_time:{0}".format(self.elapsed_time))
        ave_time = sum(self.elapsed_time) / float(len(self.elapsed_time))
        print("elapsed_time_ave:{0}".format(ave_time))
        print(self.average_loss)
        print(sum(self.average_loss) / float(len(self.average_loss)))


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


if __name__ == '__main__':
    ae = AutoEncoder('./data/dataset.dat')
    ae.calc_loss_and_params()
    data.print_params2file(ae.get_best_params(), './results/result_test.txt')
