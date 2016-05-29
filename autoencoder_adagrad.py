# -*- coding: utf-8 -*-
'''
AutoEncoderAdagradクラス
Python 3.5.1
OSX 10.10.5 Yosemite
'''

import numpy as np
import random
import sys
import data
import autoencoder as ae

class AutoEncoderAdagrad(ae.AutoEncoder):
    def __init__(self, data_file_path, hidden_dim_num=5, epoch_num=1000):
        ae.AutoEncoder.__init__(self, data_file_path, hidden_dim_num, epoch_num)
        self.eta = 0.1

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

        for i in range(k):
            self.initialize_params_randomly()
            self.init_grad_sum()
            test_data = self.data['data'][i]
            train_data = np.vstack(np.delete(self.data['data'], i, 0))

            for j in range(self.epoch):
                sys.stderr.write('\r\033[K' + str(i) + " " +str(j))
                sys.stderr.flush()
                np.random.shuffle(train_data)
                for x in train_data:
                    self.x = x
                    self.learn_one_iteration()
            print("")

            total_loss = 0.0
            for x in test_data:
                self.x = x
                self.forward()
                total_loss += self.loss

            self.average_loss[i] = total_loss / test_data_num
            if i == 0 or self.average_loss[i] < self.average_loss[i-1]:
                self.updata_best_params()

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
        self.gb2 = self.y - self.x
        self.gW2 = np.outer(self.gb2, self.h)
        self.gb1 = self.W2.transpose().dot(self.gb2)
        self.gW1 = np.outer(self.gb1, self.x)

        self.gb2_sum += self.gb2 ** 2
        self.gW2_sum += self.gW2 ** 2
        self.gb1_sum += self.gb1 ** 2
        self.gW1_sum += self.gW1 ** 2

    def param_update(self):
        '''
        パラメータの更新
        '''
        self.W2 -= self.eta / (np.sqrt(self.gW2_sum + 1)) * self.gW2
        self.b2 -= self.eta / (np.sqrt(self.gb2_sum + 1)) * self.gb2
        self.W1 -= self.eta / (np.sqrt(self.gW1_sum + 1)) * self.gW1
        self.b1 -= self.eta / (np.sqrt(self.gb1_sum + 1)) * self.gb1

    def init_grad_sum(self):
        self.gb2_sum = np.zeros(self.data['D'])
        self.gW2_sum = np.zeros((self.data['D'], self.hidden_dim_num))
        self.gb1_sum = np.zeros(self.hidden_dim_num)
        self.gW1_sum = np.zeros((self.hidden_dim_num, self.data['D']))


if __name__ == '__main__':
    ae = AutoEncoderAdagrad('./data/dataset.dat')
    ae.calc_loss_and_params()
    data.print_params2file(ae.get_best_params(), './results/result_test.txt')
