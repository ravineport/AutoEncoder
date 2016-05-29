# -*- coding: utf-8 -*-
'''
AutoEncoderSigmoidクラス
Python 3.5.1
OSX 10.10.5 Yosemite
'''

import numpy as np
import random
import sys
import data
import autoencoder as ae

class AutoEncoderSigmoid(ae.AutoEncoder):
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

    def back_propagation(self):
        '''
        逆伝播
        '''
        self.gy  = self.y - self.x
        self.gb2 = self.gy
        self.gW2 = np.outer(self.gy, self.h)
        self.gh  = self.W2.transpose().dot(self.gy) * sigmoid(self.h) * (1 - sigmoid(self.h))
        self.gb1 = self.gh
        self.gW1 = 2.0 * np.outer(self.gh, self.x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':
    ae = AutoEncoderSigmoid('./data/dataset.dat')
    ae.calc_loss_and_params()
    data.print_params2file(ae.get_best_params(), './results/result_test.txt')
