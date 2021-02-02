#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:41:44 2021

@author: zen
"""
import numpy as np
import learner as ln
from learner.utils import accuracy

def callback(data, net):
    X_test, y_test = data.load_test()
    print('{:<9}Accuracy: {:<25}'.format(
        '', accuracy(net(X_test), y_test)), flush=True)

class MIMICData(ln.Data):
    def __init__(self):
        super(MIMICData, self).__init__()
        self.X_train = np.load('data/train.npy')
        self.y_train = np.load('data/train_label.npy')
        self.X_test = np.load('data/test.npy')
        self.y_test = np.load('data/test_label.npy')  
        
def main():
    device = 'cpu' # 'cpu' or 'gpu'
    net_type = 'LSTM' # 'LSTM' or 'GRU' or 'RNN'
    width = 20
    ind = 96 # input dimension (n if the input is U1, U2, ... Un)
    outd = 1 # output dimension (k if the input is S1, S2, ... Sk)
    # training
    lr = 0.01
    iterations = 100000
    print_every = 1000
    batch_size = 100
    return_all = False
    
    criterion = 'Sigmoid'
    data = MIMICData()
    net = ln.nn.RNN(ind, outd, width, net_type, return_all)
    args = {
        'data': data,
        'net': net,
        'criterion': criterion,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': callback,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run_rnn()
    ln.Brain.Restore()
    ln.Brain.Output(path = './outputs/lstm')

if __name__ == '__main__':
    main()