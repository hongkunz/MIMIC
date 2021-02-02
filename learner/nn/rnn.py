#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:42:36 2020

@author: zen
"""

import torch
import torch.nn as nn
from .module import StructureNN
import torch.jit as jit
from typing import Tuple
from torch import Tensor

class GradientModule(jit.ScriptModule):
    '''Gradient symplectic module.
    '''
    def __init__(self, ind, width, activation):
        super(GradientModule, self).__init__()
        self.ind = ind
        self.width = width
        self.act = activation
        
        d = self.width // 2
        self.K = nn.Parameter((torch.randn([d, d]) * 0.1).requires_grad_(True))
        self.W = nn.Parameter((torch.randn([self.ind, d]) * 0.1).requires_grad_(True))
        self.a = nn.Parameter((torch.rand([d]) * 2 - 1).requires_grad_(True))
        self.b = nn.Parameter(torch.zeros([d]).requires_grad_(True))
        self.eta = nn.Parameter(torch.zeros([d]).requires_grad_(True))
        torch.nn.init.xavier_uniform(self.K)
        torch.nn.init.xavier_uniform(self.W)
    
    @jit.script_method
    def forward(self, inputs, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        p, q = state
        gradV = (self.act(q @ self.K + inputs @ self.W + self.b)) @ self.K.t()
        return q + self.eta, -p + 0.001 * gradV

class SRU(jit.ScriptModule):
    def __init__(self, ind, width):
        super(SRU, self).__init__()
        self.ind = ind
        self.width = width
        
        self.cell = GradientModule(self.ind, self.width, torch.tanh)
       
    @jit.script_method
    def forward(self, x, init_state):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        p, q = init_state[0].chunk(2, 1)
        inputs = x.unbind(1)
        for i in range(len(inputs)):
            p, q = self.cell(inputs[i], (p, q))
        pq = torch.cat([p,q], dim = 1)
        return torch.stack([pq], dim = 1), pq
        
class ASCell(jit.ScriptModule):
    '''Antisymmetric cell
    '''
    def __init__(self, ind, width, activation, gamma):
        super(ASCell, self).__init__()
        self.ind = ind
        self.width = width
        self.act = activation
        self.gamma = gamma
        
        self.W = nn.Parameter((torch.empty([self.width, self.width])).requires_grad_(True))
        self.V = nn.Parameter((torch.empty([self.ind, self.width])).requires_grad_(True))
        self.b = nn.Parameter(torch.zeros([self.width]).requires_grad_(True))
        torch.nn.init.xavier_uniform(self.W)
        torch.nn.init.xavier_uniform(self.V)
    
    @jit.script_method
    def forward(self, inputs, state):
         # type: (Tensor, Tensor) -> Tensor
        return state + 0.01 * self.act(state @ (self.W - self.W.t()) - self.gamma * state + inputs @ self.V + self.b)

class ASNN(jit.ScriptModule):
    def __init__(self, ind, width):
        super(ASNN, self).__init__()
        self.ind = ind
        self.width = width
        self.gamma = 0.01
        
        self.cell = ASCell(self.ind, self.width, torch.tanh, self.gamma)
       
    @jit.script_method
    def forward(self, x, init_state):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        state = init_state[0]
        inputs = x.unbind(1)
        outputs = []
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs.append(state)
        return torch.stack(outputs, dim = 1), state

class RNN(StructureNN):
    def __init__(self, ind, outd, width, cell, return_all):
        super(RNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.width = width
        self.cell = cell
        self.modus = self.__init_modules()
        self.return_all = return_all   
 
    def forward(self, x):
        to_squeeze = True if len(x.size()) == 2 else False
        if to_squeeze:
            x = x.view(1, self.len_in, self.dim_in)
        zeros = torch.zeros([1, x.size(0), self.width], dtype=x.dtype, device=x.device)
        init_state = (zeros, zeros) if self.cell == 'LSTM' else zeros
        x, _ = self.modus['RNN'](x, init_state)
        if self.return_all:
            y = self.modus['LinMOut'](x)
        else:
            y = self.modus['LinMOut'](x[:,-1])
        return y
    
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.cell == 'RNN':
            modules['RNN'] = nn.RNN(self.ind, self.width, batch_first=True)
        elif self.cell == 'LSTM':
            modules['RNN'] =  nn.LSTM(self.ind, self.width, batch_first=True)
        elif self.cell == 'GRU':
            modules['RNN'] = nn.GRU(self.ind, self.width, batch_first=True)
        elif self.cell == 'SRU':
            modules['RNN'] = SRU(self.ind, self.width)
        elif self.cell == 'ASNN':
            modules['RNN'] = ASNN(self.ind, self.width)
        else:
            raise NotImplementedError
            
        modules['LinMOut'] = nn.Linear(self.width, self.outd)
        return modules
