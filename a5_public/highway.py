#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayNet(nn.Module):
	def __init__(self,input_size):
		super(HighwayNet,self).__init__()
		self.input_size=input_size
		self.fc1=nn.Linear(self.input_size,self.input_size)
		self.gate=nn.Linear(self.input_size,self.input_size)

	def forward(self,x):
		x_proj=F.relu(self.fc1(x))
		x_gate=torch.sigmoid(self.gate(x))
		x_highway=(x_gate * x_proj)+((1-x_gate) * x)
		return x_highway

### END YOUR CODE 

