#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(self,char_embed,word_embed,m_word=21,k=5,kernel_size=5):
		super(CNN,self).__init__()
		self.conv1d=nn.Conv1d(char_embed,word_embed,kernel_size=kernel_size)

	def forward(self,x):
		(x_out,_)=torch.max(F.relu(self.conv1d(x)),dim=2)
		return x_out


### END YOUR CODE

