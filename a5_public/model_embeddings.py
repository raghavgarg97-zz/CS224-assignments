#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
from cnn import CNN
import torch
from highway import HighwayNet

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
# from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        pad_token_idx = vocab.char2id['<pad>']
        self.embed_size=embed_size
        self.char_size=50
        self.embeddings=nn.Embedding(len(vocab.char2id),self.char_size,padding_idx=pad_token_idx)
        self.cnn=CNN(self.char_size,self.embed_size)
        self.highway=HighwayNet(self.embed_size)
        self.dropout=nn.Dropout(0.3)
        

        ### YOUR CODE HERE for part 1j


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        
        embed_output=self.embeddings(input)
        sentence_lst=[]

        for i in range(0,embed_output.shape[0]):
            sentence_lst.append(self.cnn(embed_output[i].permute(0,2,1)))
        sentence_lst=torch.stack(sentence_lst,dim=0)
        return self.dropout(self.highway(sentence_lst))
        
        
        ### END YOUR CODE

