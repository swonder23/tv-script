# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:55:01 2020

@author: Steven
"""

##################
## GET THE DATA ##
##################

# load in data
import helper
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)




######################
## EXPLORE THE DATA ##
######################

view_line_range = (0, 10)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))




########################################
## Implement Pre-processing Functions ##
########################################

import problem_unittests as tests


# create a list of words
words = text.split()
words[:50]
words.index("jerry:")
# /n is not kept when splitting with split function. When running, token_lookup() will be ran first, replacing /n by ||Return||, it's how we will capture
# the line breaks
# ALSO, why do we need "text = text[81:] in the helper file?" put that in comments?


from collections import Counter


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(text)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    dict_tuple = (vocab_to_int, int_to_vocab)

    # return vocab_to_int, int_to_vocab
    return dict_tuple

# vocab_to_int, int_to_vocab = create_lookup_tables(words)
dict_tuple = create_lookup_tables(words)
tests.test_create_lookup_tables(create_lookup_tables)



def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    
    tokendict = {
      '.': '||Period||',
      ',': '||Comma||',
      '"': '||Quotation_mark||',
      ';': '||Semicolon||',
      '!': '||Exclamation_mark||',
      '?': '||Question_mark||',
      '(': '||Left_Parenthese||',
      ')': '||Right_Parenthese||',
      '-': '||Dash||',
      '\n': '||Return||'
    }
        
    return tokendict

token_dict = token_lookup()
tests.test_tokenize(token_lookup)



##########################################
## PRE-PROCESS ALL THE DATA AND SAVE IT ##  
##########################################

# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)



################
## CHECKPOINT ##  
################

import helper
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()



####################
## INPUT BATCHING ##  
####################

import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
    
    
    


###########################   
#### development tests ####

# words = int_text
# sequence_length = 5 ## learn 5 words to generate the 6th
# batch_size = 10 ## rnn will be trained iteratively with batches of 10 random mini-batches of words sequences of length 5 

# batch_size_total = batch_size * sequence_length
# # total number of batches we can make
# n_batches = len(words)//batch_size_total

# # Keep only enough words to make full batches
# words = words[:n_batches * batch_size_total]
# # Reshape into batch_size rows
# words = np.asarray(words)
# words.shape
# words = words.reshape((batch_size, -1))
 


# # iterate through the array, one sequence at a time
# # for n in range(0, words.shape[1], sequence_length):
# for n in range(0, 20, sequence_length):
#     # The features
#     x = words[:, n:n+sequence_length] ## all the rows, extract columns between n and n+sequence_length
#     # The targets, shifted by one
#     y = np.zeros_like(x)
#     try:
#         y[:, :-1], y[:, -1] = x[:, 1:], words[:, n+sequence_length]
#         # y[:, :-1] --> all the rows, all columns except the last one
#         # y[:, -1] --> all the rows, but extract only last column
#     except IndexError:
#         y[:, :-1], y[:, -1] = x[:, 1:], words[:, 0] ## when we reach the last line, we go back to the start of the script
#     yield x, y
        
        

# def get_batches(arr, batch_size, seq_length):
#     '''Create a generator that returns batches of size
#        batch_size x seq_length from arr.
       
#        Arguments
#        ---------
#        arr: Array to make batches from
#        batch_size: Batch size, the number of sequences per batch
#        seq_length: Number of encoded words in a sequence
#     '''
    
#     batch_size_total = batch_size * seq_length
#     # total number of batches we can make
#     n_batches = len(arr)//batch_size_total
    
#     # Keep only enough words to make full batches
#     arr = arr[:n_batches * batch_size_total]
#     # Reshape into batch_size rows
#     arr = arr.reshape((batch_size, -1))
    
#     # iterate through the array, one sequence at a time
#     for n in range(0, arr.shape[1], seq_length):
#         # The features
#         x = arr[:, n:n+seq_length]
#         # The targets, shifted by one
#         y = np.zeros_like(x)
#         try:
#             y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
#         except IndexError:
#             y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
#         yield x, y

# test_batches = get_batches(np.asarray(int_text), batch_size=10, seq_length=5)
# x, y = next(test_batches)



def get_feature_target(arr, batch_size, seq_length):
    '''Create a function that returns feature and target arrays with the right content
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Input array to make feature/target arrays from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded words in a sequence
    '''
    
    # total number of batches 
    n_batches = len(arr)//batch_size
    
    # keep only enough words to make full batches
    arr = arr[:n_batches * batch_size]
    # Reshape into batch_size rows
    arr = arr.reshape(batch_size, -1)
    
    # total number of batches we can make to avoid index issues
    n_batches_max = n_batches - seq_length
    
    # initialize feature and target arrays
    feature = np.zeros(shape=(n_batches_max * batch_size, seq_length))
    target = np.zeros(shape=(n_batches_max * batch_size, 1))
    
    # iterate through the input array, one sequence at a time
    for n in range(0, n_batches_max - 1):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets
        y = arr[:, n+seq_length]
        y = y[:, None]

        feature[(batch_size * n):(batch_size * (n+1)), :] = x
        target[(batch_size * n):(batch_size * (n+1)), :] = y
        
    # convert target 2D array to 1D
    target = target.flatten()
    
    return feature, target

feature, target = get_feature_target(np.asarray(int_text), batch_size=10, seq_length=5)


# arr = np.asarray(int_text)
# batch_size=10
# seq_length=5
# n=0
# last possible n must be
# n=89207




from torch.utils.data import TensorDataset, DataLoader

def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    
    # generate feature and target arrays
    feature, target = get_feature_target(words, batch_size, sequence_length)
    # create a known format for accessing our data
    data = TensorDataset(torch.from_numpy(feature), torch.from_numpy(target))  
    # return a dataloader
    data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    
    return data_loader



test_data_loader = batch_data(np.asarray(int_text), sequence_length=5, batch_size=10)


# test dataloader

data_iter = iter(test_data_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)

print(sample_y.shape)
print(sample_y)




##############################
## BUILD THE NEURAL NETWORK ##  
##############################

import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

     
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=dropout, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        
        ## define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)
        
        # embeddings and lstm_out
        nn_input = nn_input.long()  
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # reshape into (batch_size, seq_length, output_size)
        out = out.view(batch_size, -1, self.output_size)
        # get last batch
        out = out[:, -1]  

        # return one batch of output word scores and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        
        # initialize hidden state with zero weights, and move to GPU if available

        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


tests.test_rnn(RNN, train_on_gpu)





def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # move data to GPU, if available
    if(train_on_gpu):
        inp, target = inp.cuda(), target.cuda()
    
    # perform backpropagation and optimization
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    hidden = tuple([each.data for each in hidden])
    
    # zero accumulated gradients
    rnn.zero_grad()
    
    # get the output from the model
    output, h = rnn(inp, hidden)
    
    # calculate the loss and perform backprop
    loss = criterion(output.squeeze(), target.long())
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    clip=5 # gradient clipping
    nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()
    loss_value = loss.item()

    # return the loss over a batch and the hidden state produced by our model
    return loss_value, h


tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)



