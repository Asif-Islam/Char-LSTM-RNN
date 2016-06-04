################
# LSTM_MAIN.PY #
################

import numpy as np
from LSTM import *
from layers import *
from solver import *
from oneHot import *
from optim import *

#Read the data and create our list of all possible characters
text_data = open('shakespeare.txt', 'r').read()
char_list = list(set(text_data))
print char_list
train_data = {}
train_data['train'] = text_data
train_data['val'] = None

#Values for our LSTM Neural Net
char_dim = len(char_list)
hidden_dim = 256
seq_length = 25
dtype = np.float32

LSTM = LSTM_Network(char_dim, hidden_dim, seq_length, dtype)
solver = Solver(LSTM, train_data, {})

for k, v in LSTM.params.iteritems():
	print k, v.shape
solver.train(char_list, 'train')


