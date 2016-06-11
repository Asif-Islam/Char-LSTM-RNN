################
# LSTM_MAIN.PY #
################

import numpy as np
from LSTM import *
from layers import *
from solver import *
from oneHot import *
from optim import *
from validation import *

#Read the data and create our list of all possible characters
"""This is for text??"""
text_data = open('Aristotle_text.txt', 'r').read()
char_list = list(set(text_data))
print char_list
train_data = {}
train_data['train'] = np.asarray(list(text_data)).reshape(1,len(text_data))


"""---"""

"""This is for songs"""

"""---"""




#Values for our LSTM Neural Net
char_dim = len(char_list)
hidden_dim = 512
seq_length = 25
dtype = np.float32


#First run validation to determine the best learning rate
#learning_rate = cross_validate_text(text_data, char_dim, hidden_dim, seq_length, dtype, char_list)
learning_rate = 0.01
lr_data = open('learning_rate.txt', 'w')
lr_data.write(str(learning_rate))
lr_data.close()
#Now run for true training over the text
LSTM = LSTM_Network(char_dim, hidden_dim, seq_length, 1, dtype)
optim_config = {}
optim_config['learning_rate'] = learning_rate
solver = Solver(LSTM, train_data, optim_config)

for k, v in LSTM.params.iteritems():
	print k, v.shape
mode = {}
mode['pass'] = 'train'
mode['zh'] = 0.5
mode['zc'] = 0.05
solver.train(char_list, mode)


