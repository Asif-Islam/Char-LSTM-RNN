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
"""
text_data = open('Aristotle_text.txt', 'r').read()
char_list = list(set(text_data))
print char_list
train_data = {}
train_data['train'] = np.asarray(list(text_data)).reshape(1,len(text_data))
"""

#Values for our LSTM Neural Net
char_dim = len(char_list)
hidden_dim = 512
seq_length = 25
batch_size = 1
dtype = np.float32

#FILL IN DATA HERE
HPconfig = tune_parameters(data, char_dim, hidden_dim, seq_length, batch_size, )
for k, v in self.model.params.iteritems():
	np.savetxt(k + '.csv', v, delimiter=',')

#Now run for true training over the text
LSTM = LSTM_Network(char_dim, hidden_dim, seq_length, batch_size, dtype)
LSTM.qxh = HPconfig['qxh']
LSTM.qxh = HPconfig['qhy']
optim_config = {}
optim_config['learning_rate'] = HPconfig['learning_rate']
solver = Solver(LSTM, data, optim_config)

for k, v in LSTM.params.iteritems():
	print k, v.shape

solver.train(char_list)


