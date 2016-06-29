################
# LSTM_MAIN.PY #
################

"""


"""



import numpy as np
from LSTM import *
from layers import *
from solver import *
from oneHot import *
from optim import *
from validation import *
from ABCKeyChange import *
import os

NUM_SONGS = 490
MAX_LEGNTH = 3109

char_list = []
song_itr_count = 0

#Load all of our songs and filter them
path = 'C:\Users\Asif\Documents\Deep Learning\DeepVGM\qabc_train_data'

for filename in os.listdir(path):
	song_path = path + "\\" + filename
	song_filtered =  filterABC(song_path, False)
	char_list = list(set(char_list) | set(song_filtered))

	if song_itr_count == 0:
		train_data = np.asarray(list(song_filtered + '@' * (MAX_LEGNTH - len(song_filtered))))
	else:
		song_string = np.asarray(list(song_filtered + '@' * (MAX_LEGNTH - len(song_filtered))))
		train_data = np.vstack((train_data, song_string))

	song_itr_count = song_itr_count + 1

char_list.append('@')

#Values for our LSTM Neural Net
char_dim = len(char_list)
seq_length = 25
batch_size = 35
dtype = np.float32

#Grid Search over to find our best hyper parameters
HPconfig = tune_parameters(train_data, char_dim, seq_length, batch_size, dtype, char_list)
stats = np.array([HPconfig['hidden_dim'],HPconfig['learning_rate'],HPconfig['qxh'],HPconfig['qhy']])
np.savetxt('HPconfig.csv', stats, delimiter = ',')


#Now run for true training over the text
LSTM = LSTM_Network(char_dim, HPconfig['hidden_dim'], seq_length, batch_size, dtype)
LSTM.qxh = HPconfig['qxh']
LSTM.qxh = HPconfig['qhy']
optim_config = {}
optim_config['learning_rate'] = HPconfig['learning_rate']
solver = Solver(LSTM, train_data, optim_config)
solver.train(char_list)


