#################
# VALIDATION.PY #
#################
import numpy as np
from solver import *
from oneHot import *
from LSTM import *
from layers import *
import time

def cross_validate(data, char_dim, hidden_dim, seq_length, dtype, char_list):
	"""
	Inputs:
		data - The full set of data that we will later be fully training over
		char_list - The list of possible characters that might exist
	"""
	#Pre-processing of sizes
	data_size = len(data)
	batch_train_size = int(data_size / 50)
	batch_val_size = int(data_size / 100)
	train_start_index, val_start_index = 0, 0
	best_learning_rate = None
	best_val_acc = 0

	#Create a list of learning_rates we wish to try
	learning_rates = [1e-3, 5e-2, 1e-2, 5e-1, 1e-1]

	#Choose a random starting position for training
	while True:
		train_start_index = np.random.choice(range(data_size))
		if train_start_index + batch_train_size + 1 < len(data):
			break; 

	#Choose a random starting position for validation
	while True:
		val_start_index = np.random.choice(range(data_size))
		if val_start_index + batch_val_size + 1 < len(data):
			break; 

	#Slice the data into train and val set
	batch_data = {}
	batch_data['train'] = data[train_start_index:train_start_index+batch_train_size+1]
	batch_data['val'] = data[val_start_index:val_start_index+batch_val_size+1]

	#Loop over every learning rate choice
	for lr in learning_rates:
		print 'Beginning validation with the learning rate of ' + str(lr)
		
		#Create a model, solver and train the model
		model = LSTM_Network(char_dim, hidden_dim, seq_length, dtype)
		optim_config = {}
		optim_config['learning_rate'] = lr
		solver = Solver(model, batch_data, optim_config)
		solver.train(char_list, 'val')

		#Creates indices
		val_indices = []
		val_len = len(batch_data['val'])
		val_char_vecs = np.zeros((1,val_len, len(char_list)))
		val_char_vecs[0,:,:] = convert_chars_to_vec(char_list, batch_data['val'])

		#Preparing inputs
		h= np.zeros((1, hidden_dim))
		c = np.zeros_like(h)
		Wxh = solver.model.params['Wxh']
		Whh = solver.model.params['Whh']
		Why = solver.model.params['Why']
		b1 = solver.model.params['b1']
		b2 = solver.model.params['b2']

		print 'Starting validation!'
		#Loop over each letter and try to predict the next letter
		for i in range(val_len - 1):
			input_vec = val_char_vecs[0,i,:].reshape(1, len(char_list))
			h, c, _ = lstm_step_forward(input_vec, h, c, Wxh, Whh, b1)
			scores, _ = forward_pass(h, Why, b2)
			probs = softmax(scores, 0.9)
			index = np.random.choice(range(len(char_list)),p=probs.ravel())
			val_indices.append(index)

		#Reform the list of indices into characters, and then to vectors
		predicted_chars = [char_list[i] for i in val_indices]
		predicted_char_vecs = np.zeros((1, val_len - 1, len(char_list)))
		predicted_char_vecs[0,:,:] = convert_chars_to_vec(char_list, predicted_chars)
		actual_char_vecs = val_char_vecs[0, range(1,val_len), :]

		val_acc = np.mean(predicted_char_vecs == actual_char_vecs)
		if (val_acc > best_val_acc):
			best_val_acc = val_acc
			best_learning_rate = lr
		print 'Validation accuracy was: ' + str(val_acc)
		print 'The best validation accuracy is: ' + str(best_val_acc)
		print 'The best learning rate is so far: ' + str(best_learning_rate)

	print 'The best learning rate after looping is: ' + str(best_learning_rate)
	return best_learning_rate









	