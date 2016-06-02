###################
# LSTM_NETWORK.PY #
###################

import numpy as np 
from layers import *

class LSTM_Network(object):

	def __init__(self, char_dim, hidden_dim, dtype):
		"""
		Inputs:
			char_dim   - Number of characters involved for training
			hidden_dim - Number of hidden neurons
			dtype      - numpy datatype to use
		"""

		self.dtype = dtype
		self.char_dim = char_dim
		self.hidden_dim = hidden_dim
		self.params = {}

		self.params['Wxh'] = np.random.randn(char_dim, 4 * hidden_dim)
		self.params['Wxh'] /= np.sqrt(char_dim)
		self.params['Whh'] = np.random.randn(hidden_dim, 4 * hidden_dim)
		self.params['Whh'] /= np.sqrt(hidden_dim)
		self.params['b1']  = np.zeros((hidden_dim))
		self.params['Why'] = np.random.randn(hidden_dim, char_dim)
		self.params['Why'] /= np.sqrt(hidden_dim)
		self.params['b2']  = np.zeros(char_dim)


		for k, v in self.params.iteritems():
			self.params[k] = v.astype(self.dtype)


	def loss(self, chars, char_list, h0, temp=1.0):
		"""
		Inputs:
			chars - collection of characters from training data;
					There are T + 1 characters, where T is the number of timesteps we're
					alloting. This is because we will use the 1st to Tth for input and
					2nd to T+1th letters as output
			h0    - An initial hidden state; dimensions (N, H)
		Outputs:
			loss - The scalar loss value of the neural net's classification
			grads - Dictionary of gradients of Wxh, Whh, b1, Why and b2
		"""
		#Preprocessing
		N, H = h0.shape
		T, D = len(chars) - 1, len(char_list)

		loss = 0.0
		grads = {}
		char_vecs = np.zeros((N, T, D))
		idx_matrix = np.zeros((N, T))

		#Load values from our dictionary for easy use
		Wxh = self.params['Wxh']
		Whh = self.params['Whh']
		b1  = self.params['b1']
		Why = self.params['Why']
		b2  = self.params['b2']

		#Split our chars into input and output of equal size now
		input_chars = [:-1]
		output_chars = [1:]

		for n in range(N):
			char_vecs[n,:,:] = convert_chars_to_vec(char_list, input_chars)
			idx_matrix[n,:]  = np.asarray(convert_chars_to_idx(char_list, output_chars)).T

		#Forward pass through dropout layer of Wxh, Whh, b1
		hidden_states, forward_cache = lstm_seq_forward(input_chars, h0, Wxh, Whh, b1)		#Dimensions (N, T, H)
		scores, scores_cache = temporal_forward_pass(hidden_states, Why, b2)				#Dimensions (N, T, D)
		loss, dout = lstm_softmax_loss(scores, idx_matrix, temp)
		dscores, grads['Why'], grads['b2'] = temporal_backward_pass(dout, scores_cache)
		dhidden_states, dh_init, grads['Wxh'], grads['Whh'], grads['b1'] = lstm_seq_backward(dscores, forward_cache)
		#Backprop through a dropout layer on Wxh, Whh, b1

		return loss, grads

	
	def sample(h, c, input_vec, itr, cache):
		"""
		Inputs:
			h         - Contains the current hidden state; dimension (N, H)
			c         - Contains the current cell state; dimension (N, H)
			input_vec - Contains the input one-hot vector for the first letter; dimensions (V,)
			itr       - Number of characters to produce; scalar

		Outputs:
			char_indices - Contains a list of all indices of each character
		"""

		#Pre-processing
		char_list_size = input_vec.shape[0]
		Wxh, Whh, b1 = cache['Wxh'], cache['Whh'] cache['b1']
		Why, b2 = cache['Why'], cache['b2']
		char_indices = []
		X = np.zeros_like(input_vec)

		#Main Loop
		for i in xrange(itr):
			h, c, _ = lstm_step_forward(input_vec, h, c, Wxh, Whh, b1)
			scores, _ = forward_pass(h, Why, b2)
			probs = softmax(out, 1.0)
			index = np.random.choice(range(char_list_size),p=probs.ravel())
			input_vec = np.zeros_like(input_vec)
			input_vec[index] = 1
			char_indices.append(index)

		return char_indices
	
