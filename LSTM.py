###################
# LSTM_NETWORK.PY #
###################

import numpy as np 
from layers import *
from oneHot import *

class LSTM_Network(object):

	def __init__(self, char_dim, hidden_dim, seq_length, batch_size, dtype):
		"""
		Inputs:
			char_dim   - Number of characters involved for training
			hidden_dim - Number of hidden neurons
			seq_length - Size of batch used for forward/backward prop
			dtype      - numpy datatype to use
		"""

		self.dtype = dtype
		self.char_dim = char_dim
		self.hidden_dim = hidden_dim
		self.seq_length = seq_length
		self.batch_size = batch_size
		self.current_hidden_state = np.zeros((batch_size,hidden_dim));
		self.params = {}

		self.params['Wxh'] = np.random.randn(char_dim, 4 * hidden_dim)
		self.params['Wxh'] /= np.sqrt(char_dim)
		self.params['Whh'] = np.random.randn(hidden_dim, 4 * hidden_dim)
		self.params['Whh'] /= np.sqrt(hidden_dim)
		self.params['b1']  = np.zeros((4*hidden_dim))
		self.params['Why'] = np.random.randn(hidden_dim, char_dim)
		self.params['Why'] /= np.sqrt(hidden_dim)
		self.params['b2']  = np.zeros(char_dim)


		for k, v in self.params.iteritems():
			self.params[k] = v.astype(self.dtype)


	def loss(self, chars, char_list, h0,  mode, temp=1.0):
		"""
		Inputs:
			chars - Numpy Array of characters to train over; dimensions (N, T)
					where N is the batch size and S is the sequence length
			h0    - An initial hidden state; dimensions (N, H)
			mode -  Dictionary for Zoneout mode 
		Outputs:
			loss - The scalar loss value of the neural net's classification
			grads - Dictionary of gradients of Wxh, Whh, b1, Why and b2
			hprev - The last hidden state, acting as the "initial state" for the next batch
		"""
		#Preprocessing
		N, H = h0.shape
		T, D = chars.shape[1] - 1, len(char_list)

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
		input_chars = chars[:,:-1]
		output_chars = chars[:,1:]

		char_vecs  = convert_chars_to_vec(char_list, input_chars)
		idx_matrix = convert_chars_to_idx(char_list, output_chars).T

		#Forward pass through dropout layer of Wxh, Whh, b1
		hidden_states, forward_cache = lstm_seq_forward(char_vecs, h0, Wxh, Whh, b1, mode)		#Dimensions (N, T, H)
		scores, scores_cache = temporal_forward_pass(hidden_states, Why, b2)				#Dimensions (N, T, D)
		loss, dout = lstm_softmax_loss(scores, idx_matrix, temp)
		dscores, grads['Why'], grads['b2'] = temporal_backward_pass(dout, scores_cache)
		dhidden_states, dh_init, grads['Wxh'], grads['Whh'], grads['b1'] = lstm_seq_backward(dscores, forward_cache, mode)
		#Backprop through a dropout layer on Wxh, Whh, b1

		hprev = hidden_states[:,-1,:].reshape(N, H)

		return loss, grads, hprev

	
	def sample(self,h, c, input_vec, itr, cache, mode='sample'):
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
		char_list_size = input_vec.shape[1]
		Wxh, Whh, b1 = cache['Wxh'], cache['Whh'], cache['b1']
		Why, b2 = cache['Why'], cache['b2']
		char_indices = []
		X = np.zeros_like(input_vec)
		char_indices.append(np.argmax(input_vec))
		seq_mode = {}
		seq_mode['pass'] = 'val'
		#Main Loop
		for i in xrange(itr):
			h, c, _ = lstm_step_forward(input_vec, h, c, Wxh, Whh, b1, seq_mode)
			h *= 0.5
			c *= 0.05
			scores, _ = forward_pass(h, Why, b2)
			probs = softmax(scores, 1.0)
			index = np.random.choice(range(char_list_size),p=probs.ravel())
			input_vec = np.zeros_like(input_vec)
			input_vec[:,index] = 1
			char_indices.append(index)

		return char_indices
	
