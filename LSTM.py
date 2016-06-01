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


	def loss(self, chars, h0 = None):	#NOTE, I HAVE TO CALL THIS FUNCTION BY SENDING IN THE COMPLETE MATRIX OF CHAR-VECS
		"""
		Inputs:
			chars - collection of characters from training data; dimensions (N, T + 1, D)
					There are T + 1 characters, where T is the number of timesteps we're
					alloting. This is because we will use the 1st to Tth for input and
					2nd to T+1th letters as output
			h0    - An initial hidden state
		"""
		#Prepare our outputs
		loss = 0.0
		grads = {}

		#Load values from our dictionary for easy use
		Wxh = self.params['Wxh']
		Whh = self.params['Whh']
		b1  = self.params['b1']
		Why = self.params['Why']
		b2  = self.params['b2']

		#Initialize initial hidden state if it does not exist
		N, _, _ = chars.shape
		if h0 is None:
			h0 = np.random.randn(N, self.hidden_dim)

		#Split our chars into input and output of equal size now
		input_chars = [:, :-1, :]	#Dimensions (N, T, D)
		output_chars = [:, 1:, :] 	#Dimensions (N, T, D)



		hidden_states, forward_caches = lstm_seq_forward(input_chars, h0, Wxh, Whh, b1)		#Dimensions (N, T, H)
		scores, scores_cache = temporal_forward_pass(hidden_states, Why, b2)				#Dimensions (N, T, D)
		#loss, dscores = lstm_softmax_loss(scores, ....)			#Reconsider lstm_softmax_los
		

		
		pass

	
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
	
