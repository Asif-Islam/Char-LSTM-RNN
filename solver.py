#############
# SOLVER.PY #
#############

import numpy as np
from layers import *

class Solver(object):

	"""
	The solver class is responsible for all duties related to training 
	our LTSM neural net. 
	"""

	def __init__():
		pass

	def _reset():
		pass

	def _step():
		pass

	def check_accuracy():
		pass

	def sample(h, c, input_vec, itr, cache):
		"""
		Inputs:
			h - Contains the current hidden state; dimension (N, H)
			c - Contains the current cell state; dimension (N, H)
			input_vec - Contains the input one-hot vector for the first letter; dimensions (V,)
			itr - Number of characters to produce; scalar

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
			probs = softmax(out)
			index = np.random.choice(range(char_list_size),p=probs.ravel())
			input_vec = np.zeros_like(input_vec)
			input_vec[index] = 1
			char_indices.append(index)

		return char_indices



	def train():
		pass


