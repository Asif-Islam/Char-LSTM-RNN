#############
# SOLVER.PY #
#############

import numpy as np
from layers import *
from optim import *

class Solver(object):

	"""
	The solver class is responsible for all duties related to training 
	our LTSM neural net. 
	"""

	def __init__(self, model, train_data, optm_config):

		self.model = model
		self.X_train = train_data['X_train']
		self.y_train = train_data['y_train']
		self.X_val = train_data['X_val']
		self.y_val = train_data['y_val']
		self.optim_config = optim_config
		self.epochs = model.seq_length				#Have to think about the implications of this
		self.print_interval = model.seq_length

		self.reset()

	def reset(self):
		"""
		The reset function refreshes storage parameters of our solver to an initial state
		"""

		self.epoch = 0
		self.best_val_acc = 0
		self.best_params = {}
		self.past_losses = []
		self.past_train_acc = []
		self.past_val_acc = []

		self.optim_configs = {}
		for p in self.model.params;
      		d = {k: v for k, v in self.optim_config.iteritems()}
      		self.optim_configs[p] = d 

	
	def step(self, chars, h0, temp=1.0):
		"""
		Inputs:
			chars - The list batch of characters that we are forward and backward propagating over; dimensions
			h0 - The initial state coming into the batch of character; dimensions (N, H)
			temp - The softmax temperature scale, default to 1.0

		"""
		loss, grads, hprev = self.model.loss(chars, self.char_list, h0, temp)
		self.past_losses.append(loss)

		for param, weights in self.models.best_params.iteritems():
			dweights = grads[param]
			config = self.optim_configs[param]	#TO DO: Create copies for optim config for each param
			weights_updated, next_config = adam_update(weights, dweights, config)
			self.model.params[param] = weights_updated
			self.optim_configs[param] = next_config 

	def check_accuracy():
		
		"""
		Inputs:

		Outputs:

		"""





	def train():
		pass


