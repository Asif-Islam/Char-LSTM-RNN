#############
# SOLVER.PY #
#############

import numpy as np
from layers import *
from optim import *
import time

class Solver(object):

	"""
	The solver class is responsible for all duties related to training 
	our LTSM neural net. 
	"""

	def __init__(self, model, train_data, optim_config=None):

		self.model = model
		self.data_train = train_data['train']	#This will be a list of lists, per song basis
		self.optim_config = optim_config
		self.epochs = model.seq_length				
		self.print_interval = model.seq_length

		self.reset()

	def reset(self):
		"""
		The reset function refreshes storage parameters of our solver to an initial state
		"""

		self.epoch = 0
		self.past_losses = []

		self.optim_configs = {}
		for p in self.model.params:
      			d = {k: v for k, v in self.optim_config.iteritems()}
      			self.optim_configs[p] = d 

	
	def step(self, chars, char_list, mode, temp=1.0):
		"""
		Inputs:
			chars - The list batch of characters that we are forward and backward propagating over; dimensions
			char_list - list of all possible characters we can print
			temp - The softmax temperature scale, default to 1.0

		"""
		qxh, qhy = self.model.qxh, self.model.qhy
		masks = {}
		masks['Wxh'] = (np.random.rand(*self.model.params['Wxh'].shape) < qxh)
		masks['Why'] = (np.random.rand(*self.model.params['Why'].shape) < qhy)
		loss, grads, hprev = self.model.loss(chars, char_list, self.model.current_hidden_state, masks, mode, temp)
		self.model.current_hidden_state = hprev
		self.past_losses.append(loss)

		for param, weights in self.model.params.iteritems():
			dweights = grads[param]
			config = self.optim_configs[param]	#TO DO: Create copies for optim config for each param
			weights_updated, next_config = adam_update(weights, dweights, config)
			self.model.params[param] = weights_updated
			self.optim_configs[param] = next_config 


	def sample_training(self,char_list, n):

		h= np.zeros((1,self.model.hidden_dim))
		c = np.zeros_like(h)
		input_vec = np.zeros((1, len(char_list)))
		input_vec[0,np.random.choice(range(len(char_list)))] = 1
		sample_indices = self.model.sample(h, c, input_vec, 500, self.model.params)

		chars = [char_list[i] for i in sample_indices]
		output =  ''.join(chars)
		#TO DO: Run this section iff mode='train'
		text_file = open('Output-iter-' + str(n) + '.txt','w')
		text_file.write(output)
		text_file.close()


	def train(self, char_list, mode):

		data = self.data_train

		seq_length = self.model.seq_length
		num_train = data.shape[0]
		train_length = data.shape[1]

		print 'Starting Train! the data length is ' + str(train_length)
		epoch = 0
		for i in range(0, num_train, self.model.batch_size):

			n,p, final_loop = 0, 0 , False
			self.model.current_hidden_state = np.zeros_like(self.model.current_hidden_state)

			while True:
				print 'Batch set: ' + str(epoch)
				print 'Iteration Number: ' + str(n)

				if p + seq_length + 1 >= train_length:
					final_loop = True

				if (final_loop == False):
					input_train = data[i:i+self.model.batch_size, p:p+seq_length+1]
				else:
					input_train = data[epoch*self.model.batch_size:i, p:]

				self.step(input_train, char_list, mode)
				p += seq_length
				n += 1

				if (self.model.batch_size == 1):
					if (n % 1000 == 0):
						self.sample_training(char_list, n)

				if (final_loop == True):
					break

			epoch +=1
			self.sample_training(char_list, n)
			for k, v in self.model.params.iteritems():
				np.savetxt(k + '.csv', v, delimiter=',')


