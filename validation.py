#################
# VALIDATION.PY #
#################
import numpy as np
from solver import *
from oneHot import *
from LSTM import *
from layers import *
import time

def tune_parameters(data, char_dim, hidden_dim, seq_length, batch_size, dtype, char_list):
	batch_iter = int(data / batch_size)
	val_iter = int(batch_iter /  5)
	train_iter = batch_iter - val_iter
	batch_train_data = data[:train_iter*batch_size,:]
	batch_val_data = data[train_iter*batch_size:,:]
	best_config = {}
	best_val_acc = 0

	learning_rates = [1e-3, 5e-2, 1e-2, 5e-1, 1e-1]
	Wxh_mask_probs = [0.5, 0.35, 0.2, 0.1]
	Why_mask_probs = [0.5, 0.35, 0.2, 0.1]

	for lr in learning_rates:
		for qxh in Wxh_mask_probs:
			for qhy in Why_mask_probs:
				print 'learning rate %f, qxh %f and qhy %f' % (lr, qxh, qhy)
				model = LSTM_Network(char_dim, hidden_dim, seq_length, batch_size, dtype)
				model.qxh = qxh
				model.qhy = qhy
				solver = Solver(model, batch_train_data, optim_config)
				solver.train(char_list, mode)

				#Prepare Validation
				exm_len = data.shape[1]
				val_char_vecs = convert_chars_to_vec(char_list, batch_val_data)
				h = np.zeros((batch_size, hidden_dim))
				c = np.zeros_like(h)
				Wxh = solver.model.params['Wxh']
				Whh = solver.model.params['Whh']
				Why = solver.model.params['Why']
				b1 = solver.model.params['b1']
				b2 = solver.model.params['b2']

				#predict
				h, c, _ = lstm_seq_forward(val_char_vecs[:,:-1,:], h, Wxh, Whh, b1)
				scores, _ = temporal_forward_pass(h, Why, b2)
				N, T, D = scores.shape
				flat_scores = scores.reshape(N * T, D)
				probs = softmax(flat_scores, 1.0)
				probs = probs.reshape(N, T, D)
				predictions = np.zeros((N,T))
				for n in xrange(N):
					for t in xrange(T):
						predictions[n,t] = np.random.choice(range(len(char_list)),p=probs[n,t,:].ravel())

				true_chars = convert_chars_to_idx(batch_val_data)
				val_acc = np.mean(predictions == true_chars)
				if (val_acc > best_val_acc):
					best_val_acc = val_acc
					best_config['learning_rate'] = lr
					best_config['qxh'] = qxh
					best_config['qhy'] = qhy
					print 'Validation accuracy was: ' + str(val_acc)
					print 'The best validation accuracy is: ' + str(best_val_acc)
					print 'The best config is lr %f, qxh %f and qhy %f' % (lr, qxh, qhy)

				print 'The FINAL best config is lr %f, qxh %f and qhy %f' % (lr, qxh, qhy)
				return best_config	
