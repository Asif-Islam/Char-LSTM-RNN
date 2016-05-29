#########################
# FUNCTIONTESTCHECKS.PY #
#########################

import numpy as np 
from layers import *
#TO DO: Add a call to every testable function; create test cases and manually calculate the expected value
#Then check if our function does as we expect

def relative_error(x, y):
	return np.max(np.abs(x-y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


#############
# LAYERS.PY #
#############

def forward_pass_test():
	N, D, H = 4, 2, 2
	X = np.linspace(-3,7, num=N*D).reshape((N,D))
	W = np.linspace(-6,3, num=D*H).reshape((D,H))
	b = np.linspace(-2,-1, num=H).reshape((H,))
	expected_out = np.array([
							[16, 3.2857],
							[-1.1429, 3.2857],
							[-18.2857, 3.2858],
							[-35.4286,3.2857]])

	out, cache = forward_pass(X,W,b)

	print '=============================='
	print 'FORWARD PASS ERROR:'
	print relative_error(out,expected_out)
	print '=============================='

def backward_pass_test():
	N, D, H = 4, 2, 2
	X = np.linspace(-3,7, num=N*D).reshape((N,D))
	W = np.linspace(-6,3, num=D*H).reshape((D,H))
	b = np.linspace(-2,-1, num=H).reshape((H,))
	dout = np.linspace(-4,4,num=N*H).reshape((N,H))


	expected_dX = np.array([
							[32.5714, -8.5714],
							[12.0000, -1.7143],
							[-8.5714, 5.1429],
							[-29.1429, 12.0000]])

	expected_dW = np.array([
							[29.714, 35.592],
							[26.449, 38.857]
							])

	expected_db = np.array([-2.2857, 2.2857])


	cache = {}
	cache['X'], cache['W'], cache['b'] = X, W, b
	dX, dW, db = backward_pass(dout, cache)

	print '=============================='
	print 'BACKWARD PASS ERROR:'
	print relative_error(dX,expected_dX)
	print relative_error(dW, expected_dW)
	print relative_error(db, expected_db)
	print '==============================' 


def lstm_step_forward_test():
	pass

def lstm_step_backward_test():
	pass

def lstm_seq_forward_test():
	pass

def sigmoid_test():
	pass

def lstm_softmax_loss_test():
	pass


#############
# ONEHOT.PY #
#############

def convert_char_to_vec_test():
	pass

def convert_char_to_idx_test():
	pass

def convert_vec_to_char_test():
	pass

def convert_idx_to_char_test():
	pass


############
# OPTIM.PY #
############

def sgd_test():
	pass

def adam_test():
	pass


#############
# SOLVER.PY #
#############

def _step_test():
	pass

def sample_test():
	pass

def train_test():
	pass


print 'START'
forward_pass_test()
backward_pass_test()