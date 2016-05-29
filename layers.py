#############
# LAYERS.PY #
#############

import numpy as np


"""
layers.py contains a collection of all the necessary forward and backward passes
of our LSTM necessary. 
"""

def forward_pass(X, W, b):
	"""
	Inputs:
		X - input matrix; dimension: (N, D)
		W - weight matrix; dimension: (D, H)
		b - bias vectorl dimension: (H,)

	Outputs:
		out - Output of the linear forward pass; dimension: (N, H)
		cache - dictionary of values: X, W and b required for the backward pass
	"""

	cache = {}

	out = X.dot(W) + b

	cache['X'] = X
	cache['W'] = W
	cache['b'] = b

	return out, cache


def backward_pass(dout, cache):
	"""
	Inputs:
		dout - Upstream gradient of output; dimension: (N, H)
		cache - dictionary of values: X, W and b

	Outputs:
		dX - gradient of loss with respect to X; dimension: (N, D)
		dW - gradient of loss with respect to W; dimension: (D, H)
		db - gradient of loss with respect to b; dimension: (H,)
	"""

	X = cache['X']
	W = cache['W']
	b = cache['b']
	
	dX = np.zeros_like(X)
	dW = np.zeros_like(W)
	db = np.zeros_like(b)

	dX = dout.dot(W.T)
	dW = X.T.dot(dout)
	db = np.sum(dout, axis=0)

	return dX, dW, db


def lstm_step_forward(X, h_prev, c_prev, Wxh, Whh, b):
	"""
	Inputs:
		X - Input data; dimension: (N, D)
		h_prev - Hidden state of previous timestep; dimension: (N, H)
		c_prev - Cell state of previous timestep; dimension: (N, H)
		Wxh - Weight matrix mapping input layer to hidden layer; dimensions (D, 4H)
		Whh - Weight matrix mapping hidden-to-hidden between timsteps; dimensions (H, 4H)
		b - Biases; dimensions (4H,)
	Outputs:
		h_next - Next hidden state; dimension (N, H)
		c_next - Next cell state; dimension (N, H)
		cache - Dictionary of values required for backward pass
	"""
	#Pre-processing
	h_next, c_next = None
	cache = {}
	N, D = X.shape
	_, H = h_prev.shape

	#Tabulate the activation
	a = np.dot(X, Wxh) + np.dot(h_prev, Whh) + b

	# Tabulate Input, Forget, Output, and Gated (ifog) results from activation a
	i = sigmoid(a[:,0:H])
  	f = sigmoid(a[:, H:2*H])
  	o = sigmoid(a[:,2*H:3*H])
  	g = np.tanh(a[:, 3*H:4*H])

  	#Tabulate the next hidden and cell state
  	c_next = (f * c_prev) + (i * g)
  	h_next = o * np.tanh(c_next)

  	#Store our relevant values for the backward pass
  	cache['i'], cache['f'], cache['o'], cache['g'] = i, f, o, g
  	cache['X'], cache['Wxh'], cache['Whh'], cache['b'] = X, Wxh, Whh, b
  	cache['h_prev'], cache['c_prev'], cache['h_next'], cache['c_next'] = h_prev, c_prev, h_next, c_next

  	return h_next, c_next, cache


def lstm_step_backward(dh_next, dc_next, cache):
	"""
	Inputs:
		dh_next - Upstream gradients of next hidden state; dimension (N, H)
		dc_next - Upstream gradients of next cell state; dimension (N , H)
		cache - Values stored from forward pass

	Outputs:
		dX - Gradient of the input data; dimension (N, D)
		dh_prev - Gradient of previous hidden state; dimension (N, H)
		dc_prev - Gradient of previous cell state; dimension (N, H)
		dWxh - Gradient of input-2-hidden weights; dimension (D, 4H)
		dWhh - Gradient of hidden-2-hidden weights; dimension (H, 4H)
		db - Gradient of biases; dimension (4H,)
	"""

	#Unpack values from cache
	i, f, o, g =  cache['i'], cache['f'], cache['o'], cache['g']
  	X, Wxh, Whh, b = cache['X'], cache['Wxh'], cache['Whh'], cache['b']
  	h_prev, c_prev, h_next, c_next = cache['h_prev'], cache['c_prev'], cache['h_next'], cache['c_next']

  	#Pre-processing
  	N, H = dc_next.shape
  	dX, dh_prev, dc_prev, dWxh, dWhh, db = None, None, None, None, None, None
  	dX = np.zeros_like(X)
  	dh_prev = np.zeros_like(h_prev)
  	dc_prev = np.zeros_like(c_prev)
  	dWxh = np.zeros_like(Wxh)
  	dWhh = np.zeros_like(Whh)
  	db = np.zeros_like(b)

  	#Tabulate gradients via Computational Graph
  	do = np.tanh(c_next) * dh_next
  	dc_next = dc_next + (dh_next * o) * (1 - np.tanh(c_next) * np.tanh(c_next))
  	dc_prev = dc_next * f
  	df = dc_next * c_prev
  	di = dc_next * g
  	dg = dc_next * i

  	#Tabulate ifog gradients through their non-linearities (sigmoid/tanh)
  	ddi = di * i * (1 - i)
  	ddf = df * f * (1 - f)
  	ddo = do * o * (1 - o)
  	ddg = dg * (1 - g*g)					#Tanh, not sigmoid here
  	dact = np.hstack((ddi, ddf, ddo, ddg))

	#Compute remaining gradients at the front of the computational graph 
  	dWx = np.dot(X.T, dact)
  	dX = np.dot(dact, Wxh.T)
  	dWh = np.dot(h_prev.T, dact)
  	dh_prev = np.dot(dact, Whh.T)
  	db = np.sum(dact, axis=0)

 	return dX, dh_prev, dc_prev, dWx, dWh, db


def lstm_seq_forward(X, h0, Wxh, Whh, b):
	"""
	Inputs:
		X - Input data through time sequence; dimension (N, T, D)
		h0 - Initial hidden state; dimension (N, H)
		Wxh - Weight matrix for input-to-hidden mapping; dimension (D, 4H)
		Whh - Weight matrix for hidden-to-hidden mapping; dimension (H, 4H)
		b - Biases of shape (4H,)

	Outputs:
		h - Hidden states for all timesteps in the sequence; dimension (N, T, H)
		cache - Values needed for the backward pass
	"""

	#Pre-processing
	h, cache = None, None
	N, T, D = X.shape
	_, H = h0.shape
	h = np.zeros((N, T, H))
	hs = {}						#Dictionary of Hidden States
	cs = {}						#Dictionary of Cell States
	forward_caches = {}
	cs[-1] = np.zeros((N,H))	
	hs[-1] = h0

	#Iterate over all timestep forward steps
	for i in xrange(T):
		hs[i], cs[i], forward_caches[i] = lstm_step_forward(X[:,i,:], hs[i-1], cs[i-1], Wxh, Whh, b)
		h[:,i,:] = hs[i]

	#Store our values into the cache
	cache['X'], cache['h0'], cache['Wxh'], cache['Whh'] = X, h0, Wxh, Whh
	cache['hs'], cache['cs'], cache['forward_caches'] = hs, cs, forward_caches

	return h, cache


def lstm_seq_backward():
	pass


def sigmoid(X):

	"""
	Numerically Stable Sigmoid Function:

	Inputs:
		X - input matrix to be put through sigmoid function;

	Outputs:
		out - input matrix after evaluation through simgoid function;
	"""

  	p_mask = (X >= 0)
  	n_mask = (X < 0)
  	z = np.zeros_like(X)
  	z[p_mask] = np.exp(-X[p_mask])
  	z[n_mask] = np.exp(X[n_mask])
  	top = np.ones_like(X)
  	top[n_mask] = z[n_mask]

  	return top / (1 + z)


def lstm_softmax_loss():
	pass

