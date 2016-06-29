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
	out   = X.reshape(X.shape[0],-1).dot(W) + b

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

def temporal_forward_pass(X, W, b):
	"""
	Inputs:
			X - Input data; dimensions (N, T, D)
			W - Weight matrix; dimensions (D, M)
			b - Biases; dimension (M,)

	Outputs:
			out - Output; dimensions (N, T, M)
			cache - Stored values required for backward pass
	"""

	N, T, D = X.shape
	M = b.shape[0]
	out = X.reshape(N * T, D).dot(W)
	out = out.reshape(N, T, M) + b
	cache = X, W, b, out

	return out, cache

def temporal_backward_pass(dout, cache, mask, qhy):
	"""
	Inputs:
			dout -Upstream gradient: dimensions (N, T, M)
			cache - Stored values from forward pass
			mask - Dropout mask associated with the hidden-to-output weight;
				   dimensions (D, M)
			qhy - Dropout probability for hidden-to-output weight;

	Outputs:
			dX - gradient with respect to input X; dimensions (N, T, D)
			dW - gradient with respect to weight matrix W; dimensions (D, M)
			b - gradient with respect to bias b; dimensions (M,)
	"""

	X, W, b, out = cache
	N, T, D = X.shape
	M = b.shape[0]
	dX = dout.reshape(N * T, M).dot(W.T).reshape(N, T, D)
	dW = dout.reshape(N * T, M).T.dot(X.reshape(N * T, D)).T

	if qhy != 0:
		dW[mask == 0] = 0

	db = np.sum(dout, axis =(0,1))

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
	h_next = None
	c_next = None
	cache  = {}
	N, D   = X.shape
	_, H   = h_prev.shape

	#Tabulate the activation
	a = np.dot(X, Wxh) + np.dot(h_prev, Whh) + b

	# Tabulate Input, Forget, Output, and Gated (ifog) results from activation a
	i = sigmoid(a[:,0:H])
  	f = sigmoid(a[:, H:2*H])
  	o = sigmoid(a[:,2*H:3*H])
  	g = np.tanh(a[:, 3*H:4*H])

  	c_next = (f * c_prev) + (i * g)
  	h_next = o * np.tanh(c_next)

  	#Store our relevant values for the backward pass
  	cache['i'], cache['f'], cache['o'], cache['g'] = i, f, o, g
  	cache['h_prev'], cache['c_prev'], cache['h_next'], cache['c_next'] = h_prev, c_prev, h_next, c_next
  	cache['X'], cache['Wxh'], cache['Whh'], cache['b'] = X, Wxh, Whh, b
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
	h, cache = None, {}
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
	cache['X'], cache['h0'], cache['Wxh'], cache['Whh'], cache['b'] = X, h0, Wxh, Whh, b
	cache['hs'], cache['cs'], cache['forward_caches'] = hs, cs, forward_caches

	return h, cache


def lstm_seq_backward(dh, cache, mask, qxh):
	"""
	Inputs:
		dh - Upstream gradients of hidden states; dimensions (N, T, H)
		cache - Values saved during forward pass
		mask - Dropout mash for input to hidden weight matrix; dimensions (D, 4H)
		qxh - Dropout probability for input to hidden weight matrix
	Output:
		dX - Gradient of input data; dimensions (N, T, D)
		dh0 - Gradient of first hidden state; dimensions (N, H)
		dWxh - Gradient of input-hidden weight matrix; dimensions (D, 4H)
		dWhh - Gradient of hidden-hidden weight matrix; dimensions (H, 4H)
		db - Gradient of biases; dimensions (4H,)
	"""

	#Unpack values from our cache
	X, h0, Wxh, Whh, b = cache['X'], cache['h0'], cache['Wxh'], cache['Whh'], cache['b']
	hs, cs, forward_caches = cache['hs'], cache['cs'], cache['forward_caches']

	#Pre-processing
	dX   = np.zeros_like(X)
	dh0  = np.zeros_like(h0)
	dc   = np.zeros_like(h0)
	dWxh = np.zeros_like(Wxh)
	dWhh = np.zeros_like(Whh)
	db   = np.zeros_like(b)
	N, T, D = X.shape

	#Main Loop
	for i in reversed(xrange(T)):
		forward_caches[i]
		dX[:,i,:], dh0, dc, dWxh_b, dWhh_b, db_b = lstm_step_backward(dh[:,i,:] + dh0, dc, forward_caches[i])
		
		if qxh != 0:
			dWxh_b[mask == 0 ] = 0

		dWxh += dWxh_b
		dWhh += dWhh_b
		db   += db_b

	return dX, dh0, dWxh, dWhh, db


def sigmoid(X):

	"""
	Numerically Stable Sigmoid Function:

	Inputs:
		X - Input matrix to be put through sigmoid function;

	Outputs:
		out - Input matrix after evaluation through simgoid function;
	"""

  	p_mask = (X >= 0)
  	n_mask = (X < 0)
  	z = np.zeros_like(X)
  	z[p_mask] = np.exp(-X[p_mask])
  	z[n_mask] = np.exp(X[n_mask])
  	top = np.ones_like(X)
  	top[n_mask] = z[n_mask]

  	return top / (1 + z)


def softmax(X, temp):
	"""
	Numerically Stable Softmax:

	Inputs:
		X - Input vector after an affine forward pass for scores; dimensions (N, D)
		temp - Scaling temperature applied to calculated probabilities

	Outputs:
		probs - Output probabilities for each letter; dimensions (N, D)
	"""

	probs = np.exp((X - np.max(X, axis=1, keepdims=True))/ temp)
  	probs /= np.sum(probs, axis=1, keepdims=True)
  	return probs



def lstm_softmax_loss(X, y, null_mask, temp=1.0):
	"""
	Inputs:
		X - Input scores through forward passes; dimensions (N, T, V)
		y - Ground truth labels for predicted next letter Each element is 
		    between 0 (inclusive) to V (exlcusive); dimensions (N, T)
		null_mask - boolean mask representing which elements should contribute to the loss and backprop;
					dimensions (N, T)
		temp - Scaling temperature applied to calculated probabilities


	Outputs:
		loss - Tabulated loss of the forward pass; Scalar
		ds - Gradient of loss with respect to input scores 

	Outputs:

	"""
	#Pre-processing shapes
	N, T, V = X.shape
	flat_X = X.reshape(N * T, V)
	flat_y = [int(idx) for idx in y.reshape(N * T)]
	mask_flat = null_mask.reshape(N * T)
	
	#Calculating probability scores and computing loss
	probs = softmax(flat_X, temp)

	log_probs = mask_flat * np.log(probs[np.arange(N*T), flat_y])
	loss = -np.sum(log_probs) / N

	#Determing grad Loss wrt input scores
	flat_ds = probs.copy()
	flat_ds[np.arange(N*T), flat_y] -= 1
	flat_ds /= N
	flat_ds *= mask_flat[:, None]
	ds = flat_ds.reshape(N, T, V)
	return loss, ds

