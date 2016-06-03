############
# OPTIM.PY #
############

import numpy as np 

def sgd(W, dW, config=None):
	"""
	Inputs:
		W - The parameter that we are updating
		dW - the gradient with respect to the parameter W
		config - configuration values:
			Learning Rate -Scalar value that determines the scale of our descent

	Outputs:
		W - Updated parameter
		config - configuration values
	"""

	if config is None:
		config = {}

	config.setdefault('learning_rate', 1e-1)

	W -= config['learning_rate'] * dW

	return W, config

#Code optimized and cleaned with the help of CS231n Notes
def adam_update(W, dW, config=None):

	"""
	Inputs:
		W - The parameter that we are updating
		dW - the gradient with respect to the parameter W
		config - configuration values:
			Learning Rate - Scalar value that determines the scale of our descent
			beta1: Decay rate for first order mapping of gradient
			beta2: Decay rate for second order mapping of gradient
			m: Moving average of gradient
			v: Moving average of gradient squared
			t: Count on # of iterations

	Outputs:
		next_W - Updated parameter
		config - configuration values
	"""


	if config is None: config = {}
  		config.setdefault('learning_rate', 1e-1)
		config.setdefault('beta1', 0.9)
		config.setdefault('beta2', 0.999)
		config.setdefault('epsilon', 1e-7)
		config.setdefault('m', np.zeros_like(W))
		config.setdefault('v', np.zeros_like(W))
		config.setdefault('t', 0)
  
  next_W = None

  beta1, beta2, eps = config['beta1'], config['beta2'], config['epsilon']
  t, m, v = config['t'], config['m'], config['v']

  m = beta1 * m + (1 - beta1) * dW
  v = beta2 * v + (1 - beta2) * (dW * dW)
  t += 1
  alpha = config['learning_rate'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
  W -= alpha * (m / (np.sqrt(v) + eps))
  config['t'] = t
  config['m'] = m
  config['v'] = v
  next_W = W

  return next_W, config