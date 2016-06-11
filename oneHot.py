#############
# ONEHOT.PY #
#############


import numpy as np 


def convert_chars_to_vec(char_list, chars):
	"""
	Input:
		char_list - List of all the characters in the data
		chars - Numpy array of characters; dimensions (N, T)

	Output:
		char_vec - matrix representation of characters; dimensions (N, T, D), each row represents a step
	"""
	N, T, D = chars.shape[0], chars.shape[1], len(char_list)
	indices = convert_chars_to_idx(char_list, chars)
	char_vec = np.zeros((N*T,D))
	indices = [int(idx) for idx in indices.reshape(N * T)]
	char_vec[np.arange(N*T), indices] = 1
	char_vec = char_vec.reshape(N,T,D)
	return char_vec

def convert_chars_to_idx(char_list, chars):
	"""
	Input:
		char_list - List of all the characters in the data
		chars - Numpy array of characters; dimensions (N, T)
	
	Output:
		- Return an array the same size of the chars, where each element represents the index of the char
	"""
	N, T = chars.shape[0], chars.shape[1]
	idx = np.zeros((chars.shape[0],chars.shape[1]))
	indices = lambda list, elem: [[i for i,x in enumerate(list) if x == e] for e in elem]
	chars = chars.reshape(N*T)
	return np.asarray(indices(char_list,chars)).reshape(N,T)



def convert_vec_to_char():
	pass

def convert_idx_to_char():
	pass

