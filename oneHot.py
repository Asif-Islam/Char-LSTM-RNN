#############
# ONEHOT.PY #
#############


import numpy as np 


def convert_chars_to_vec(char_list, chars):
	"""
	Input:
		char_list - List of all the characters in the data
		chars - List of characters we wish to convert to vectors

	Output:
		char_vec - matrix representation of characters; dimensions (T, D), each row represents a step
	"""

	indices = np.asarray(convert_chars_to_idx(char_list, chars))
	char_vec = np.zeros((len(chars),len(char_list)))
	indices = np.ravel(indices)
	char_vec[np.arange(len(chars)), indices] = 1
	return char_vec
	pass

def convert_chars_to_idx(char_list, chars):
	"""
	Input:
		char_list - List of all the characters in the data
		chars - List of the characters we wish to determine the index of
	
	Output:
		- Return a list the same size of the chars, where each element represents the index of the char
	"""

	indices = lambda list, elem: [[i for i,x in enumerate(list) if x == e] for e in elem]
	return indices(char_list, chars) 


def convert_vec_to_char():
	pass

def convert_idx_to_char():
	pass

