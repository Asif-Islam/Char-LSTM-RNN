import numpy as np
import music21

score = music21.converter.parse('SpiderDANCE.mid')
key = score.analyze('key')
print(key.tonic.name, key.mode)

"""
data = open('greatexpectations.txt', 'r').read()
chars = list(set(data))
print chars

input_d = data[0:5]
indices = lambda list, elem: [[i for i,x in enumerate(list) if x == e] for e in elem]
print indices(chars,input_d)
a = np.asarray(indices(chars,input_d))
print a.shape


b = [1,2,3,4,5]
print b[1:-1]"""


"""
Order of Tasks:
---------------

(1) Finish ALL of layers.py
	-> Softmax
	-> LSTM Softmax Loss
	-> Seq backward

(2) Update sample in solver.py

(3) Write tests for ALL of layers.py

(4) Finish ALL of oneHot.py

(5) Write test cases for ALL of oneHot.py

(6) Starting with train(), finish ALL of solver.py

(7) Create LSTM.py class

(8) Write validation block

(9) LSTM_main.py for final product




"""