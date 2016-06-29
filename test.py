import numpy as np
import music21
"""
score = music21.converter.parse('SpiderDANCE.mid')
key = score.analyze('key')
print(key.tonic.name, key.mode)
"""
"""
(1) Predefined char array size (N, W), where W is the length of the song after null appends
(2) In song processing, we store each song into a list
(3) In array processing, we loop over the list N time, and set array(n,:) equal to list(song) as
    an np array!
(4) This results in a sliceable array of characters!
"""


a = np.array([[0,1,0,0],
			[0,1,0,0],
			[1,0,0,0],
			[0,0,1,0],
			[0,0,1,0],
			[0,0,1,0]])

x = np.where(a == 1)[1]
y = np.where(x == 3)[0]
print y
d = np.ones((a.shape[0],1))
d[y,:] = 0
print d

b = np.array([[1,2,3,4,5,6],
			[3,4,5,6,7,8],
			[3,4,5,6,7,43],
			[3,4,5,6,7,9],
			[3,4,5,6,7,10],
			[3,4,5,6,7,11]])

print b * d


"""
s = np.random.rand(5,)
print s
print s[2:]

exm = [];
str1 = 'abc and'
str2 = 'efg ond'
str3 = 'hij end'
exm.append(str1)
exm.append(str2)
exm.append(str3)

b = np.chararray((3,7))

for i in xrange(len(exm)):
	b[i,:] = np.asarray(list(exm[i]))

print b == ' '
print b
"""
"""charar = np.chararray((3,3))
charar[0] = str1
charar[1] = str2
charar[2] = str3 
print charar"""


"""list1 = []
list2 = []
list3 = []
list4 = []
list5 = []

for i in xrange(5):
	list1.append(i)

for j in xrange(5,10):
	list2.append(j)

for k in xrange(10,15):
	list4.append(k)

for l in xrange(15,20):
	list5.append(l)

list3.append(list1)
list3.append(list2)
list3.append(list4)
list3.append(list5)


print list1, list2, list3
print list3[0:2][1][0]
"""

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