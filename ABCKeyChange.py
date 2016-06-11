import music21 as m21
import numpy as np
import sys

x = np.array([[1,2,3],[4,5,6]])
print 1 - x
def matchKey(ABCpass, next_note):
	try:
		if (_BmajorKey[next_note] != None):
			ABCpass += _BmajorKey[next_note]
			return ABCpass
	except KeyError:
		ABCpass += next_note
		return ABCpass

filename = "ValeHome"

#Load the desired key

if (len(sys.argv) != 3):
	raise ValueError("Insufficient number of arguments")
else:
	target_tonic = sys.argv[1]
	target_mode = sys.argv[2]


#Load the song and determine the apparent key
score = m21.converter.parse(filename + ".mid")
key = score.analyze("key")
print key.tonic.name, key.mode

#Load all the possible characters in order
#C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B, C
notes_v1 = ["=C,","^C,", "=D,", "_E,", "=E,", "=F,", "^F,", "=G,", "_A,", "=A,", "_B,", "=B",
		   "=C", "^C", "=D", "_E", "=E", "=F", "^F", "=G", "_A", "=A", "_B", "=B",
		   "=c", "^c", "=d", "_e", "=e", "=f", "^f", "=g", "_a", "=a", "_b", "=b",
		   "=c'", "^c'", "=d'", "_e'", "=e'", "=f'", "^f'", "=g'", "_a'", "=a'", "_b'", "=b'"]

notes_v2 = ["=C,","_D,", "=D,", "^D,", "=E,", "=F,", "_G,", "=G,", "^G,", "=A,", "^A,", "=B,"
		   "=C", "_D", "=D", "^D", "=E", "=F", "_G", "=G", "^G", "=A", "^A", "=B",
		   "=c", "_d", "=d", "^d", "=e", "=f", "_g", "=g", "^g", "=a", "^a", "=b",
		   "=c'", "_d'", "=d'", "^d'", "=e'", "=f'", "_g'", "=g'", "^g'", "=a'", "^a'", "=b'"]	


letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
letter_ornaments = ["'", ',']
print ''' ' '''
accidentals = ['_', '=', '^']
headers = ['X', 'M', 'Q']
key_signatures = {}
key_signatures['Cmajor'] = 0
key_signatures['Aminor'] = 0

key_signatures['C#major'] = 1
key_signatures['D-major'] = 1
key_signatures['A#minor'] = 1 #UGH THIS ISN'T RIGHT??
key_signatures['B-minor'] = 1

key_signatures['Dmajor'] = 2
key_signatures['Bminor'] = 2

key_signatures['D#major'] = 3
key_signatures['E-major'] = 3
key_signatures['Cminor']  = 3

key_signatures['Emajor']  = 4
key_signatures['C#minor'] = 4
key_signatures['D-minor'] = 4

key_signatures['Fmajor'] = 5
key_signatures['Dminor'] = 5

key_signatures['F#major'] = 6
key_signatures['G-major'] = 6
key_signatures['D#minor'] = 6
key_signatures['E-minor'] = 6

key_signatures['Gmajor'] = -5
key_signatures['Eminor'] = -5

key_signatures['G#major'] = -4
key_signatures['A-major'] = -4
key_signatures['Fminor']  = -4

key_signatures['Amajor']  = -3
key_signatures['F#minor'] = -3
key_signatures['G-minor'] = -3

key_signatures['A#major'] = -2
key_signatures['B-major'] = -2
key_signatures['Gminor']  = -2

key_signatures['Bmajor']  = -1
key_signatures['C-major'] = -1
key_signatures['G#minor'] = -1
key_signatures['A-minor'] = -1
#key_signatures

CmajorKey = {}
FmajorKey = {}
_BmajorKey = {}
GmajorKey = {}

FmajorKey['^a'] = '_b'
FmajorKey["^a'"] = "_b'"
FmajorKey['^A'] = '_B'
FmajorKey['^A,'] = '_B,'

_BmajorKey['^a'] = '_b'
_BmajorKey["^a'"] = "_b'"
_BmajorKey['^A'] = '_B'
FmajorKey['^A,'] = '_B,'
_BmajorKey['^d'] = '_e'
_BmajorKey["^d'"] = "_e'"
_BmajorKey['^D'] = '_E'
_BmajorKey['^D,'] = '_E,'


GmajorKey['_g'] = '^f'
GmajorKey["_g'"] = "^f'"
GmajorKey['_G'] = '^F'
GmajorKey['_G,'] = '^F,'	



#Open The file and store it in a string
abc_score = open(filename + ".abc", "r").read()


#First go through the entire string and append sharps/flats/naturals to 
#unsigned notes that follow a note with an accidental; Ignore Comments while we're ati t


output = open("pass0.txt", "w")
output.write(abc_score)

"""
########
PASS 1 #
########

Remove all comments, T and K fields
"""

i = 0; #String iterator
pass1 = ""
while True:
	if (i >= len(abc_score)):
		break;

	if (abc_score[i] == '%' or abc_score[i] == 'T' or abc_score[i] == 'K'):
		while True:
			i +=1
			if(abc_score[i] == '\n'):
				break;	
	else:
		pass1 = pass1 + abc_score[i]	
	
	i += 1

output = open("pass1.txt", "w")
output.write(pass1)

"""
########
PASS 2 #
########

Remove all extra whitespace
"""

i = 0
pass2 = ""
while True:
	if (i >= len(pass1)):
		break;

	if (pass1[i] == '\n'):
		i += 1
		if (i >= len(pass1)):
			break;
		else:
			if (pass1[i] == '\n'):
				pass2 = pass2 + '\n'
				i += 1
			elif (pass1[i] == 'X'):
				pass2 = pass2 + pass1[i]
				i +=1
			else:
				pass2 = pass2 + '\n' + pass1[i]
				i += 1
	else:
		pass2 = pass2 + pass1[i]
		i += 1
output = open("pass2.txt", "w")
output.write(pass2)

"""
########
PASS 3 #
########
Add Accidentals to every note
"""
i = 0
state = {}
state['dyn'] = False
pass3 = ""
while True:
	if (i >= len(pass2)):
		break

	if (pass2[i] in headers):
		while True:
			if (pass2[i] == '\n'):
				break
			pass3 = pass3 + pass2[i]
			i += 1

	if (pass2[i] == '|'):
		state = {}
		state['dyn'] = False

	elif (pass2[i] == '+'):
		if (state['dyn'] == False):
			state['dyn'] = True
			i += 1
			continue 
		else:
			state['dyn'] = False
			i += 1
			if (pass2[i] == ' '):
				i += 1
			continue 

	elif (state['dyn'] == True):
		i += 1
		continue

	elif (pass2[i] in letters):	#TO DO!!! MAKE THE STATE OF THE NOTE EQUAL TO = !!!
		if not state:
			pass3 = pass3 + '='
		else:
			if (pass2[i] in state):
				pass3 = pass3 + state[pass2[i]]
			else:
				pass3 = pass3 + '='

	elif (pass2[i] in accidentals):
		state[pass2[i+1]] = pass2[i]
		pass3 = pass3 + pass2[i] + pass2[i+1]
		i += 2



	pass3 = pass3 + pass2[i]
	i += 1

output = open("pass3.txt", "w")
output.write(pass3)

"""
########
PASS 4 #
########
TRANSPOSE STEP! First we determine the semitone distance and then do a step-down pass
"""
song_key = key.tonic.name + key.mode
print song_key
desired_key = sys.argv[1] + sys.argv[2]
print desired_key 
base_distance = key_signatures[song_key]
print base_distance

key_signatures[desired_key] += -1*base_distance
if (key_signatures[desired_key] > 6):
	key_signatures[desired_key] = key_signatures[desired_key] - 12
elif (key_signatures[desired_key] <= -6):
	key_signatures[desired_key] = key_signatures[desired_key] + 12

rel_distance = key_signatures[desired_key]
print rel_distance

#being forward pass

i = 0
index = 0
note = ""
pass4 = ""
no_orn = False
while True:
	if (i >= len(pass3)):
		break;

	if (pass3[i] in accidentals):
		note += pass3[i]
		i +=1
		if (i >= len(pass3)):
			break
		note += pass3[i]
		i += 1
		if (i >= len(pass3)):
			break
		if (pass3[i] in letter_ornaments):
			note += pass3[i]
			i += 1
			no_orn = False
		else:
			no_orn = True
		try:
			index = notes_v1.index(note)
			next_note = notes_v1[index + rel_distance]
			pass4 = matchKey(pass4, next_note)


		except ValueError:
			index = notes_v2.index(note)
			next_note = notes_v2[index + rel_distance]
			pass4 = matchKey(pass4, next_note)
			#Error Checking To make sure we don't get beyond the length of our array
			#Shift notes according to the key signature

		if (no_orn == True):
			pass4 += pass3[i]
			i += 1

		note = ""
		no_orn = False
	else:
		note = ""
		no_orn = False
		pass4 += pass3[i]
		i += 1

output = open("pass4.txt", "w")
output.write(pass4)

#Now recreate the score letter by transposed by X semitones
#Remove all comments in the process