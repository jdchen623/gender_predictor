/..............................................................................................................................................................................................................................#!/usr/bin/env python
import os

"""
Put new text files in gender-predictor folder, run this script, then template will be in labelsTemp.py. Label
new ones, then put new text files in /data, and copy and paste labelsTemp.py into labels.py
"""
with open('labels.py', 'r') as o:
	dictionary = list(o.read())
	dictionaryCopy = list(dictionary)
	startIndex = dictionary.index('}')
	dictionary[startIndex] = ', '
	dictionary = ''.join(dictionary)
	labelCount = 0
	with open('labelsTemp.py','w') as f:
		dictionary += '\n#START LABELING HERE#\n'
		for author in os.listdir('.'):
			if (not author.endswith('.txt') or author == 'test.txt'):
				continue
			if (labelCount != 0):
				dictionary += ' '
			dictionary += "'data/" + author + "': 1"
			dictionary += ','
			if (labelCount % 3 == 0):
				dictionary += '\\\n'
			labelCount += 1
		if (labelCount == 0):
			dictionary = ''.join(dictionaryCopy)
		else:
			dictionary = list(dictionary)
			while(dictionary[-1] == ',' or dictionary[-1] == '\\' or dictionary[-1] == '\n'):
				del dictionary[-1]
			dictionary = ''.join(dictionary)
			dictionary += '}'
		f.write(dictionary)
