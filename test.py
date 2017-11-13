#!/usr/bin/env python
from featureExtractor import *

file = 'data/mary-shelley.txt'
featureVector = extractWordFeatures(file)
print("Book: %s" % file)
print("Exclamation points per sentence: %s" % featureVector['!'])
print("Commas per sentence: %s" % featureVector[','])
print("Instances of 'the' per word: %s" % featureVector['the'])
print("Mean sentence length: %s" % featureVector['sentenceLength'])