#!/usr/bin/env python
from featureExtractor import *

file = 'test.txt'
featureVector = extractWordFeatures(file)
print("File: %s" % file)
print("Exclamation points per sentence: %s" % featureVector['!'])
print("Commas per sentence: %s" % featureVector[','])
print("Instances of 'the' per word: %s" % featureVector['the'])
print("Mean sentence length: %s" % featureVector['sentenceLength'])