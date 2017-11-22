#!/usr/bin/env python
from featureExtractor import *
from labels import *

trainingLabels = allLabels

weights = learnPredictor(trainingLabels, None, extractWordFeatures, 200, 0.01)

correctClassifications = 0
totalClassifications = len(trainingLabels)
#We'll change this later...decompose and stuff
for (file, y) in trainingLabels.iteritems():
	featureVector = extractWordFeatures(file)
	score = dotProduct(featureVector, weights)
	classification = -1 if score < 0 else 1
	if (classification == y):
		correctClassifications += 1
		print("Correctly classified %s as %d with score %f." % (file, classification, score))
	else:
		print("Incorrectly classified %s as %d with score %f." % (file, classification, score))
percentCorrect = 100 * float(correctClassifications) / totalClassifications
print("%d correct classifications out of %d (%f percent correct)." % (correctClassifications, totalClassifications, percentCorrect))


