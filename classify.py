#!/usr/bin/env python
from featureExtractor import *
from labels import *
from random import shuffle

verboseMode = False if len(sys.argv) == 1 else bool(sys.argv[1])
PERCENT_TRAINING = 0.8
NUM_ITERS = 20
ETA = 0.01

# Create training and test sets
labelList = list(allLabels.keys())
shuffle(labelList) # shuffle data to get fresh training, data set each time
separationIdx = int(len(labelList)*PERCENT_TRAINING)
trainingExamples = {labelList[i]: allLabels[labelList[i]] for i in range(0, separationIdx)}
testExamples = {labelList[j]: allLabels[labelList[j]] for j in range(separationIdx, len(labelList))}

# Train classifier on training set
weights = learnPredictor(trainingExamples, None, extractWordFeatures, NUM_ITERS, ETA, verboseMode)

# Calculate classifier accuracy on traning and test sets
def calculateAccuracy(classifierWeights, examples, isVerbose):
	correctClassifications = 0
	totalClassifications = len(examples)
	for (file, y) in examples.iteritems():
		featureVector = extractWordFeatures(file)
		score = dotProduct(featureVector, classifierWeights)
		classification = -1 if score < 0 else 1
		if (classification == y):
			correctClassifications += 1
			if isVerbose:
				print("Correctly classified %s as %d with score %f." % (file, classification, score))
		elif isVerbose:
			print("Incorrectly classified %s as %d with score %f." % (file, classification, score))
	return correctClassifications

trainingNumCorrect = calculateAccuracy(weights, trainingExamples, verboseMode)
testingNumCorrect = calculateAccuracy(weights, testExamples, verboseMode)
print("Training set accuracy: %d / %d : %f percent correct" % (trainingNumCorrect, len(trainingExamples), 100 * float(trainingNumCorrect)/len(trainingExamples)))
print("Testing set accuracy: %d / %d : %f percent correct" % (testingNumCorrect, len(testExamples), 100 * float(testingNumCorrect)/len(testExamples)))
