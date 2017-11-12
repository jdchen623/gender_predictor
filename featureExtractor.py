#!/usr/bin/python

import random
import collections
import math
import sys
from util import *


############################################################
# stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for step in range(0, numIters):
        def predictor(x):
            return 1 if dotProduct(featureExtractor(x), weights) > 0 else -1
        misclassifiedTraining = evaluatePredictor(trainExamples, predictor)
        misclassifiedTest = evaluatePredictor(testExamples, predictor)
        print("Training error: %f, Test error: %f" % (misclassifiedTraining, misclassifiedTest))
        for (x, y) in trainExamples:
            phiX = featureExtractor(x)
            if dotProduct(weights, phiX)*y < 1:
                gradient = {key: (-phiX[key]*y) for key in phiX}
            else:
                gradient = {}

            increment(weights, -eta, gradient)

    # END_YOUR_CODE
    return weights

def gradientLossHinge(phiX, y, w):
    if -dotProduct(w, phiX)*y < 1:
        return {key: (-phiX[key]*y) for key in phiX}
    else:
        return {}


def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        ngrams = {}
        x = "".join(x.split())
        for i in range(0, len(x) - n + 1):
            ngrams[x[i:i+n]] = ngrams[x[i:i+n]] + 1 if x[i:i+n] in ngrams else 1
        return ngrams
        # END_YOUR_CODE
    return extract

def extractWordFeatures(file):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    dict = {}
    with open(file) as f:
		content = f.read()
    	for word in content.split():
        	dict[word] = dict[word] + 1 if word in dict else 1
    return dict
    # END_YOUR_CODE
