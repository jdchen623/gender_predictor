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
    featureVectors = {}
    for step in range(0, numIters):
        """
        def predictor(x):
            return 1 if dotProduct(featureExtractor(x), weights) > 0 else -1
        misclassifiedTraining = evaluatePredictor(trainExamples, predictor)
        misclassifiedTest = evaluatePredictor(testExamples, predictor)
        print("Training error: %f, Test error: %f" % (misclassifiedTraining, misclassifiedTest))
        """
        for (x, y) in trainExamples.iteritems():
            if (x in featureVectors):
                phiX = featureVectors[x]
            else:
                phiX = featureExtractor(x)
                featureVectors[x] = phiX
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
    You can run stuff with test.py or classify.py

    TODO: Dealing with - and --, more features, consider using the form (count / number of words) for all rate features,
    maybe use defaultdict?
    Current features:
        -occurences of specific word / words
        -! per sentence
        -? per sentence
        -, per sentence
        -' per sentence
        -average length of sentences (proxy for # sentences is number of periods)
    Feature to add:
        -Average word length
        -Percent dialogue
        -Sentiment

    """

    wordDict = {} #For word features or any per-word features
    sentenceDict = {} #For per-sentence features
    miscellaneousDict = {} #Any other non-rate based features we want
    numWords = 0
    numSentences = 0
    with open(file,'r') as f:
    	content = f.read()
        words = [word.lower() for word in content.split()]
        numWords = len(words)
        avgWordLength = 0
    	for word in words:
            strippedWord = word.strip('.,_!?()";:|*')
            wordDict[strippedWord] = wordDict[strippedWord] + 1 if strippedWord in wordDict else 1
            punctuation = word[-1]
            if (punctuation == '!' or punctuation == '?' or punctuation == ','):
                sentenceDict[punctuation] = sentenceDict[punctuation] + 1 if punctuation in sentenceDict else 1 #default dict?
            if ("'" in word):
                sentenceDict["'"] = sentenceDict["'"] + 1 if "'" in sentenceDict else 1
            avgWordLength += len(strippedWord)
        avgWordLength /= len(words)
        numSentences = len(content.split('.'))
        avgSentenceLength = float(numWords) / numSentences
        miscellaneousDict['sentenceLength'] = avgSentenceLength
        miscellaneousDict['wordLength'] = avgWordLength


    #turn values in vectors to rates
    wordDict = {key: float(wordDict[key]) / numWords for key in wordDict.keys() if wordDict[key] > 2 and wordDict[key] < 50}
    sentenceDict = {key: float(sentenceDict[key]) / numSentences for key in sentenceDict.keys()}

    #Merging dicts and adding bias
    featureVector = wordDict
    featureVector.update(sentenceDict)
    featureVector.update(miscellaneousDict)
    featureVector['_bias_'] = 1

    return featureVector
