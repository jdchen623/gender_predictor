from featureExtractor import *

trainingLabels = {'data/anne-jane-couples.txt': 1, 'data/arthur-conan-doyle.txt': -1, 'data/charles-dickens.txt': -1, \
'data/charlotte-bronte.txt': 1, 'data/charlotte-eaton.txt': 1, 'data/emily-bronte.txt': 1, 'data/james-joyce.txt': -1, \
'data/jane-austen.txt': 1, 'data/joseph-conrad.txt': -1, 'data/karl-brown.txt': -1, 'data/lewis-carroll.txt': -1, \
'data/mark-twain.txt': -1, 'data/mary-shelley.txt': 1}

testLabels = {'data/nathaniel-hawthorne.txt': -1, 'data/oscar-wilde.txt': -1, 'data/victor-hugo.txt': -1,\
'data/virginia-hughes.txt': 1, 'data/virginia-woolf.txt': 1, 'data/willa-cather.txt': 1} 

weights = learnPredictor(trainingLabels, None, extractWordFeatures, 10, 0.01)

