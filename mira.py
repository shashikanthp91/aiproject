# Mira implementation
import util
import math
import itertools
import time
import random
import numpy as np
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
    self.features = trainingData[0].keys() # could be useful later
    combinedData = list(zip(trainingData,trainingLabels))
    randomCombinedData = random.sample(combinedData,100)
    trainingData,trainingLabels = zip(*randomCombinedData)
    tic = time.clock()    
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)
    tok = time.clock()
    print "difference in time"
    print (tok - tic)*1000

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    "*** YOUR CODE HERE ***"
    self.minstepsize = util.Counter()
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      for trainingLabel,tData in itertools.izip(trainingLabels,trainingData):
        score = util.Counter()
        for legalLabel in self.legalLabels:
          score[legalLabel] = self.weights[legalLabel] * tData
        maxLabel = score.argMax()
        if maxLabel == trainingLabel :
          continue
        else :
          self.rms = self.getRMS(tData)
          self.stepsize = (((self.weights[maxLabel] - self.weights[trainingLabel]) * tData) + 1) / (2 * self.rms)
          self.minstepsize = min(self.C,self.stepsize)
          newlist = util.Counter()
          for key,value in itertools.izip(tData.keys(),tData.values()):
            newlist[key] = value * self.minstepsize
          self.weights[maxLabel] -= newlist 
          self.weights[trainingLabel] += newlist

  def getRMS(self, tData):
    sum = 0
    for c in tData.values():
      sum += c**2
    return math.sqrt(sum)

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"

    return featuresOdds

