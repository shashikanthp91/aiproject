import util
import classificationMethod
import math
import itertools
import time
import random
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]

    combinedData = list(zip(trainingData,trainingLabels))
    randomCombinedData = random.sample(combinedData,100)
    trainingData,trainingLabels = zip(*randomCombinedData)
    tic = time.clock()    
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
    tok = time.clock()
    print "difference in time"
    print (tok - tic)*1000
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    self.features = [0,1]
    keyCounter = util.Counter()         #this is a counter for features i.e c(fi,y)
    labelCounter = util.Counter()
    self.probabilityofLabelCounter = util.Counter()     #calculating probability i.e P(F=fi|Y=y)
    probabilityCounter = util.Counter()
    for legalLabel in self.legalLabels:         #iterate through the legal labels i.e from 0 to 9
      for trainingLabel,tData in itertools.izip(trainingLabels,trainingData):         #iterate through the training data and if training label matches with legal label proceed.
        if trainingLabel == legalLabel:
          labelCounter[legalLabel] += 1       #this is to keep track of the number of training(label) instances for each legal label
          for key,value in itertools.izip(tData.keys(),tData.values()):     
            keyCounter[key,value,legalLabel] += 1       #c(fi,y)
            keyCounter[key,legalLabel] += 1       #sigma(c(fi',y))
         
    # this is to calculate P(Y)
    for label in self.legalLabels:
      self.probabilityofLabelCounter[label] = float(labelCounter[label])/labelCounter.totalCount()


    self.probabilityCounter = util.Counter()
    bestAccuracy = None
    bestProbabilityCounter = {}

    #autotune implemented
    for k in kgrid:
      correct = 0
      for legalLabel in self.legalLabels: 
        for key in trainingData[0].keys():
          for value in self.features:
            probabilityCounter[key,value,legalLabel] = float (keyCounter[key,value,legalLabel] + self.k) / (keyCounter[key,legalLabel] + 2*self.k)    #smoothing
      
      # Check the accuracy associated with this k
      self.probabilityCounter = probabilityCounter                
      guesses = self.classify(validationData)
      for i, guess in enumerate(guesses):
        correct += (validationLabels[i] == guess and 1.0 or 0.0)
        accuracy = correct / len(guesses)
    
      # Keep the best k so far
      if accuracy > bestAccuracy or bestAccuracy is None:
        bestAccuracy = accuracy
        bestProbabilityCounter = probabilityCounter
        self.k = k
    self.probabilityCounter = bestProbabilityCounter
      


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    for legalLabel in self.legalLabels:
      labelValue = 0
      for key,value in itertools.izip(datum.keys(),datum.values()):
        labelValue += math.log(self.probabilityCounter[key,value,legalLabel])
      logJoint[legalLabel] = labelValue + math.log(self.probabilityofLabelCounter[legalLabel])
    return logJoint
    
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    """
    featuresOdds = []
        
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
