import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from modelHelper import*

def classify(pathToFolder):
    trainY = np.array([2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 3, 3, 3])
    # trainY = np.array([2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 3, 3])



    # create train set
    pathToTrain = pathToFolder+'/train'
    trainX = getFeatures(pathToTrain, True)

    # create test set
    pathToTest= pathToFolder+'/test'
    testX = getFeatures(pathToTest, False)

    AB = AdaBoostClassifier()
    AB.fit(trainX, trainY)
    testY = AB.predict(testX)
    
    print(testY)
    print([classes[x] for x in testY])

classify('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF')
