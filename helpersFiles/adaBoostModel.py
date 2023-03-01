import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from modelHelper import*

def classify2D(pathToFolder, axis):
    trainY = np.repeat([2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3], NUM_OF_SLICES)

    # create train set
    pathToTrain = pathToFolder+'/train'
    trainX = getFeatures2D(pathToTrain, axis)

    # create test set
    pathToTest= pathToFolder+'/test'
    testX = getFeatures2D(pathToTest, axis)

    AB = AdaBoostClassifier()
    AB.fit(trainX, trainY)
    testY = AB.predict(testX)

    testPredictions = []

    for i in range(len(testY)//NUM_OF_SLICES):
        testPredictions.append(np.bincount(testY[i*NUM_OF_SLICES:i*NUM_OF_SLICES+NUM_OF_SLICES]).argmax())
    
    print(f'axis {axis}: ', testPredictions)
    print([classes[x] for x in testPredictions])
    print(f1_score(y_true=np.array([0, 2, 1, 3]), y_pred=testPredictions, average='micro'))


def classify3D(pathToFolder):
    trainY = np.repeat([2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3], 1)

    # create train set
    pathToTrain = pathToFolder+'/train'
    trainX = getFeatures3D(pathToTrain)

    # create test set
    pathToTest= pathToFolder+'/test'
    testX = getFeatures3D(pathToTest)

    AB = AdaBoostClassifier()
    AB.fit(trainX, trainY)
    testY = AB.predict(testX)
    
    print(testY)
    print([classes[x] for x in testY])
    # plotRocCurve(testY, 'AdaBoost 3D')
    print(f1_score(y_true=np.array([0, 2, 1, 3]), y_pred=testY, average='micro'))


# classify3D('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF')
classify2D('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF', axis='A')
classify2D('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF', axis='C')
