from __future__ import division
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy.linalg import inv
import numpy as np
import numpy


class RidgeRegression:
    def __init__(self, dataSet, p, c):
        self.dataSet = dataSet.fillna(0)
        self.p = p
        self.c = c

    def transferlDataSet(self, dataset, p):
        columns = dataset.shape[1] - 1
        powerDataset = pd.DataFrame(index=dataset.index)
        for i in range(1, p+1):
            powerDataset[np.arange(columns * (i - 1), columns * i)] = np.power(dataset[range(columns)], i)
            powerDataset[columns * p] = dataset[columns]
        return powerDataset

    def normalize(self, dataset, p, means=[]):
        normalizedDataset = self.transferlDataSet(dataset, p)
        attributes = normalizedDataset.shape[1] - 1
        if len(means) == attributes:
            for i in range(attributes):
                mean = means[i]
                normalizedDataset[i] = normalizedDataset[i].apply(lambda x: x - mean)

        elif len(means) != attributes:
            means = []
            for i in range(attributes):
                mean = normalizedDataset[i].mean()
                means.append(mean)
                normalizedDataset[i] = normalizedDataset[i].apply(lambda x: x - mean)
        return normalizedDataset, means

    def ridgeRegression(self, dataSet, c):
        attributes = dataSet.shape[1] - 1
        X = dataSet.as_matrix(range(attributes))
        y = dataSet[attributes]

        weight = numpy.dot(
            numpy.dot(
                inv(numpy.dot(X.transpose(), X) +numpy.dot(c, numpy.identity(attributes))), X.transpose()),y)
        weight = numpy.insert(weight, 0, y.mean())

        return weight

    def predictresult(self, row, w):
        
        h = w[0]
        attributes = len(row) - 1

        for i in range(attributes):
            h  = h + row[i] * w[i + 1] 
        result = h - row[attributes]

        return result

    def getRMSE(self, dataSet, w):
        RMSE =0

        for index, row in dataSet.iterrows():
            RMSE += self.predictresult(row, w)**2
        result=RMSE, math.sqrt(RMSE / dataSet.shape[0])
        return result

    def validate(self):
        for p in self.p:
            print("p :: {}".format(p))
            TrainRMSE = []
            TestRMSE = []
            for c in self.c:
                print("c :: {}".format(c))
                testSSE = []
                testRMSE = []
                trainSSE = []
                trainRMSE = []
             
                fold = 1
                print("Fold\tSSE Trained\tRMSE Trained\tSSE Tested \tRMSE Tested ")
                for trainIndex, testIndex in KFold(n_splits=10, shuffle=True).split(self.dataSet):
                    trainDataSet, trainAttributeMeans = self.normalize(
                        self.dataSet.iloc[trainIndex], p)
                    weight = self.ridgeRegression(trainDataSet, c)
                    trainSumSquaredErrors, trainRootMeanSquareError = self.getRMSE(
                        trainDataSet, weight)
                    trainSSE.append(trainSumSquaredErrors)
                    trainRMSE.append(trainRootMeanSquareError)
                    testDataSet, testAttributeMeans = self.normalize(
                        self.dataSet.iloc[testIndex],
                        p,
                        trainAttributeMeans,
                    )
                    testSumSquaredErrors, testRootMeanSquareError = self.getRMSE(testDataSet, weight)
                    testSSE.append(testSumSquaredErrors)
                    testRMSE.append(testRootMeanSquareError)
                    print("{}\t{}\t{}\t{}\t{}".format(
                        fold, trainSumSquaredErrors, trainRootMeanSquareError,
                        testSumSquaredErrors, testRootMeanSquareError))
                    fold += 1
                print("{}\t{}\t{}\t{}\t{}".format('Mean', numpy.mean(trainSSE),
                                                  numpy.mean(trainRMSE),
                                                  numpy.mean(testSSE),
                                                  numpy.mean(testRMSE)))
                TrainRMSE.append(numpy.mean(trainRMSE))
                TestRMSE.append(numpy.mean(testRMSE))
                print("{}\t{}\t{}\t{}\t{}".format('Standard Deviation',
                                                  numpy.std(trainSSE),
                                                  numpy.std(trainRMSE),
                                                  numpy.std(testSSE),
                                                  numpy.std(testRMSE)))
            plt.plot(self.c, TrainRMSE, label='Training Data Set')
            plt.plot(self.c, TestRMSE, label='Test Data Set')
            plt.xlabel('c')
            plt.ylabel('Mean RMSE')
            plt.title('Ridge Regression - {}'.format(p))
            plt.legend(loc='best')
            plt.show()
