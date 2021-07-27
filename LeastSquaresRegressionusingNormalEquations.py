#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from sklearn.model_selection import KFold
from numpy.linalg import inv
import numpy as np


class LeastSquaresRegressionusingNormalEquations:

    def __init__(self, dataset):
        self.dataset = dataset

    def normalize(
        self,
        dataset,
        attributeMeans=[],
        attributeStds=[],
        ):
        normalizedDataset = dataset.copy()
        attributes = dataset.shape[1] - 1

        if len(attributeMeans) == attributes:
            for i in range(attributes):
                mean = attributeMeans[i]
                std = attributeStds[i]
                normalizedDataset[i] = \
                    normalizedDataset[i].apply(lambda x: ((x - mean)
                        / std if std > 0 else 0))
        else:

            attributeMeans = []
            attributeStds = []
            for i in range(attributes):
                mean = dataset[i].mean()
                attributeMeans.append(mean)
                std = dataset[i].std()
                attributeStds.append(std)
                normalizedDataset[i] = \
                    normalizedDataset[i].apply(lambda x: ((x - mean)
                        / std if std > 0 else 0))

        return (normalizedDataset, attributeMeans, attributeStds)

    def addingConstantFeature(self, dataset):
        result = dataset.copy()
        lastCol=result.shape[1] + 1
        result.columns = range(1, lastCol)
        result.insert(0, 0, 1)
        return result
    
    def predictResult(self, row, w):
        h = 0
        attributes = len(row) - 1
        for i in range(attributes):
            h = h +  row[i] * w[i] 
        result=h - row[attributes]
        return result


    def getRMSE(self, dataset, w):
        SSE = 0
        for (index, row) in dataset.iterrows():
            SSE =SSE+ self.predictResult(row, w) ** 2
        rmse=(SSE, (SSE / dataset.shape[0]) ** .5)
        return rmse


    def normalEquation(self, dataset):
        attributes = dataset.shape[1] - 1
        x = dataset.iloc[:, :-1].values
        y = dataset[attributes]
        return np.dot(np.dot(inv(np.dot(x.transpose(), x)),
                      x.transpose()), y)

    def validate(self):
        trainSSEset = []
        trainRMSEset = []
        testSSEset = []
        testRMSEset = []
        currentFold = 1
        print ('currentFold\tSSE Trained\tRMSE Trained\tSSE Tested\tRMSE Testd')
        for (train_index, test_index) in KFold(n_splits=10,
                shuffle=True).split(self.dataset):
            (trainDataset, trainMeans, trainStds) = \
                self.normalize(self.dataset.iloc[train_index])
            trainDataset = self.addingConstantFeature(trainDataset)
            w = self.normalEquation(trainDataset)

            (trainSSE, trainRMSE) = self.getRMSE(trainDataset, w)
            trainSSEset.append(trainSSE)
            trainRMSEset.append(trainRMSE)

            (testDataset, testMeans, testStds) = \
                self.normalize(self.dataset.iloc[test_index],
                               trainMeans, trainStds)
            testDataset = self.addingConstantFeature(testDataset)

            (testSSE, testRMSE) = self.getRMSE(testDataset, w)
            testSSEset.append(testSSE)
            testRMSEset.append(testRMSE)

            print ('{}\t{}\t{}\t{}\t{}'.format(currentFold, trainSSE,
                    trainRMSE, testSSE, testRMSE))
            currentFold += 1

        print ('{}\t{}\t{}\t{}\t{}'.format('Mean', np.mean(trainSSEset),
                np.mean(trainRMSEset), np.mean(testSSEset),
                np.mean(testRMSEset)))
        print ('{}\t{}\t{}\t{}\t{}'.format('Standard Deviation',
                np.std(trainSSEset), np.std(trainRMSEset),
                np.std(testSSEset), np.std(testRMSEset)))



			