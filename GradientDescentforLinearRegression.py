#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from sklearn.model_selection import KFold
import numpy as np
import random
import matplotlib.pyplot as plt


class GradientDescentforLinearRegression:

    def __init__(
        self,
        dataset,
        learningRate,
        tolerance,
        ):
        
        self.dataset = dataset
        self.learningRate = learningRate
        self.tolerance = tolerance

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
                    normalizedDataset[i].apply(lambda x: ((x - mean)/ std if std > 0 else 0))
                    
        elif len(attributeMeans) != attributes:
            attributeMeans = []
            attributeStds = []
            for i in range(attributes):
                mean = dataset[i].mean()
                attributeMeans.append(mean)
                std = dataset[i].std()
                attributeStds.append(std)
                normalizedDataset[i] = \
                    normalizedDataset[i].apply(lambda x: ((x - mean)/ std if std > 0 else 0))

        return (normalizedDataset, attributeMeans, attributeStds)

    def addingConstantFeature(self, dataset):
        result = dataset.copy()
        lastCol=result.shape[1] + 1
        result.columns = range(1, lastCol)
        result.insert(0, 0, 1)
        return result

    def predictResult(self, row, weight):
        attributes = len(row) - 1
        h = 0
        for each in range(attributes):
            h = h +  row[each] * weight[each] 
        result=h - row[attributes]
        return result
    
    def getRMSE(self, dataset, weight):
        SSE = 0
        for (ind, row) in dataset.iterrows():
            SSE = SSE+self.predictResult(row, weight) ** 2
        result=(SSE, (SSE / dataset.shape[0]) ** .5)
        return result

    def gradientDescent(self, dataset, plot):
        RMSE = 0
        RMSEList = []
        attributes = dataset.shape[1] - 1
        weight = np.zeros(attributes)

        for i in range(1000):
            SSE = 0
            for (ind, row) in dataset.iterrows():
                error = self.predictResult(row, weight)
                SSE =SSE+ error ** 2
                for j in range(attributes):
                    weight[j] = weight[j] - self.learningRate * error * row[j]

            currentRMSE = (SSE / dataset.shape[0]) ** .5
            RMSEList.append(currentRMSE)

            if i == 0 or RMSE - currentRMSE > self.tolerance:
                RMSE = currentRMSE
            else:
                break

        if plot:
            plt.plot(RMSEList)
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('Gradient Descent')
            plt.show()

        return weight



    def validate(self):
        trainRMSEset = []
        trainSSEset = []
        testSSEset = []
        testRMSEset = []
        currentFold = 1
        plotFoldPlan = random.randint(1, 10)
        print ('Planed PLOTTED FOLD :: {}'.format(plotFoldPlan))
        print ('currentFold\tSSE Trained\tRMSE Trained\tSSE Tested\tRMSE Testd')
        for (train_index, test_index) in KFold(n_splits=10,
                shuffle=True).split(self.dataset):
            (trainDataset, trainMeans, trainStds) = \
                self.normalize(self.dataset.iloc[train_index])
            trainDataset = self.addingConstantFeature(trainDataset)
            w = self.gradientDescent(trainDataset, currentFold == plotFoldPlan)

            (trainSSE, trainRMSE) = self.getRMSE(trainDataset, w)
            trainRMSEset.append(trainRMSE)
            trainSSEset.append(trainSSE)
         

            (testDataset, testMeans, testStds) = \
                self.normalize(self.dataset.iloc[test_index],
                               trainMeans, trainStds)
            testDataset = self.addingConstantFeature(testDataset)

            (testSSE, testRMSE) = self.getRMSE(testDataset, w)
            testRMSEset.append(testRMSE)
            testSSEset.append(testSSE)
           

            print ('{}\t{}\t{}\t{}\t{}'.format(currentFold, trainSSE,
                    trainRMSE, testSSE, testRMSE))
            currentFold += 1

        print( '{}\t{}\t{}\t{}\t{}'.format('Mean', np.mean(trainSSEset),
                np.mean(trainRMSEset), np.mean(testSSEset),
                np.mean(testRMSEset)))
        print ('{}\t{}\t{}\t{}\t{}'.format('Standard Deviation',
                np.std(trainSSEset), np.std(trainRMSEset),
                np.std(testSSEset), np.std(testRMSEset)))



			