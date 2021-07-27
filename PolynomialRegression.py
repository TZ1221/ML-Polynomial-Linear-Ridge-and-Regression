#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from sklearn.model_selection import KFold
from numpy.linalg import inv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PolynomialRegression:

    def __init__(
        self,
        trainDataset,
        validationDataset,
        p,
        ):
        self.trainingDataset = trainDataset
        self.validationDataset = validationDataset
        self.p = p


    def normalization(
        self,
        dataset,
        p,
        means=[],
        stds=[],
        ):
        
        columns = dataset.shape[1] - 1
        normalizedDataset = pd.DataFrame(index=dataset.index)
        for i in range(1, p + 1):
            normalizedDataset[np.arange(columns * (i - 1), columns * i)] = \
                np.power(dataset[range(columns)], i)
        normalizedDataset[columns * p] = dataset[columns]
        

        attributes = normalizedDataset.shape[1] - 1
        if len(means) == attributes:
            for i in range(attributes):
                mean = means[i]
                std = stds[i]
                normalizedDataset[i] = \
                    normalizedDataset[i].apply(lambda x: ((x - mean)
                        / std if std > 0 else 0))
        else:

            means = []
            stds = []
            for i in range(attributes):
                mean = normalizedDataset[i].mean()
                means.append(mean)
                std = normalizedDataset[i].std()
                stds.append(std)
                normalizedDataset[i] = \
                    normalizedDataset[i].apply(lambda x: ((x - mean)
                        / std if std > 0 else 0))

        return (normalizedDataset, means, stds)
    
    

    def addingConstantFeature(self, dataset):
        result = dataset.copy()
        lastCol=result.shape[1] + 1
        result.columns = range(1, lastCol)
        result.insert(0, 0, 1)
        return result

    def normalEquation(self, dataset):
        attributes = dataset.shape[1] - 1
        x = dataset.iloc[:, :-1].values
        y = dataset[attributes]
        return np.dot(np.dot(inv(np.dot(x.transpose(), x)),
                      x.transpose()), y)

    def getRMSE(self, dataset, weight):
        SSE = 0
        for (index, row) in dataset.iterrows():
            SSE =SSE+ self.predictResult(row, weight) ** 2
        result=(SSE, (SSE / dataset.shape[0]) ** .5)
        return  result

    def predictResult(self, row, weight):
        h = 0.0
        attributes = len(row) - 1
        for i in range(attributes):
            h += weight[i] * row[i]
        return h - row[attributes]

    def validate(self):
        trainSSEs = []
        testSSEs = []



        if self.validationDataset is not None:
            for p in self.p:
                print ('Sinusoid data section')
                print ('p :: {}'.format(p))
                print ('SSE Trained\tRMSE Trained \tSSE Tested \tRMSE Tested')
                (trainDataset, trainMeans, trainStds) = \
                    self.normalization(self.trainingDataset, p)
                trainDataset = self.addingConstantFeature(trainDataset)
                w = self.normalEquation(trainDataset)

                (trainSSE, trainRMSE) = self.getRMSE(trainDataset, w)
                trainSSEs.append(trainSSE / trainDataset.shape[0])

                (testDataset, testMeans, testStds) = \
                    self.normalization(self.validationDataset, p,
                        trainMeans, trainStds)
                testDataset = self.addingConstantFeature(testDataset)
                (testSSE, testRMSE) = self.getRMSE(testDataset, w)
                testSSEs.append(testSSE / testDataset.shape[0])
                print ('{}\t{}\t{}\t{}'.format(trainSSE, trainRMSE,
                        testSSE, testRMSE))
        else:



            for p in self.p:
                print ('  Yacht data section')
                print ('p :: {}'.format(p))
                _trainSSEs = []
                _trainRMSEs = []
                _testSSEs = []
                _testRMSEs = []
                fold = 1
                print ('Fold\tSSE Train\tRMSE Trained\tSSE Tested\tRMSE Tested')
                for (train_index, test_index) in KFold(n_splits=10,
                        shuffle=True).split(self.trainingDataset):
                    (trainDataset, trainMeans, trainStds) = \
                        self.normalization(self.trainingDataset.iloc[train_index],
                            p)
                    trainDataset = self.addingConstantFeature(trainDataset)
                    w = self.normalEquation(trainDataset)

                    (trainSSE, trainRMSE) = self.getRMSE(trainDataset, w)
                    _trainSSEs.append(trainSSE)
                    _trainRMSEs.append(trainRMSE)

                    (testDataset, testMeans, testStds) = \
                        self.normalization(self.trainingDataset.iloc[test_index],
                            p, trainMeans, trainStds)
                    testDataset = self.addingConstantFeature(testDataset)

                    (testSSE, testRMSE) = self.getRMSE(testDataset, w)
                    _testSSEs.append(testSSE)
                    _testRMSEs.append(testRMSE)

                    print ('{}\t{}\t{}\t{}\t{}'.format(fold, trainSSE,
                            trainRMSE, testSSE, testRMSE))
                    fold += 1

                print ('{}\t{}\t{}\t{}\t{}'.format('Mean',
                        np.mean(_trainSSEs), np.mean(_trainRMSEs),
                        np.mean(_testSSEs), np.mean(_testRMSEs)))
                print ('{}\t{}\t{}\t{}\t{}'.format('Standard Deviation',
                        np.std(_trainSSEs), np.std(_trainRMSEs),
                        np.std(_testSSEs), np.std(_testRMSEs)))

                trainSSEs.append(np.mean(_trainRMSEs))
                testSSEs.append(np.mean(_testRMSEs))

        plt.plot(self.p, trainSSEs, label='Train Dataset')
        plt.plot(self.p, testSSEs, label='Test Dataset')
        plt.xlabel('p')
        plt.ylabel(('Mean RMSE' if self.validationDataset
                   is None else 'Mean SSE'))
        plt.title('Polynomial Regression')
        plt.legend()
        plt.show()



			