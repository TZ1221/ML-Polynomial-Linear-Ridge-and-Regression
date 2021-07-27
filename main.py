from GradientDescentforLinearRegression import GradientDescentforLinearRegression
from LeastSquaresRegressionusingNormalEquations import LeastSquaresRegressionusingNormalEquations
from PolynomialRegression import PolynomialRegression
from RidgeRegression import RidgeRegression
import pandas as pd
import numpy as np

def load(filepath):
    data = pd.read_csv(filepath, header=None)
    result=data.drop_duplicates()
    return  result

if __name__ == '__main__':


    print("Problem 2 Housing - Gradient Descent")
    print()
    GradientDescentforLinearRegression(load('./dataset/housing.csv').fillna(0), 0.0004, 0.005).validate()
    print("END: Problem 2 Housing - Gradient Descent")
    print('-=====================================-')
    print()
    
    print("Problem 2 Yacht - Gradient Descent")  
    print()
    GradientDescentforLinearRegression(load('./dataset/yachtData.csv').fillna(0), 0.001, 0.001).validate()
    print("END: Problem 2 Yacht - Gradient Descent")
    print('-=====================================-')
    print()
    
    print("Problem 2 Concrete - Gradient Descent")  
    print()
    GradientDescentforLinearRegression(load('./dataset/concreteData.csv').fillna(0), 0.0007, 0.0001).validate()
    print("END: Problem 2 Concrete - Gradient Descent")
    print('-=====================================-')
    print()




    print("Problem 3 Housing - Normal Equation")
    print()
    LeastSquaresRegressionusingNormalEquations(load('./dataset/housing.csv').fillna(0)).validate()
    print("END: Problem 3 Housing - Normal Equation")
    print('-=====================================-')
    print()

    print("Problem 3 Yacht - Normal Equation")
    print()
    LeastSquaresRegressionusingNormalEquations(load('./dataset/yachtData.csv').fillna(0)).validate()
    print("END: Problem 3 Yacht - Normal Equation")
    print('-=====================================-')
    print()


    print("Problem 5 Sinusoid - Polynomial Regression")
    print()
    PolynomialRegression(load('./dataset/sinData_Train.csv').fillna(0), 
    load('./dataset/sinData_Validation.csv').fillna(0), np.arange(1, 16)).validate()
    print("END: Problem 5 Sinusoid - Polynomial Regression")
    print('-=====================================-')
    print()


    print("Problem 5 Yacht - Polynomial Regression")
    print()
    PolynomialRegression(load('./dataset/yachtData.csv').fillna(0), None, np.arange(1, 8)).validate()
    print("END: Problem 5 Yacht - Polynomial Regression")
    print('-=====================================-')
    print()


 
    print('Problem 7 Sinusoid Ridge Regression - 1')
    print()
    RidgeRegression(load('./dataset/sinData_Train.csv').fillna(0), np.arange(1, 6), np.arange(0.0, 10.2, 0.2)).validate()
    print('END : Problem 7 Sinusoid Ridge Regression - 1')
    print('-=====================================-')
    print()

    
    
    
    print('Problem 7 Sinusoid Ridge Regression - 2')
    print()
    RidgeRegression(load('./dataset/sinData_Train.csv').fillna(0), np.arange(1, 10), np.arange(0.0, 10.2, 0.2)).validate()
    print('end: Problem 7 Sinusoid Ridge Regression - 2')
    print('-=====================================-')
    print()
