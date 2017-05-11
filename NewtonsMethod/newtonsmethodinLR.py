# -*- coding:utf-8 -*-
'''
    implement newton's method in logistic regression
'''
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def data():
    dataX = pd.read_csv("./logistic_x.txt", header=None, sep="\s+")
    dataY = pd.read_csv("./logistic_y.txt", header=None, sep="\s+")
    return dataX, dataY

def updateTheta(dataX, dataY, theta):
    row = dataX.shape[0]
    grad = [0.0 for i in range(col)]
    H = np.matrix(np.zeros([col, col], float))
    for i in range(row):
        xi = dataX.iloc[i, :]
        yi = dataY.iloc[i, 0]
        exp = yi * np.dot(np.matrix(xi), np.matrix(theta).getT()).getA()
        prob = 1.0 / (1.0 + np.e**(exp))[0][0]
        H += prob * (1 - prob) * np.dot(np.matrix(xi).getT(), np.matrix(xi))
        grad += xi * prob * yi
    grad = -grad / row
    H = H / row
    theta = theta - np.dot(H.getI(), grad)
    return theta

def drawScatter(dataX, dataY, theta):
    fig = plt.figure()
    ax = plt.subplot(211)
    row = dataX.shape[0]
    col = dataX.shape[1]
    #draw samples
    typeX1 = pd.DataFrame(None)
    typeX2 = pd.DataFrame(None)
    for i in range(row):
        if(dataY.iloc[i, 0]) == 1:
            typeX1 = typeX1.append(dataX.iloc[i, :col-1])
        else:
            typeX2 = typeX2.append(dataX.iloc[i, :col-1])
    type1 = ax.scatter(typeX1.iloc[:, 0], typeX1.iloc[:, 1], s=40, c='blue')
    type2 = ax.scatter(typeX2.iloc[:, 0], typeX2.iloc[:, 1], s=40, c='red')
    #draw hypothesis
    dataX = np.matrix(dataX)
    min_x = min(dataX[:, 0])[0, 0]
    max_x = max(dataX[:, 0])[0, 0]
    theta = np.matrix(theta).getT().getA()
    y_min = float(-theta[2] - theta[0] * min_x) / theta[1]
    y_max = float(-theta[2] - theta[0] * max_x) / theta[1]
    plt.plot([min_x, max_x], [y_min, y_max], '-g')
    #finally show
    plt.show()

if __name__ == "__main__":
    MAX_ITTER = 20
    dataX, dataY = data()
    col = dataX.shape[1]
    theta = [0.0 for i in range(col+1)]
    dataX[col] = 1.0
    col += 1
    for j  in range(MAX_ITTER):
        theta = updateTheta(dataX, dataY, theta)
    print(theta)
    drawScatter(dataX, dataY, theta)
