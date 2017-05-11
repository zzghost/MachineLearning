# -*- coding: utf-8 -*-
'''
A logistic regression using stochastic gradient descent.
'''
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib

LEARNING_RATE = 0.1
THRESHOLD = 0.0001
#MAX_ITER = 20

def get_data():
	dataX = pd.read_csv("./logistic_x.txt", header=None, sep="\s+")
	dataY = pd.read_csv("./logistic_y.txt", header=None, sep="\s+")
	dataX[dataX.shape[1]] = 1.0
	return dataX, dataY

def sigmoid(theta, dataX):
	return (1.0 / (1.0 + np.exp(-(np.dot(theta, np.matrix(dataX).getT())))))

def updateTheta(theta, dataX, dataY):
	row = dataX.shape[0]
	col = dataX.shape[1]
	randRow = random.randint(0, row-1)
	xi = dataX.iloc[randRow, :]
	hypo = sigmoid(theta, xi)
	theta = theta.getT()
	newTheta = np.matrix(theta)
	for j in range(col):
		xj = dataX.iloc[randRow, j]
		yi = dataY.iloc[randRow, 0]
		newTheta[j] = theta[j] + LEARNING_RATE * (yi - hypo) * xj
	return newTheta.getT()

def distance(value1, value2):
	dst = 0
	value1 = np.matrix(value1).getT()
	value2 = np.matrix(value2).getT()
	for i in range(len(value1)):
		dst += pow(value1[i] - value2[i], 2)
	return dst

def loss(theta, dataX, dataY):
	row = dataX.shape[0]
	J = 0
	for i in range(row):
		yi = dataY.iloc[i, 0]
		xi = dataX.iloc[i, :]
		hypo = sigmoid(theta, xi)
		J += distance(yi, hypo)
	return J / 2

def drawScatter(theta, dataX, dataY):
	fig = plt.figure()
	ax = plt.subplot(111)
	#ax.scatter(dataX.iloc[:, 0], dataX.iloc[:, 1], c='r')
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
	#print(min_x, max_x)
	y_min = float(-theta[2] - theta[0] * min_x) / theta[1]
	y_max = float(-theta[2] - theta[0] * max_x) / theta[1]
	plt.plot([min_x, max_x], [y_min, y_max], '-g')
	plt.show()



if __name__ == "__main__":
	dataX, dataY = get_data()
	row = dataX.shape[0]
	#dataX[dataX.shape[1]] = 1.0
	col = dataX.shape[1]
	#init theta
	theta = np.matrix([0.0 for i in range(col)])
	newTheta = np.matrix([10.0 for i in range(col)])

	while(True):
		newTheta = updateTheta(theta, dataX, dataY)
		print(newTheta, theta)
		if(distance(newTheta, theta) < THRESHOLD):
			break
		else:
			theta = newTheta
	'''
	for i in range(MAX_ITER):
		newTheta = updateTheta(theta, dataX, dataY)
	theta = newTheta
	'''
	print(theta)
	print("the loss is :%f" % loss(theta, dataX, dataY))
	drawScatter(theta, dataX, dataY)