import math
import csv
import argparse

def gradDescent(theta, X, Y, err, alpha):
	ret = []
	cost_hist = []
	J = float("inf")
	cost_hist.append(0.0)
	cost_hist.append(J)

	m = len(X)
	n = len(theta)

	for i in range(n):
		ret.append(theta[i])

	while(J > err):
		#Cauchy convergence : gradDescnet has converged to some other point but error is still larger
		#than what user desires. Thus, means we must use a higher order polynomial.
		if(math.fabs(cost_hist[len(cost_hist)-1]-cost_hist[len(cost_hist)-2]) < err/1000.0):
			ret[0] = 'bad'
			break

		temp = []
		for j in range(n):
			theta_j = ret[j]

			temp.append(theta_j)
			for i in range(m):
				theta_j -= (alpha/m)*(pderiv(j, ret, X, Y))
			temp[j] = theta_j
		
		for j in range(n):
			ret[j] = temp[j];
		J = cost(ret, X, Y)
		cost_hist.append(J)	

	return ret

def cost(theta, X, Y):
	J = 0
	m = len(X)
	for i in range(m):
		J += ((polyEval(theta, X[i])-Y[i])**2)
	J /= (2*m)
	return J

def polyEval(theta, x):
	eval = theta[0]
	for i in range(len(theta)):
		eval += theta[i]*(x**i)
	return eval

def pderiv(j, theta, X, Y):
	delta = 0
	m = len(X)

	if(j == 0):
		for i in range(1, m):
			delta += (polyEval(theta, X[i])-Y[i])
	else:
		for i in range(1, m):
			delta += (polyEval(theta, X[i])-Y[i])*(X[i]**j)

	return delta/m

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Produces a polynomial that best fits the given data set. Uses Gradient Descent algorithm.')
	parser.add_argument('file', type=str, help = 'Must pass absolute path to a .csv file (formatted in two columns)')
	parser.add_argument('err', type=float, help = 'Pass a floating point value that denotes the error threshold (Mean Square Error).')
	arg = parser.parse_args()

	f = arg.file
	err = arg.err;

	X = []
	Y = []

	with open(f, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		for row in reader:
			X.append(float(row[0]))
			Y.append(float(row[1]))

	print(X)
	print(Y)
	alpha = 0.01
	theta = [0.0]

	coeff = gradDescent(theta, X, Y, err, alpha)
	while(coeff[0] == 'bad'):
		theta.append(0.0)
		coeff = gradDescent(theta, X, Y, err, alpha)

	print('\n')
	print(coeff)
