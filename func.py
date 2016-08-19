import math
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


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
        # Cauchy convergence : gradDescnet has converged to some other point but error is still larger
        # than what user desires. Thus, means we must use a higher order
        # polynomial.
        if(math.fabs(cost_hist[len(cost_hist) - 1] - cost_hist[len(cost_hist) - 2]) < err / 10000):
            ret[0] = 'bad'
            break

        temp = []
        for j in range(n):
            theta_j = ret[j]

            temp.append(theta_j)
            theta_j -= (alpha / m) * (pderiv(j, ret, X, Y))
            temp[j] = theta_j

        for j in range(n):
            ret[j] = temp[j]
        # print(ret)
        J = cost(ret, X, Y)
        # print(J)
        cost_hist.append(J)

    return ret


def cost(theta, X, Y):
    J = 0
    m = len(X)
    for i in range(m):
        J += ((polyEval(theta, X[i]) - Y[i])**2)
    J /= (2 * m)
    return J


def polyEval(theta, x):
    evaluate = theta[0]
    for i in range(len(theta)):
        evaluate += theta[i] * (x**i)
    return evaluate


def pderiv(j, theta, X, Y):
    # Partial derivative of cost function with respect to theta_j
    delta = 0
    m = len(X)

    if(j == 0):
        for i in range(1, m):
            delta += (polyEval(theta, X[i]) - Y[i])
    else:
        for i in range(1, m):
            delta += (polyEval(theta, X[i]) - Y[i]) * (X[i]**j)

    return delta / m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Produces a polynomial that best fits the given data set. Uses Gradient Descent algorithm.')
    parser.add_argument(
        'file', type=str, help='Must pass absolute path to a .csv file (formatted in two columns)')
    parser.add_argument(
        'err', type=float, help='Pass a floating point value that denotes the error threshold (Mean Square Error).')
    arg = parser.parse_args()

    f = arg.file
    err = arg.err

    X = []
    Y = []

    with open(f, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1]))

    plt.scatter(X, Y)

    alpha = 0.0009  # Somewhat arbitrary learning rate
    theta = [0.0]

    coeff = gradDescent(theta, X, Y, err, alpha)
    while(coeff[0] == 'bad'):
        theta.append(0.0)
        coeff = gradDescent(theta, X, Y, err, alpha)

    plt.plot(X, [polyEval(coeff, x) for x in X])
    plt.show()

    print('Coefficients: ' + str(coeff))
    print('Cost: ' + str(cost(coeff, X, Y)))
    print('\n')
