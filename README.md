# Function-Classification
This small project is a tiny taste of machine learning (supervised learning and regression). This code, when given an arbitrary set of `(x, y)` 
coordinates and error threshold, produces a polynomial function that approximates the data within the error margin. This implementation uses
the gradient descent algorithm to determine the values of the coefficients. I hope to extend the scope of this project from strictly polynomials
to more exotic functions, such as trigonometric, radical, and rational functions among others, which is a quite ambitious endeavor.

## Usage
This module takes a data set of `(x, y)` coordinates from a `.csv` file, with each coordinate pair delimited by a newline. The format should be as follows:
```
x_1,y_1
x_2,y_2
x_3,y_3
...
...
...
x_M,y_M
```
Furthermore, an error threshold must be passed, and a polynomial function will be generated with the overall cost falling underneath the bound. Running the module is done as follows
```
python func.py (DATASET.csv) (ERROR <float>)
```
The module produces a graph with both the points from the dataset and best fit polynomial plotted on. In addition, the coefficients and cost of the best fit are written to a `.csv` in the current working directory.

## The Algorithm
[Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is used to produce the best fit polynomial. The general idea behind gradient descent is to descend down a cost surface that is defined over our coefficient parameters to a minimum cost. From multivariable calculus, we know that the direction of maximum increase in cost is given by the gradient of the cost surface, which directly implies that the minimum is simply in the opposite direction. With the knowledge of this direction, we can incrementally move down with some distance alpha (known as the learning rate). At each step, we recalculate the gradient so we are sure to get to the minimum and not overshoot. 

Given a set of coordinates, we first start off with polynomials of order 1 (i.e. linear). A cost surface exists for this polynomial as the cost surface is parameterized by the order of hypothesis polynomial. In other words, each order has its own associated cost surface. However, it is not known _a priori_ whether the data should be fit with linear, quadratic, cubic, etc. So the real issue is when to switch from trying linear to quadratic, and so on. 

Here we use the notion of [Cauchy sequences](https://en.wikipedia.org/wiki/Cauchy_sequence), in which we look at the sequence of differences between costs. As we move down the cost surface, we see if the difference between the cost from the previous step and our current cost is below a particular threshold. If it is, we heuristically know that the cost sequence is converging. If we hit this threshold and our cost is still not underneath our original error bound, we know that the current order of polynomial is not a good fit, as it will never reach the bound. Thus we break and try the next order of polynomial. This process repeats until we get a polynomial that has a cost underneath the bound.
