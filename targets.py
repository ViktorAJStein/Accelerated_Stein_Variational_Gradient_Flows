# -*- coding: utf-8 -*-
"""
densities and potential gradients of four different target densities
@author: vglom
"""
import numpy as np
import scipy as sp
from scipy.special import iv
from scipy.integrate import dblquad


class target():
    '''
    target measure pi = 1/ Z e^(- f)
    density:    function handle for the unnormalized density e^(-f)
    score:      function handle for \nabla ln(pi) = - \nabla f
    name:       string, name given to this target
    mean:       array of shape (d, ), mean of pi
    cov:        array of shape (d, d), covariance of pi
    lnZ:        float, the normalization constant log(Z)
    '''

    def __init__(self, density, score, name, mean, cov, lnZ):
        self.density = density
        self.score = score
        self.name = name
        self.mean = mean
        self.cov = cov
        self.lnZ = lnZ


# U from ChatGPT is my f
# overdamped (ULA) is \dot{x} = - \nabla U(x) + sigma * N(0, 1)


# TODO: add mixture of eights Gaussians in a circle, all very disjoint


def U2_density(x, y):
    return np.exp(- 25/8 * (y - np.sin(np.pi/2*x))**2)


def U2_grad(x):
    factor = U2_density(x[0], x[1]) * (x[1] - np.sin(np.pi / 2 * x[0]))
    return factor * np.array([25/8 * np.pi * np.cos(np.pi / 2 * x[0]), -25/4])


def U3_density(x, y):
    first = np.exp(-1/2*((y - np.sin(np.pi / 2 * x))/(.35))**2)
    w2 = 3*np.exp(-1/2*((x-1)/(.6))**2)
    second = np.exp(-1/2*((y - np.sin(np.pi / 2 * x) + w2)/(.35))**2)
    return first + second


def U3_grad(x):
    term0 = np.sin((np.pi * x[0]) / 2)
    term1 = np.exp(-200/49 * (x[1] - term0)**2)
    term2 = np.exp(-25/18 * (x[0] - 1)**2)
    term3 = np.exp(-200/49 * (3 * term2 - term0 + x[1])**2)
    A = 200 * np.pi * np.cos((np.pi * x[0]) / 2) * term1 * (x[1] - term0)
    B = -25/3 * term2 * (x[0] - 1) - 0.5 * np.pi * np.cos((np.pi * x[0]) / 2)
    C = 3 * term2 - term0 + x[1]
    num1 = A - 2 * B * C * term3
    den1 = 49 * (term3 + term1)
    first_expr = num1 / den1
    num2 = (-400/49 * C * term3 - 400/49 * term1 * (x[1] - term0))
    den2 = term3 + term1
    second_expr = num2 / den2
    return np.array([first_expr, second_expr])


def U4_density(x, y):
    first = np.exp(-1/2*((y - np.sin(np.pi / 2 * x))/(.4))**2)
    w3 = 3/(1 + np.exp(10/3*(1-x)))
    second = np.exp(-1/2*((y - np.sin(np.pi / 2 * x) + w3)/(.35))**2)
    return first + second


def U4_grad(x):
    term0 = np.sin((np.pi * x[0]) / 2)
    term1 = np.pi * np.cos((np.pi * x[0]) / 2)
    exp_a = np.exp(-25/8 * (x[1] - term0)**2)
    exp_b = np.exp((10 * (1 - x[0])) / 3)
    common_expr = 3/(exp_b + 1) - term0 + x[1]
    exp_c = np.exp(-200/49 * common_expr**2)
    denom = exp_c + exp_a
    num1 = (25/8) * exp_a * (x[1] - term0) - (400/49) * ( (10 * exp_b) / ((exp_b + 1)**2) - 0.5 * term1) * (3/(exp_b + 1) - term0 + x[1]) * exp_c
    first_component = num1 / denom
    num2 = - (400/49) * common_expr * exp_c - (25/4) * exp_a * (x[1] - term0)
    second_component = num2 / denom
    return np.array([first_component, second_component])


def bananas_density(x, y):
    first = np.exp(-2*(np.sqrt(x**2 + y**2) - 3)**2)
    second = np.exp(-2*(x-3)**2) + np.exp(-2*(x+3)**2)
    return first * second


def bananas_score(x):
    nx = np.linalg.norm(x)
    first = 4 * x * (1 - 3 / nx)
    second = np.array([
        4*(np.exp(24*x[0]) * (x[0] - 3) + x[0] + 3) / (np.exp(24*x[0]) + 1),
        0])
    return - first - second


def gmm_density(x, y, a1=1/2, a2=1/2):
    first = np.exp(-1/2*(x-a1)**2 - 1/2*(y - a2)**2)
    return 1/(4*np.pi) * (first + np.exp(-1/2*(x+a1)**2 - 1/2*(y + a2)**2))


def gmm_score(x, a=np.array([1, 1])):
    return a - x - 2 * a / (1 + np.exp(2 * np.dot(x, a)))


def skew_Gaussian_score(x):
    return - np.linalg.inv(np.array([[10, 0], [0, .1]])) @ (x - np.ones(2))


def skew_Gaussian_density(x, y):
    point = np.dstack((x, y))
    return sp.stats.multivariate_normal.pdf(point,
                                            mean=np.ones(2),
                                            cov=np.array([[10, 0], [0, .1]]))


skewed_Gaussian = target(skew_Gaussian_density, skew_Gaussian_score,
                         'skewed_Gaussian',
                         np.ones(2), np.array([[10, 0], [0, .1]]),
                         np.log(
                             np.sqrt(2 * np.pi *
                                     np.linalg.det(np.array([[10, 0], [0, .1]]))
                                     )
                             )
                         )


U2 = target(U2_density, U2_grad, 'squiggly', np.zeros(2),
            np.eye(2) + 1/4*np.ones((2, 2)),
            np.log(dblquad(lambda y, x: U2_density(x, y), -10, 10, lambda x: -10, lambda x: 10))[0]
            )

# 1/2 N(a, Id) + 1/2 N(-a, Id)
GMM = target(gmm_density, gmm_score, 'GMM', np.zeros(2),
             np.eye(2) + 1/4*np.ones((2, 2)),
             np.log(dblquad(lambda y, x: gmm_density(x, y), -10, 10, lambda x: -10, lambda x: 10))[0]
             )


U3 = target(U3_density, U3_grad, 'squiggly2', np.zeros(2),
            np.eye(2) + 1/4*np.ones((2, 2)),
            np.log(dblquad(lambda y, x: U3_density(x, y), -10, 10, lambda x: -10, lambda x: 10))[0]
            )


U4 = target(U4_density, U4_grad, 'squiggly3', np.zeros(2),
            np.eye(2) + 1/4*np.ones((2, 2)),
            np.log(dblquad(lambda y, x: U4_density(x, y), -10, 10, lambda x: -10, lambda x: 10))[0])


bananas = target(bananas_density, bananas_score, 'bananas', np.zeros(2),
                 np.array([[1.43, 0], [0, 9.125]]),
                 np.log(dblquad(lambda y, x: bananas_density(x, y), -10, 10, lambda x: -10, lambda x: 10))[0]
                 )
