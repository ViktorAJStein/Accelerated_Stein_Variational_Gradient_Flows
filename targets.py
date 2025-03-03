# -*- coding: utf-8 -*-
"""
densities and potential gradients of four different target densities
@author: vglom
"""
import numpy as np


def U2_density(x, y):
    return np.exp(- 25/8 * (y - np.sin(np.pi/2*x))**2)


def U2_grad(x):
    factor = U2_density(x[0], x[1]) * (x[1] - np.sin(np.pi / 2 * x[0]))
    return - factor * np.array([25/8 * np.pi * np.cos(np.pi / 2 * x[0]), -25/4])


def U3_density(x, y):
    first = np.exp(-1/2*((y - np.sin(np.pi / 2 * x))/(.35))**2)
    w2 = 3*np.exp(-1/2*((x-1)/(.6))**2)
    second = np.exp(-1/2*((y - np.sin(np.pi / 2 * x) + w2)/(.35))**2)
    return first + second


def U3_grad(x):
    term1 = np.exp(-200/49 * (x[1] - np.sin((np.pi * x[0]) / 2))**2)
    term2 = np.exp(-25/18 * (x[0] - 1)**2)
    term3 = np.exp(-200/49 * (3 * term2 - np.sin((np.pi * x[0]) / 2) + x[1])**2)
    A = 200 * np.pi * np.cos((np.pi * x[0]) / 2) * term1 * (x[1] - np.sin((np.pi * x[0]) / 2))
    B = -25/3 * term2 * (x[0] - 1) - 0.5 * np.pi * np.cos((np.pi * x[0]) / 2)
    C = 3 * term2 - np.sin((np.pi * x[0]) / 2) + x[1]
    num1 = -(A - 2 * B * C * term3)
    den1 = 49 * (term3 + term1)
    first_expr = num1 / den1
    num2 = (-400/49 * C * term3 - 400/49 * term1 * (x[1] - np.sin((np.pi * x[0]) / 2)))
    den2 = term3 + term1
    second_expr = - (num2 / den2)
    return np.array([first_expr, second_expr])


def U4_density(x, y):
    first = np.exp(-1/2*((y - np.sin(np.pi / 2 * x))/(.4))**2)
    w3 = 3/(1 + np.exp(10/3*(1-x)))
    second = np.exp(-1/2*((y - np.sin(np.pi / 2 * x) + w3)/(.35))**2)
    return first + second


def U4_grad(x):
    exp_a = np.exp(-25/8 * (x[1] - np.sin((np.pi * x[0]) / 2))**2)
    exp_b = np.exp((10 * (1 - x[0])) / 3)
    common_expr = 3/(exp_b + 1) - np.sin((np.pi * x[0]) / 2) + x[1]
    exp_c = np.exp(-200/49 * common_expr**2)
    denom = exp_c + exp_a
    num1 = -(
        (25/8) * np.pi * np.cos((np.pi * x[0]) / 2) * exp_a * (x[1] - np.sin((np.pi * x[0]) / 2))
        - (400/49) * (
            (10 * exp_b) / ((exp_b + 1)**2) - 0.5 * np.pi * np.cos((np.pi * x[0]) / 2)
        ) * (
            3/(exp_b + 1) - np.sin((np.pi * x[0]) / 2) + x[1]
        ) * exp_c
    )
    first_component = num1 / denom
    num2 = -(
         - (400/49) * common_expr * exp_c
         - (25/4) * exp_a * (x[1] - np.sin((np.pi * x[0]) / 2))
    )
    second_component = num2 / denom
    return np.array([first_component, second_component])


def bimodal_density(x, y):
    first = np.exp(-2*(np.sqrt(x**2 + y**2) - 3)**2)
    second = np.exp(-2*(x-3)**2) + np.exp(-2*(x+3)**2)
    return first * second


def bimodal_grad(x):
    nx = np.linalg.norm(x)
    first = 4 * x * (1 - 3 / nx)
    second = np.array([
        8*(np.exp(24*x[0]) * (x[0] - 3) + x[0] + 3) / (np.exp(24*x[0]) + 1),
        0])
    return first + second


def bimodal2_density(x, y):
    first = np.exp(-2*(np.sqrt(x**2 + y**2) - 3)**2)
    second = np.exp(-2*(y-3)**2) + np.exp(-2*(y+3)**2)
    return first * second


def gauss_mix_density(x, y, a1=1/2, a2=1/2):
    first = np.exp(-1/2*(x-a1)**2 - 1/2*(y - a2)**2)
    return 1/(4*np.pi) * (first + np.exp(-1/2*(x+a1)**2 - 1/2*(y + a2)**2))


def gauss_mix_grad(x, a=np.array([1, 1])):
    # gradient of the potential of N(a, Id) + N(-a, Id)
    return x - a + 2 * a / (1 + np.exp(2 * np.dot(x, a)))