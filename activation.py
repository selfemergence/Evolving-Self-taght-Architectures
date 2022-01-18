# -*- coding: utf-8 -*-
import numpy as np


def sigmoid1(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return x*(1.0-x)

def sigmoid(z):
    """
    Sigmoidal activation function
    """
    z = min(40, max(-40, z))
    return 1.0 / (1.0 + np.exp(-4.9 * z))

def tanh(x, derivative=False):
    if (derivative == True):
        return (1 - (x ** 2))
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x*x

def relu(x, derivative=False):
    if derivative:
        if x == 0:
            return 0
        else:
            return 1
    
    if x <= 0:
        return 0
    else:
        return x

def relu_prime(x):
    if x > 0:
        return 1
    elif x <= 0:
        return 0

def linear(x, derivative=False):
    return x

def linear_prime(x):
    return 1

def softmax(x, derivative=False):
    """Compute softmax values for each sets of scores in x."""
    if derivative:
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_prime(x):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def arctan(x, derivative=False):
    if (derivative == True):
        return (np.cos(x) ** 2)
    return np.arctan(x)

def step(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                x[i][k] = 1
            else:
                x[i][k] = 0
    return x

def squash(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(0, len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = (x[i][k]) / (1 + x[i][k])
                else:
                    x[i][k] = (x[i][k]) / (1 - x[i][k])
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            x[i][k] = (x[i][k]) / (1 + abs(x[i][k]))
    return x

def gaussian(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(0, len(x[i])):
                x[i][k] = -2* x[i][k] * np.exp(-x[i][k] ** 2)
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            x[i][k] = np.exp(-x[i][k] ** 2)
    return x