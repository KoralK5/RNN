import numpy as np

def tanh(x):
    x_e = np.exp(x)
    return (x_e - 1/x_e) / (x_e + 1/x_e)

def sigmoid(x):
    return 1/(1 + 1/np.exp(x))
