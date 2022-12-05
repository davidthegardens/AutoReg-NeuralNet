
import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def ReLu(x):
    return np.maximum(x,0,x)
    #return np.max(0,x)

def ReLu_prime(x):
    return 1 * (x > 0)