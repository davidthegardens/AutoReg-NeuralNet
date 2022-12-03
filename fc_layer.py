from layer import Layer
import numpy as np
import pandas as pd

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def forward_prop_from_load(self,input_data,weight,bias):
        self.input = input_data
        self.output = np.dot(self.input, weight) + bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
    def save(self,modeldf,iter):
        #print(np.reshape(self.weights,self.weights.shape[0]*self.weights.shape[1]))
        modeldf[str(iter)+' Weights']=np.reshape(self.weights,self.weights.shape[0]*self.weights.shape[1])
        modeldf[str(iter)+' Bias']=np.reshape(self.bias,self.bias.shape[1])
        return modeldf

    def what(self):
        return 'FC'
    
    def export(self):
        return self.weights,self.bias