# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:02:47 2019

@author: kyle and brian

"""
from numpy import exp, array, random, dot
import numpy as np

"""
This a function that initialzes the synaptic weights for trying to predict the stock market. It then changes the synaptic weights 
to try to get the inputs to equal the outputs with as little error as possible. 
"""


class NeuralNetwork():
    from numpy import exp, array, random, dot
    def __init__(self):
        # We model a single neuron, with 4 input connections and 1 output connection.
        # We assign random weights to a 4 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        
        self.synaptic_weights = array([[.40], [.40], [.40], [.25]])

    # The Cost Function shows us how our model is performing. We chose MSE to show how far each point
    # deviates from the actual point. The goal is to minimize this number.
    def cost_function(self, features, targets):

        #Find length of list
        N = len(targets)

        predictions = dot(features, self.synaptic_weights)
    
        # Find square error for entire matrix
        sq_error = (predictions - targets)**2

        # Return average squared error among predictions
        return 1.0/(2*N) * sq_error.sum()

    # The derivative of the MSE cost function.
    # This is the gradient of the MSE function that uses partial derivatives of each parameter.
    # It indicates how confident we are about the existing weight.
    def update_weights(self,features, targets, lr):

        # Find predictions using w1x1 + w2x2 + w3x3...concept 
        predictions = dot(features, self.synaptic_weights)

        #Extract our features
        x1 = features[:,0]
        x2 = features[:,1]
        x3 = features[:,2]
        x4 = features[:,3]

        """ 
        print (x1)
        print (x2)
        print (x3)
        print (x4)
        """
        # Use matrix cross product (*) to simultaneously
        # calculate the derivative for each weight
        d_w1 = -x1*(targets - predictions)
        d_w2 = -x2*(targets - predictions)
        d_w3 = -x3*(targets - predictions)
        d_w4 = -x4*(targets - predictions)

        # Multiply the mean derivative by the learning rate
        # and subtract from our weights (remember gradient points in direction of steepest ASCENT)
        self.synaptic_weights[0][0] -= (lr * np.mean(d_w1))
        self.synaptic_weights[1][0] -= (lr * np.mean(d_w2))
        self.synaptic_weights[2][0] -= (lr * np.mean(d_w3))
        self.synaptic_weights[3][0] -= (lr * np.mean(d_w4))
        
        
    def get_output(self,feature):
        # Return predictions using w1x1 + w2x2 + w3x3... concept 
        return dot(feature,self.synaptic_weights)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs,training_set_outputs)
            # Show the MSE of each iteration
            #print ("MSE:")
            #print (output)
            # Update the weights by calling the function
            self.update_weights(training_set_inputs,training_set_outputs,.8)
            #Show new weights to see the change
            #print ("New weights:")
            #print (self.synaptic_weights)

    # The neural network thinks.
    def think(self, inputs, outputs):
        # Pass inputs through our neural network (our single neuron).
        return self.cost_function(inputs,outputs)
