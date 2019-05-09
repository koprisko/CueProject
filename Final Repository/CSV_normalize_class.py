# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:05:51 2019

@author: Brian Heckman and Kyle Oprisko
"""
import csv

"""this file opens a csv file created in the csv creator class. The main purpose of this class is to 
normalize the data in the csv file, so that it can be read by the neural network. 
"""

class CSV_Normalize:
    stock = ""

    # Initialize the lists for the 4 parameters
       
    close_prices = []
    high_prices = []
    prev_prices = []
    sentiments = []
    
    # Initialize max and min values for normalization calc
    
    max_sent = 0.0
    min_sent = 0.0
    min_close = 1000
    max_close = 0  
    min_high = 1000
    max_high = 0
    min_prev = 1000
    max_prev = 0

    # Initialize lists for normalized values of parameters
    
    normalized_close = []
    normalized_high = []
    normalized_prev = []
    normalized_sent = []
   
    # Initialize output parameters
    
    open_prices = []

    # Initialize max and min for normalization calc
    
    min_open= 1000
    max_open = 0

    # Initialize the normalized output list
    
    normalized_open = []

    # Create arrays to separate into training and testing lists
    
    inputs = []
    training_inputs = []
    testing_inputs = []

  
    training_outputs = []
    testing_outputs = []

    # Set name of stock
    
    def set_stock(self,stock):
        self.stock = stock
        
    # Set input values
    
    def set_input(self):
        
        # Open CSV and read each row and append to specific list
        
        with open(self.stock + '.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
            for row in readCSV:
                self.close_prices.append(row[5])
                self.high_prices.append(row[3])
                self.prev_prices.append(row[2])
                self.sentiments.append(row[7])

        # Remove the headers and the last row because the data is trailing
        
        self.close_prices = self.close_prices[1:-1]
        self.high_prices = self.high_prices[1:-1]
        self.prev_prices = self.prev_prices[1:-1]
        self.sentiments = self.sentiments[1:-1]

        # Turn data values into floats
        
        for m in range(len(self.close_prices)):
            if self.close_prices[m] != "Close":
                self.close_prices[m] = float(self.close_prices[m])
        for n in range(len(self.high_prices)):
            if self.high_prices[n] != "High":
                self.high_prices[n] = float(self.high_prices[n])
        for pp in range(len(self.prev_prices)):
            if self.prev_prices[pp] != "Open":
                self.prev_prices[pp] = float(self.prev_prices[pp])


        #Set Min and Max values for normalization

        for p in range(len(self.close_prices)):
            if self.close_prices[m] != "Close":
                if (self.close_prices[p] > self.max_close):
                    self.max_close = self.close_prices[p]
            if (self.close_prices[p] < self.min_close):
                self.min_close = self.close_prices[p]
        for q in range(len(self.high_prices)):
            if (self.high_prices[q] > self.max_high):
                self.max_high = self.high_prices[q]
            if (self.high_prices[q] < self.min_high):
                self.min_high = self.high_prices[q]  

        for s in range(len(self.prev_prices)):
            if (self.prev_prices[s] > self.max_prev):
                self.max_prev = self.prev_prices[s]
            if (self.prev_prices[s] < self.min_prev):
                self.min_prev = self.prev_prices[s]
                
        for s in range(len(self.sentiments)):
            self.sentiments[s] = float(self.sentiments[s])
            if (self.max_sent > self.max_sent):
                self.max_sent = self.sentiments[s]
            if (self.sentiments[s] < self.min_sent):
                self.min_sent = self.sentiments[s]

    # Perform normalization calculation and set normalized inputs            
    def set_normalized_input(self):
        # Call set_input function in case it was not called already
        if (self.max_prev == 0):
            self.set_input()
            
        # Perform normalization calculation under the normalized_x = (x - min)/(max - min) model
        
        for i1 in range(len(self.close_prices)):
            self.normalized_close.append((self.close_prices[i1] - self.min_close)/(self.max_close - self.min_close))

        for i2 in range(len(self.high_prices)):
            self.normalized_high.append((self.high_prices[i2] - self.min_high)/(self.max_high - self.min_high))


        for i4 in range(len(self.prev_prices)):
            self.normalized_prev.append((self.prev_prices[i4] - self.min_prev)/(self.max_prev - self.min_prev))
        
        
            
        for i5 in range(len(self.sentiments)):
            diff = self.max_sent - self.min_sent
            if diff == 0:
                self.normalized_sent.append(0)
            else:
                self.normalized_sent.append((self.sentiments[i5] - self.min_sent)/(self.max_sent - self.min_sent))
     
    # Organize the input into a zipped list
    def get_input(self):
        return (list(zip(self.close_prices,self.high_prices,self.prev_prices,self.sentiments)))
    # Organize the normalized input into a zipped list
    def get_nomralized_input(self):
        return (list(zip(self.normalized_close,self.normalized_high,self.normalized_prev,self.sentiments)))

    # Set the output data
    def set_output(self):
        
        # Open and read the output file and append the list
       
        with open(self.stock + '.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
            for row in readCSV:
                self.open_prices.append(row[2])
                
        # Remove the first two rows (header and first data point)
        self.open_prices = self.open_prices[2:]

        #
        for m in range(len(self.open_prices)):
            self.open_prices[m] = float(self.open_prices[m])

        for i in range(len(self.open_prices)):
            if (self.open_prices[i] > self.max_open):
                self.max_open = self.open_prices[i]
            if (self.open_prices[i] < self.min_open):
                self.min_open = self.open_prices[i]


    #uses min max function
    def set_normalized_output(self):
        self.set_output()
        for i1 in range(len(self.open_prices)):
            self.normalized_open.append((self.open_prices[i1] - self.min_open)/(self.max_open - self.min_open))
    #returns open_prices
    def get_output(self):
        return (self.open_prices)
    #gets the normalized output
    def get_normalized_output(self):
        return (self.normalized_open)
    #inverse function to get predicted values into actual values
    def inverse(self,normalized):
        return ((normalized * (self.max_open - self.min_open)) + self.min_open)
    #retuns what the user input
    def get_training_input(self):
        self.set_training_input()
        return self.training_inputs
    
    #sets puts all of the data into a list as a tuple
    def set_training_input(self):
        for i in range(len(self.normalized_close)):           
            temp_list = [self.normalized_close[i],self.normalized_high[i],self.normalized_prev[i],self.normalized_sent[i]]
            self.inputs.append(temp_list)
        train_end = int(.7*len(self.inputs))
        self.training_inputs = self.inputs[0:train_end]

    def get_testing_input(self):
        self.set_testing_input()
        return self.testing_inputs

    def get_training_output(self):
        self.set_training_output()
        return self.training_outputs
    
    def set_testing_input(self):
        train_end = int(.7*len(self.inputs))
        self.testing_inputs = self.inputs[train_end:]
       
    def set_training_output(self):
        train_end = int(.7*len(self.normalized_open))
        self.training_outputs = self.normalized_open[0:train_end]
           
    def get_testing_output(self):
        self.set_testing_output()
        return self.testing_outputs
    def set_testing_output(self):
        train_end = int(.7*len(self.normalized_open))
        self.testing_outputs = self.normalized_open[train_end:]
    
    def clear_lists(self):
        #everything is reinitialized 
        self.close_prices.clear()
        self.high_prices.clear()
        self.prev_prices.clear()
        self.normalized_close.clear()
        self.normalized_high.clear()
        self.normalized_prev.clear()
        self.open_prices.clear()
        self.normalized_open.clear()
        self.inputs.clear()
        self.training_inputs.clear()
        self.testing_inputs.clear()
        self.training_outputs.clear()
        self.testing_outputs.clear()
        self.sentiments.clear()
        self.normalized_sent = []
        self.max_sent = 0.0
        self.min_sent = 0.0
        self.min_close = 1000
        self.max_close = 0  
        self.min_high = 1000
        self.max_high = 0
        self.min_prev = 1000
        self.max_prev = 0
        self.min_open= 1000
        self.max_open = 0