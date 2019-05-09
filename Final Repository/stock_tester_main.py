# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:09:01 2019

@author: kyleo
"""

from Neural_Network_class import NeuralNetwork
from Sentiment_class import Sentiment
from CSV_normalize_class import CSV_Normalize
from CSV_creator_class import csv_creator
from numpy import exp, array, random, dot


""" This is the file that calls all of the classes responsible for making stock predictions on a list of stocks
that should be kept in the same folder called sp500.txt This file makes 5 predictions of the first 5 stocks that have not been predicted
in your sp500.txt, after each of which the prediction is stored in a txt file that can be quickly read by the gui (the gui is kept in the gui_run.py file). 
This file also updates two other files. The DataFound.txt and stat_file. The datafound.txt needs to be cleared if you want to rerun stocks on it, but the 
stat_file just keeps track of the ticker symbol you ran along with its current value at the time and its predicted movement. 
"""



"""This is used to see how accuratly we predicted the movements of the stock market"""
def correctness(predict, actual):
    length = len(predict)
    i = 0
    correct = 0
    #checks to see if the prediction was the same as the actual movement of the stock
    while (i < length):
        if actual[i] == predict[i]:
            correct = correct + 1
        i = i + 1
    return(correct / i)

def main(): 
    tickerSymbols = open("sp500.txt", "r")    #used to get info from the list of stocks we compiled
    stat_file = open("stat_file.txt", "a")    # stat file so that we can make metrics after
    index = 0 #counts the amount of times our apis are called
    for tick in tickerSymbols: # goes trough a list of ticker symbols
        contents  = tick.split() #splits the ticker symbol from company name
        ticker = contents[0]   # ticker was stored first and it gets saved here
        regex = "+" + ticker + "+" # created a expression that can be easily to checked to see if we ran a ticker symbol
        if regex not in open("DataFound.txt", "r").read() and index < 5: # makes sure we didnt predict a stock and then allows for 5 iterations
                string = ""
                index = index + 1
                stock = ""
                count = 0
                
                while count < len(contents)-1: # puts the rest of the line that we read as the company name
                    count = count + 1
                    stock = stock + " " + contents[count]
                    stock = stock.strip()
                
                #updates the user to which stock is being predicted
                
 
                sent = Sentiment(stock,ticker) #calls the sentiment class
                sentiments = sent.cleaned_data #stores the sentiment dates and values
                articles = sent.full_articles #stores full article names and descriptions
                
                # makes sure that the csv creator works and does not throw errors
                f1 = csv_creator(ticker, sentiments, '12-10-2010') 
                if f1.verify  == True: 
                    
                    print("Company Name: " + stock)
                    print("Ticker Symbol: " + ticker)
                    
                    #opens a txt file so that we can store all of the information
                    info_file = open(ticker + ".txt", "a")
                    info_file.write("Company Name: " + stock + " \n") 
                    info_file.write("Ticker Symbol: " + ticker  + " \n")
                    i = 0
                    
                    #writes the first 5 articles that were returned by the new york times to the text file
                    while i < len(articles):
                        if i < 5:
                            #print("sentiment" + str(i+1) + ": " + articles[i].strip() + " \n")
                            #print("---")
                            
                            info_file.write("sentiment" + str(i+1) + ": " + articles[i].strip() + " \n")
                            info_file.write("---")
                            
                        i = i + 1
                    
                    # we run our csv normalizer
                    msft = CSV_Normalize()
                    msft.set_stock(ticker)
                    msft.set_normalized_input()
                    msft.set_normalized_output()
                    training_input = msft.get_training_input()
                    test_input = msft.get_testing_input()
                    training_output = [msft.get_training_output()]
                    test_output = msft.get_testing_output()
                    
                    # running neural network
                    neural_network = NeuralNetwork()
                    neural_network.train(array(training_input), array(training_output).T, 50)
                    
                
                    results = [] #stores predicted output of neural network normalized
                    amount = [] #stores the indexing of the array so that it can be plotted
                    actual = [] #stores the actual values used
                        
                    #stores the results from nerual network 
                    for n in range(len(test_output)):
                        results.append(neural_network.get_output(array(test_input[n])))
                        amount.append(n) # keeps the indexes to plot the data
                        
                    results_regular = [] #used to keep the non normalized values
                    
                    for i in range(len(results)):
                        results_regular.append(msft.inverse(results[i])) #calls the math function to get the actual value of our prediction
                    for o in range(len(test_output)):
                        actual.append(msft.inverse(test_output[o]))  #calls the math function to get the actual value of the stock
                    
                    
                    count = 0
                    temp = 0.0
                    tester = []
                    checker = []
                            
                    # this code block checks to see if we predicted the stock to go up or down
                    for i in range(len(results_regular)):
                        if count > len(results_regular)*.8: #only wanted to check after the testing data has passed
                            temp = results_regular[i-1] 
                            if temp < results_regular[i]:
                                temp2 = "Increase"
                                tester.append(temp2)
                            else:
                                temp2 = "Decrease"
                                tester.append(temp2)
                        if count ==len(results_regular)-1:
                            last_value_predicted = temp2 # stores the last value as the prediction
                        count = count + 1            
                                 
                    count = 0
                    
                     # this code block checks to see if the stock moved up or down in price
                    for i in range(len(actual)):
                        if count > len(actual)*.8: #only wanted to check after the testing data has passed
                            temp = actual[i-1]
                            if temp < actual[i]:
                                checker.append("Increase")
                            else:
                                checker.append("Decrease")
                        count = count + 1  
                    
                    # function to see how often we were correct. returns float 
                    val = correctness(tester, checker)
                    
                    
                    #stores important values
                    info_file.write("\nCurrent Value: " + str(actual[count-1]) + " \n")#last value because count goes past the last index of the array
                    info_file.write("Prediction: " + last_value_predicted + " \n")# prediction
                    info_file.write("Perecent_correct: "+ str(val) + " \n") # movements we predicted correctly
                    info_file.write("Predicted Values: \n ") # begins the spot for all of  our predicted values
                    
                    #puts all of the predicted values into a text file 
                    for i in results_regular:
                        info_file.write(str(i))
                    
                    # this is where we put the actual values that the stock market reported into our text file

                    info_file.write("\n Actual Values: \n ")
                    for i in actual:
                        info_file.write("[" + str(i) + "] " ) # we include brackets so that numbers can be distinguished between
                    
                    info_file.write("\n ---------")
                    info_file.close()
                    msft.clear_lists()
                    
                    string = ticker + " " + str(actual[count-1]) + " " + last_value_predicted + " \n" #writes the results to a stat_file so we can make metrics later
                    stat_file.write(string)
                    
                    # free up memory and delete arrays that would be appended to in the next iteration
                    del(msft)
                    del(neural_network)
                    del(training_input)
                    del(test_input)
                    del(training_output)
                    del(test_output)
                    
                    File = open("DataFound.txt", "a")
                    File.write("+" + ticker.strip() + "+\n")
                    File.close()
                    
                    index = index + 1 # we now look at the next stock unless this is bigger than 5 so we dont overuse our api
                else: # there were problems with the csv file
                    print("There was a problem")
    #close the files so that they do not get corrupted
    tickerSymbols.close()
    stat_file.close()

    
main()
