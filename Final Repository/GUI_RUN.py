# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:53:39 2019

@author: kyleo
"""

from tkinter import *
from tkinter.tix import *

from Neural_Network_class import NeuralNetwork
from Sentiment_class import Sentiment
from CSV_normalize_class import CSV_Normalize
from CSV_creator_class import csv_creator
from numpy import exp, array, random, dot
import requests

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


"""This is the class that is incharge of displaying the prediction along with article news and other metrics.
The main method in this class is the run function. Once the user selects a stock from navigting the tkinter gui
it gets sent to run. Run checks to see if we already have the stock information in memory and if we do it reads the text file with the information in it.
If we do not have the stock in memory, then the run function is responsible for calling all of classes responsible for a stock prediction, and displaying the 
information onto the screen. 
"""

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
           
            
    
class Engine:
    def __init__(self, master):
        
        #creates the master window
        self.master = master
        #names the window 
        self.master.title("Muhlenberg Stock Predictor")
        
        #used to store the name of the stock being used
        self.stock = "" #used to 
        
        # makes a canvas the size of the window but with room at the bottom so there is no screen run off
        self.canvas = Canvas(self.master, width=self.master.winfo_screenwidth(), height= self.master.winfo_screenheight()-20, border = 0)
        self.canvas.pack()
        
        #creates a scroll bar. The scroll bar can only be used if the contents run off the screen
        swin = ScrolledWindow(self.canvas, width=self.master.winfo_screenwidth(), height= self.master.winfo_screenheight()-20)
        swin.pack()
        self.win = swin.window
        
        
        #sets canvases to store diffrent section od information. They allow for their own padding and are resizeable
        self.master1 = Canvas(self.win, width=700, height=700, border = 0)
        self.master1.pack()
        
        self.master2 = Canvas(self.win, width=700, height=700, border = 0)
        self.master2.pack()
        
        self.master3 = Canvas(self.win, width=700, height=700, border = 0)
        self.master3.pack()
        
        self.master4 = Canvas(self.win, width=700, height=700, border = 0)
        self.master4.pack()
        
        self.master5 = Frame(self.win, width=700, height=700, border = 0)
        self.master5.pack()
        
        #asks user to input a stock
        self.label1 = Label(self.master1, text= " \n What stock do you want us to predict? \n " , font=("Times New Roman", 20))
        self.label1.pack()
        
        #textfield for a user to type in
        self.textField = Entry(self.master1, width = 50, font = "Helvetica")
        self.textField.pack()
        
        
        #self.b1 = Button(self.master, text="GO!!!", command=self.quarry, activebackground = "crimson")
        self.b1 = Button(self.master1, text="GO!!!",command = self.stock_lookup, activebackground = "crimson")
        self.b1.pack()
        
        #initiated but not packed until after the button is clicked
        self.label = Label(self.master2, text= "Top Results", height = 2)

        #initializes all of the buttons that the user can click on to select a stock
        self.Result1 = Button(self.master2, text= "", border = 0, command = self.callback1)
        self.Result2 = Button(self.master2, text= "", border = 0, command = self.callback2)
        self.Result3 = Button(self.master2, text= "", border = 0, command = self.callback3)
        self.Result4 = Button(self.master2, text= "", border = 0, command = self.callback4)
        self.Result5 = Button(self.master2, text= "", border = 0, command = self.callback5)
        
        # this button is used if there is an invalid return by yahoo
        self.faile = Label(self.master2, text= "", border = 0)
        
        #other imprtant labels ofr display
        self.verification = Label(self.master3, text= "", border = 0)
        self.prediction = Label(self.master3, text= "", border = 0)
        self.findings = Label(self.master3, text= "", border = 0)
        self.labelI = Label(self.master4, text="",   border = 0)
        
        #store the newspaper article information 
        self.sentiment1 = Label(self.master4, text= "", border = 0)
        self.sentiment2 = Label(self.master4, text= "", border = 0)
        self.sentiment3 = Label(self.master4, text= "", border = 0)
        self.sentiment4 = Label(self.master4, text= "", border = 0)
        
        
        
    def clearLabels(self):
         """all of these statements check to see if the display object was packed and if it was,
         the object gets destroyed to make room for the next stocks results"""

         if self.label.winfo_exists():
             self.label.destroy()
         
         if self.labelI.winfo_exists():
             self.labelI.destroy()   
            
         if self.Result1.winfo_exists() :
             self.Result1.destroy()
             
         if self.Result2.winfo_exists() :
            self.Result2.destroy()
            
         if self.Result3.winfo_exists() :
             self.Result3.destroy()
             
         if self.Result4.winfo_exists() :
            self.Result4.destroy()
            
         if self.Result5.winfo_exists() :   
            self.Result5.destroy()
         
         if self.faile.winfo_exists():
             self.faile.destroy()
             
         if self.findings.winfo_exists():
             self.findings.destroy()   
                          
         if self.verification.winfo_exists():
             self.verification.destroy()
        
         if self.prediction.winfo_exists():
             self.prediction.destroy()
             
         if self.sentiment1.winfo_exists():
             self.sentiment1.destroy()
             
         if self.sentiment2.winfo_exists():
             self.sentiment2.destroy()
         
         if self.sentiment3.winfo_exists():
             self.sentiment3.destroy()
             
         if self.sentiment4.winfo_exists():
             self.sentiment4.destroy()
             
         if self.master5.winfo_exists():
             self.master5.destroy()
    
    """"callbacks allow for the gui to know which button was clicked
    we then take the text at the button and send it to the run methhod"""
    def callback1(self):
        temp = self.Result1['text']
        self.clearLabels()
        self.run(temp)
        
    def callback2(self):
        temp = self.Result2['text']
        self.clearLabels
        self.run(temp)
    
    def callback3(self):
        temp = self.Result3['text']
        self.clearLabels
        self.run(temp)
        
    
    def callback4(self):
        temp = self.Result4['text']
        self.clearLabels
        self.run(temp)
        
    def callback5(self):
        temp = self.Result5['text']
        self.clearLabels
        self.run(temp)
        
        
    def ticker_company(self, info):
        stock = info.split("--")
        name = stock[0][15:].strip()
        ticker = stock[1][18:].strip()
        ar = []
        ar.append(name)
        ar.append(ticker)
        return (ar)
    
    def run_file(self, info):
        f = open(info, "r")
        stock = f.readline()
        stock = stock[0:-2]
        ticker = f.readline()
        stud = f.read()
        sentis = stud.split("---")
        
        #current_val is my reused regular expression it parses the file and allows for everything to 
        #be saved and put on the screen
        
        current_val = re.compile(r'Current Value: .* \n')
        current_vals = current_val.findall(stud)
        stud = re.sub(current_val, "", stud)
        
        value_of_stock = current_vals[0] #saves current value
        
        current_val = re.compile(r'Prediction: .* \n')
        current_vals = current_val.findall(stud)
        stud = re.sub(current_val, "", stud)
        prediction = current_vals[0] #saves prediction
        
        current_val = re.compile(r'Perecent_correct: .* \n')
        current_vals = current_val.findall(stud)
        stud = re.sub(current_val, "", stud)
        percent_correct = current_vals[0] #saves the amount the network predicted corrrectly
        
        current_val = re.compile(r'sentiment: .* \n')
        current_vals = current_val.findall(stud)
        stud = re.sub(current_val, "", stud)
        
        
        cleaned_predicted = [] #used to store predicted vals
        cleaned_actual = [] #used to store actual values
        amount = [] #used to store the index
        
        #This block simply prints the values that we got from our regular expressions to the screen
        string = "This is what we concluded about " + stock + "."
        self.findings = Label(self.master3, text= string , border = 2, font=("Times New Roman", 14))
        self.findings.pack()
                    
        self.verification = Label(self.master3, text= "", border = 0, font=("Times New Roman", 14))
        self.verification.pack()
        val = float(percent_correct[19:-2].strip()) * 100  #formats the percent correct to 2 decimal places
        val = float("{0:.2f}".format(val))
        self.verification['text'] = "We have been predicting movements in this stock with a " + str(val) + "% correctness"
                                  
        self.prediction = Label(self.master3, text= "", border = 0, font=("Times New Roman", 14))
        self.prediction.pack()
                      
        if "Increase" in prediction: 
            self.prediction['text'] = "The stock is predicted to go on an upward trend!! \n \n"
        else:
            self.prediction['text'] = "The stock is predicted to go on an downward trend \n \n"
        
        i = 0 #used to index the sentiments and print them to the screen
        for s in sentis:
            if "sentiment" in s:# makes sure the line has the token sentiment in it
                    #puts the sentiments on the scrren in order that they came in
                    #does not allow for labels to have no text inside of them
                    if i == 0: 
                        string = " \n Here are some articles about " + stock + " \n "
                        self.labelI = Label(self.master4, text=string,   border = 0, font=("Times New Roman", 16))
                        self.labelI.pack()
                        
                        self.sentiment1 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14))
                        self.sentiment1.pack()
                        self.sentiment1['text'] = sentis[i][12:-2] + "\n"
                    if i == 1: 
                        self.sentiment2 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14))
                        self.sentiment2.pack()
                        self.sentiment2['text'] = sentis[i][12:-2] + "\n"
                    if i == 2: 
                        self.sentiment3 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14))
                        self.sentiment3.pack()
                        self.sentiment3['text'] = sentis[i] [12:-2]  +"\n"
                    if i == 3: 
                        self.sentiment4 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14 ))
                        self.sentiment4.pack()
                        self.sentiment4['text'] = sentis[i][12:-2] + "\n"
                    i = i + 1
            else: 
                a = s.split(":") # splits on : to seperate actual and predicted values
                for val in range(len(a)): # goes through the first part of the split
                    if "Actual Values" in a[val]: #makes sure we are at the correct information(predicted values)
                        temp = a[val].replace("Actual Values", "")#gets rid of extra text at the end 
                        temp = temp.strip() #strips white space
                        actual_vals = temp.split("]") #seperates all of the numbers into a array
                        for num in range(len(actual_vals)-2):
                                amount.append(num) #indexes is kept for the plot
                                cleaned_predicted.append(float(actual_vals[num][1:])) #we make the numbers a float
                        temp2 = (a[val+1].strip()) # used to get the index of a that stores the actual values
                        actual_vals = temp2.split("] ") #seperates the numbers into an array
                        for num in range(len(actual_vals)-1):
                                cleaned_actual.append(float(actual_vals[num][1:])) #puts all of the numbers as floats in the array
                 
        #plots all of the points into a python chart
        plt.clf() #clears the plot and makes the chart appear
        plt.plot(amount, cleaned_predicted, label = "predicted") 
        plt.plot(amount, cleaned_actual, label = 'actual')
        plt.legend() #displays legend
        plt.show() #shows graph
        
 
    def run(self, stock):
            
            
            self.label.destroy()# destroys remaning labels from the last stock
            
            info = self.ticker_company(stock) # calls a function that gets the company name and ticker symbol as an array
            stock = info[0]  #stores the company name
            ticker = info[1]  #stores the company's ticker symbol
            
            #this block is used to see if the selected stock is in our library of preprocessed stocks
            file = ticker.strip() + ".txt"   #makes sure the ticker does not have any white space and then makes a file name
            print(ticker)
            exists = os.path.isfile(file)   #returns true if the file is in our system
            if exists: #this is used to see if the stock is already in our system
                self.run_file(file) #sends the preprocessed file to be read by a part of the gui
               
            else: # we have to run the neural network and call apis
                sent = Sentiment(stock,ticker) 
                sentiments = sent.cleaned_data # gets sentiment values with dates
                c = csv_creator(ticker, sentiments, '12-10-2010') #creates csv to be used. if it cant be created then the program prints error
                if c.verify: #checks to make sure there was no problem with running the api
                    
                    sentis = sent.full_articles # full articles to be added
                    #if their is a article in the array full articles then it get appended to a button
                    for i in range(len(sentis)):
                        if i == 0: 
                            string = " \n Here are some articles about " + stock + ". \n "
                            self.labelI = Label(self.master4, text=string,   border = 0, font=("Times New Roman", 16))
                            self.labelI.pack()
                            
                            self.sentiment1 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14))
                            self.sentiment1.pack()
                            self.sentiment1['text'] = sentis[i]
                        if i == 1: 
                            self.sentiment2 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14))
                            self.sentiment2.pack()
                            self.sentiment2['text'] = sentis[i]
                        if i == 2: 
                            self.sentiment3 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14))
                            self.sentiment3.pack()
                            self.sentiment3['text'] = sentis[i]   
                        if i == 3: 
                            self.sentiment4 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14 ))
                            self.sentiment4.pack()
                            self.sentiment4['text'] = sentis[i]
                    
                    # here is where the data gets normalized to be passed into the neural network
                    msft = CSV_Normalize()
                    msft.set_stock(ticker)
                    msft.set_normalized_input()
                    msft.set_normalized_output()
                    training_input = msft.get_training_input()
                    test_input = msft.get_testing_input()
                    training_output = [msft.get_training_output()]
                    test_output = msft.get_testing_output()
                    
                    
                    neural_network = NeuralNetwork() #neural network is initialized
                    neural_network.train(array(training_input), array(training_output).T, 100) #neural network is trained using data from csv normalize
                    
                    results = [] #stores predicted output of neural network normalized
                    amount = [] #stores the indexing of the array so that it can be plotted
                    actual = [] #stores the actual values used
                    
                    #stores the results from nerual network 
                    for n in range(len(test_output)):
                        results.append(neural_network.get_output(array(test_input[n])))
                        amount.append(n) # keeps the indexes so that the data can be plotted
                
                    results_regular = [] #used to keep the non normalized values
                    
                
                    for i in range(len(results)):
                        results_regular.append(msft.inverse(results[i])) #calls the math function to get the actual value of our prediction
                    for o in range(len(test_output)):
                        actual.append(msft.inverse(test_output[o]))  #calls the math function to get the actual value of the stock
                    
                    msft.clear_lists() #arrays and min max values are put back to their initialized values for future iterations
                    
                    
                    plt.clf() #makes the plot appear and deletes all current previous contents
                    #the points are plotted
                    plt.plot(amount, results_regular, label = 'predicted')
                    plt.plot(amount, actual, label = 'actual')
                    plt.legend() #legend is included
                    plt.show() #show the updates
                    
                    
                    count = 0
                    temp = 0.0
                    tester = [] #stores if a stock was predicted to go up or down
                    checker = [] #stores if the stock went up or down
                    
                    #used to see if the predicted value increased or decreased
                    for i in range(len(results_regular)):
                        if count > len(results_regular)*.8: #only checks after training
                        
                            temp = results_regular[i-1] 
                            if temp < results_regular[i]:
                                tester.append("Increase")
                            else:
                                tester.append("Decrease")
                          
                        count = count + 1
                        if count ==len(results_regular):
                            last_value_predicted = results_regular[i] 
                            
                    count = 0
                    #used to see if the stock actually increased in value
                    for i in range(len(actual)):
                        if count > len(actual)*.8: #only checks after training
                            temp = actual[i-1]
                            if temp < actual[i]:
                                checker.append("Increase")
                            else:
                                checker.append("Decrease")
                       
                        count = count + 1
                        if count ==len(actual):
                            last_value_actual = actual[i]
                       
                    val = correctness(tester, checker) # sends the two arrays we created to see how we predicted the movemnts
                    
                    string = "This is what we concluded about " + stock + "."
                    self.findings = Label(self.master3, text= string , border = 2, font=("Times New Roman", 14))
                    self.findings.pack()
                    
                    self.verification = Label(self.master3, text= "", border = 0, font=("Times New Roman", 14))
                    self.verification.pack()
                    val = val * 100
                    val = float("{0:.2f}".format(val)) #formats the values
                    self.verification['text'] = "We have been predicting movements in this stock with a " + str(val) + "% correctness"
                    
                    last_value_predicted = last_value_predicted[0]
                    
                    
                    self.prediction = Label(self.master3, text= "", border = 0, font=("Times New Roman", 14))
                    self.prediction.pack()
                      
                    if last_value_predicted > last_value_actual:
                        self.prediction['text'] = "The stock is predicted to go on an upward trend! \n \n"
                    elif last_value_predicted < last_value_actual:
                        self.prediction['text'] = "The stock is predicted to go on an downward trend! \n \n"
                    else:
                        self.prediction['text'] = "We have no advice on the movement of this stock  \n \n "
                        
                    #frees memory
                    del(msft)
                    del(neural_network)
                    del(training_input)
                    del(test_input)
                    del(training_output)
                    del(test_output)
                    
    def stock_lookup(self):
        
        self.clearLabels() # makes sure all of the labels and buttons have been cleared
        
        name = self.textField.get()
        self.label = Label(self.master, text= "Top Results \n ", height = 2)
        self.label.pack()
        
        #calls the yahoo website api that returns the top results for stocks depending on what the user had in the search bar
        url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query='+ name + '&region=1&lang=en&callback=YAHOO.Finance.SymbolSuggest.ssCallback'
        data = requests.get(url)
        data = data.text
        
        
        company = re.compile(r'"name":"[^"]+"') # pattern to get name of company
        companys = company.findall(data)        # finds all of the company's names
        
        ticker = re.compile(r'"symbol":"[^"]+"')
        tickers = ticker.findall(data) 
    
        
        company_info = []
        company_info.append(tuple((0,0))) #used to make sure that this was an array so that errors did not get thrown
        
        #puts all of the results into anarray
        for i in range(len(companys)):      
            company_info.append((tuple(((companys[i][8:-1]),tickers[i][10:-1]))))
        
        """puts the first result into a button and makes sure their is a first result otherwise the button will not appear
        This same process is done until all of the buttons are initialized"""
        if len(company_info) > 1:
            self.Result1 = Button(self.master2, text= "", border = 0, command = self.callback1)
            self.Result1.pack()
            self.Result1['text'] = "Company name = " + company_info[1][0] + " --  Ticker Symbol = " + company_info[1][1]
        
        if len(company_info) >= 2:
            self.Result2 = Button(self.master2, text= "", border = 0, command = self.callback2)
            self.Result2.pack()
            self.Result2['text'] = "Company name = " + company_info[2][0] + " --  Ticker Symbol = " + company_info[2][1]
            
        if len(company_info) >= 3:
            self.Result3 = Button(self.master2, text= "", border = 0, command = self.callback3)
            self.Result3.pack()
            self.Result3['text'] = "Company name = " + company_info[3][0] + " --  Ticker Symbol = " + company_info[3][1]
            
        if len(company_info) >= 4:
            self.Result4 = Button(self.master2, text= "", border = 0, command = self.callback4)
            self.Result4.pack()
            self.Result4['text'] = "Company name = " + company_info[4][0] + " --  Ticker Symbol = " + company_info[4][1]
            
        if len(company_info) >= 5:
            self.Result5 = Button(self.master2, text= "", border = 0, command = self.callback5)
            self.Result5.pack()
            self.Result5['text'] = "Company name = " + company_info[5][0] + " --  Ticker Symbol = " + company_info[5][1]
        if len(company_info) == 1:
            self.faile = Label(self.master2, text= "", border = 0)
            self.faile.pack()
            self.faile['text'] = "Sorry their were no results for that search."

    #used to get the amount that was predicted correctly
    def correctness(predict, actual):
        correct = 0
        for i in range(len(predict)):
            if actual[i] == predict[i]: #sees if the predicted movement was the same as the predicted guess
                correct = correct + 1
        return(correct / len(predict))

def main():   
    #makes a tk window
    root = Tk()
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight())) #makes the window the size of the screen
    gui = Engine(root) #calls the gui
    root.mainloop() #loops until the user closes the tkinter windown
main()