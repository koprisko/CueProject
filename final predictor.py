# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:53:39 2019

@author: kyleo
"""

#from tkinter import Tk, Label, Button, Entry, LEFT, RIGHT, BOTTOM, Frame, Checkbutton, Canvas, PhotoImage
#from PIL import ImageTk

from tkinter import *
from tkinter.tix import *

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


import numpy as np
import time
import calendar as cal
import pandas as pd
import datetime as dt
import re
import requests
try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

import csv
import statistics

from numpy import exp, array, random, dot
#import LoadCSV_V2
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import datetime
import os

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))


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
    

class Sentiment:
    def __init__(self, stock, ticker):
        
        #this block initializes the url syntax nexessary for making a call to the New york times api
        key = "Q41EjWTOsr2VA3y4UaXDqMmpWg6aNbDr"
        info = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q=" + stock
        info = info + "&subject:Stocks and Bonds&api-key=" + key
        r = requests.get(info)
        r.encoding = 'utf-8'
        
        self.full_articles = [] #used to return articles to the main class
               
        data = self.collector(r.text)  #gets returned the sentiment score and data
        new_dates = self.fix_year(data) # dates get properly formatted
        
        self.cleaned_data = []
        count = 0
        for i in data:
            self.cleaned_data.append(tuple((new_dates[count],i[1])))
            count = count + 1
        
        r = requests.get("https://feeds.finance.yahoo.com/rss/2.0/headline?s=" + ticker + "&region=US&lang=en-US")
        r.encoding = 'utf-8'
        info2 = r.text
        data2 = self.yahooCollector(info2) 
        new_dates = self.fix_year2(data2) # dates get properly formatted
        count = 0
        
        for i in data2:
            self.cleaned_data.append(tuple((new_dates[count],i[1])))
            count = count + 1
        
        url = "https://stocknewsapi.com/api/v1?tickers=" + ticker + "&items=30&fallback=true&token=oroav5z0e7ov2ohmszaggk4a9pqutz3gacvvjfvo"        
        response = requests.get(url)
        response.encoding = 'utf-8'
        response = response.json()
            
        
        if 'data' in response:
            for r in response['data']:
                article = r['title'] + " " + r['text']                
                date = r['date'][4:16].strip()
                date_string = ""
                dt = datetime.datetime.strptime(date, "%d %b %Y")
                date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year)
                self.cleaned_data.append(tuple((date_string, self.classifier(article))))
        
        self.total_percent = 0
        
        count = 0 
        if data:
            for d in data: 
                #if d != 0:
                count = count + 1
                self.total_percent = self.total_percent + self.check_polar(d[1])
            
        if data2:
            for d in data2: 
                #if d != 0:
                   count = count + 1
                   self.total_percent = self.total_percent + self.check_polar(d[1])
            
        self.total_percent = self.total_percent / count
        #print("total_positive_sentiment from " + str(count) + " articles = " + str(self.total_percent))
        
        
    def fix_year(self, date):
        holder = []
        for d in date:
            dt = datetime.datetime.strptime(d[0], "%Y-%m-%d")
            date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year)
            holder.append(date_string)
        return holder   
    
    def fix_year2(self, date):
        holder = []
        for d in date:
            temp = d[0].strip()
            dt = datetime.datetime.strptime(temp, "%d %b %Y")
            date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year)
            holder.append(date_string)
        return holder  
        
        
    def clean_yahoo_Dates(self, info):
            heads = []
            for i in info:
                heads.append(i[13:-25])
            return heads
        
        
    def yahooCollector(self, data):
        #print(data)
        irrelevant = re.compile(r'<title>Yahoo! Finance:.* News</title>')
        data = re.sub(irrelevant, "", data)
        
        irrelevant = re.compile(r'<description>Latest Financial News for.*</description>')
        data = re.sub(irrelevant, "", data)
        
        title = re.compile(r'<title>.*</title>')
        titles = title.findall(data)
        
        description = re.compile(r'<description>.*</description>')
        descriptions = description.findall(data)
        
        date = re.compile(r'<pubDate>.*</pubDate>')
        dates = date.findall(data)
        
        dates = self.clean_yahoo_Dates(dates)
        
        #print(dates)
        good = True
        sentiment_array = []
        count = -1 # needed so that we can index the dates when a new title is found
        while good == True:
            polarity = 0
            #print("-------------------")
            if titles:
                count = count + 1
                temp = titles.pop().replace('<title>', "").replace('</title>', "")
                temp = self.clean_up(temp)
                temp = temp.replace("amp", "").replace("quot", "")
                Sa = self.classifier(temp)
                polarity = Sa + polarity
                
            elif descriptions:
                temp = descriptions.pop().replace('<description>', "").replace('</description>', "")
                temp = self.clean_up(temp)
                temp = temp.replace("amp", "").replace("quot", "")
                Sa = self.classifier(temp)
                polarity = polarity + Sa
            else:
                good = False
            
            if good == True:
                sentiment_array.append(tuple((dates[count],polarity)))
            
        return sentiment_array
              
    def clean_headlines(self, info):
        heads = []
        for i in info:
            heads.append(i[18:-3])
        return heads
    
            
    def clean_snippets(self, info):
        snips = []
        for i in info:
            snips.append(i[11:-1])
        return snips
    def clean_mains(self, info):
        mains = []
        for i in info:
            mains.append(i[8:-1])       
        return mains    
    def clean_paragraphs(self, info):
        pars = []
        for i in info:
            pars.append(i[18:-3])        
        return pars
            
    def clean_dates(self, info):
        dates = []
        for i in info:
            dates.append(i[12:])
        return dates
            
    def check_polar(self, pols):
        neg = -.25
        pos = .5
        if pols < pos and pols > neg:
            return 0
        if pols >= pos:
            return 1
        if pols <= neg:
            return -1
    
            
    def collector(self, stud): # stud has all of the reponse text from the http call
                                              
        head_Line = re.compile(r'"print_headline":"[^"]+","') #creates the regex to find headlines
        headers = head_Line.findall(stud)                   #puts all of the headlines into an array
        stud = re.sub(head_Line, "", stud)                  #takes out all of the headlines from the response text
        headers  = self.clean_headlines(headers)            #gets rid of the unnessary text in the headlines
        
        # the same process mentioned in the above code block is repeated for all of the important parts of the response text
        
        snips = re.compile(r'"snippet":"[^"]+"')
        snippets = snips.findall(stud)
        stud = re.sub(snips, "", stud)
        snippets = self.clean_snippets(snippets)
        
        main = re.compile(r'"main":"[^"]+"')
        main_articles = main.findall(stud)
        stud = re.sub(main, "", stud)
        main_articles = self.clean_mains(main_articles)
        
        paragraph = re.compile(r'"lead_paragraph":"[^"]+","')
        paragraphs = paragraph.findall(stud)
        stud = re.sub(paragraph, "", stud)
        paragraphs = self.clean_paragraphs(paragraphs)
                
        date = re.compile(r'"pub_date":"\d\d\d\d-\d\d-\d\d')
        dates = date.findall(stud)
        stud = re.sub(paragraph, "", stud)
        dates = self.clean_dates(dates)

        sentiment_array = [] # used to return sentiment and date values back to the init function
        count = 0 # used to index the arrays
        
        # this while loop sums the sentiment polarity for a full article (header, main_article, snippet... etc) then appends
        # the final value along with the date so that it can be attached to a csv file
        
        while main_articles: # checks to make sure the http request returned valid article info
            
            temp1 = main_articles.pop()  # stores the information to be compared later
            ma = self.classifier(temp1) # stores the sentiment value of the main_article
            polarity = ma #all of the results from the parts of the article are added to polarity just like in  this example
            
            if headers:       # checks to make sure there is a header
                temp2 = headers.pop()  #saves the article header
                he = self.classifier(temp2)  #gets polarity value
                polarity = polarity + he   # sums polarity value with other senitment values
                
                if temp1.strip() != temp2.strip() :   # checks to make sure the data for the gui is not repeated
                    self.full_articles.append(temp1 + " \n " + temp2 + " \n published on " + dates[count]) # appends sentiment information to be displayed by the gui
                else:
                    self.full_articles.append(temp1 + " \n published on: " + dates[count])  # used if the data was redundant
            
            # the same steps mentioned abvoved are repeated here (without adding these parts to the array that will be showed in the gui)
            if paragraphs:    
                pa = self.classifier(paragraphs.pop())
                polarity = polarity + pa
                
            if snippets:
                sn = self.classifier(snippets.pop())
                polarity = polarity + sn
                
            if count > len(dates):
                sentiment_array.append(tuple((dates[count], polarity))) # an array of tuples with dates and the polarity of the article on that day is created
            else:    
                sentiment_array.append(tuple((dates[count-1], polarity))) # an array of tuples with dates and the polarity of the article on that day is created
            count = count + 1 #increase index
        return(sentiment_array)   #returns tuple array
          
    def clean_up(self, info):
        
        tokenized_word=word_tokenize(info)
        filtered_sent= ""
        for w in tokenized_word:
            if w not in stop_words and w.isalpha():
                filtered_sent = filtered_sent + " " + w
        return(filtered_sent)
            
    def sentiment_analyzer_scores(self, sentence):
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(sentence)
        return(score['compound'])
     
    def classifier(self, info):
        #info = self.clean_up(info)
        sentiment = self.sentiment_analyzer_scores(info)
        #print(info + " " + str(sentiment))
        #return(self.check_polar((sentiment)))
        return(sentiment)
        
                  
class CSV_Normalize:
    import csv
    stock = ""

    close_prices = []
    high_prices = []

    prev_prices = []
    sentiments = []
    
    max_sent = 0.0
    min_sent = 0.0

    normalized_sent = []
    

    min_close = 1000
    max_close = 0  
    min_high = 1000
    max_high = 0

    min_prev = 1000
    max_prev = 0

    normalized_close = []
    normalized_high = []

    normalized_prev = []

    open_prices = []

    min_open= 1000
    max_open = 0

    normalized_open = []

    inputs = []
    training_inputs = []
    testing_inputs = []

    training_outputs = []
    testing_outputs = []

    def set_stock(self,stock):
        self.stock = stock
    def set_input(self):
        with open(self.stock + '.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
            for row in readCSV:
                self.close_prices.append(row[5])
                self.high_prices.append(row[3])
                self.prev_prices.append(row[2])
                self.sentiments.append(row[7])

        self.close_prices = self.close_prices[1:-1]
        self.high_prices = self.high_prices[1:-1]
        self.prev_prices = self.prev_prices[1:-1]
        self.sentiments = self.sentiments[1:-1]

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

    def set_normalized_input(self):
        self.set_input()
        for i1 in range(len(self.close_prices)):
            self.normalized_close.append((self.close_prices[i1] - self.min_close)/(self.max_close - self.min_close))

        for i2 in range(len(self.high_prices)):
            self.normalized_high.append((self.high_prices[i2] - self.min_high)/(self.max_high - self.min_high))

        for i4 in range(len(self.prev_prices)):
            self.normalized_prev.append((self.prev_prices[i4] - self.min_prev)/(self.max_prev - self.min_prev))

        for i5 in range(len(self.sentiments)):
            dif_sent = self.max_sent - self.min_sent
            if dif_sent != 0:
                self.normalized_sent.append((self.sentiments[i5] - self.min_sent)/(self.max_sent - self.min_sent))
        if dif_sent == 0: 
            self.normalized_sent.append(0)
            
    def get_input(self):
        return (list(zip(self.close_prices,self.high_prices,self.prev_prices,self.sentiments)))

    def get_nomralized_input(self):
        return (list(zip(self.normalized_close,self.normalized_high,self.normalized_prev,self.sentiments)))

    def set_output(self):
        with open(self.stock + '.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
            for row in readCSV:
                self.open_prices.append(row[2])
        self.open_prices = self.open_prices[2:]

        for m in range(len(self.open_prices)):
            self.open_prices[m] = float(self.open_prices[m])

        for i in range(len(self.open_prices)):
            if (self.open_prices[i] > self.max_open):
                self.max_open = self.open_prices[i]
            if (self.open_prices[i] < self.min_open):
                self.min_open = self.open_prices[i]

    def set_normalized_output(self):
        self.set_output()
        for i1 in range(len(self.open_prices)):
            self.normalized_open.append((self.open_prices[i1] - self.min_open)/(self.max_open - self.min_open))

    def get_output(self):
        return (self.open_prices)

    def get_normalized_output(self):
        return (self.normalized_open)

    def inverse(self,normalized):
        return ((normalized * (self.max_open - self.min_open)) + self.min_open)

    def get_training_input(self):
        self.set_training_input()
        return self.training_inputs

    def set_training_input(self):
        """
        print(len(self.normalized_close))
        print(len(self.normalized_high))
        print(len(self.normalized_prev))
        print(len(self.normalized_sent))
        """
        for i in range(len(self.normalized_close)):   
            if i < len(self.normalized_close) and i < len(self.normalized_high) and i < len(self.normalized_prev) and i < len(self.normalized_sent):
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

class csv_creator:
    def __init__(self, ticker, sentiments, cutoff): 
        key2=  "52O3GXSWQOEBGMOT"
        key=  "0F9TBPXWF5YV5392"
        key3 =   "S8SGZ63ZVTYFOKV0"
        
        if self.getJson(ticker,key, sentiments,cutoff) == 'false' :
            if self.getJson(ticker,key2, sentiments,cutoff) == 'false' :
                if self.getJson(ticker,key3, sentiments,cutoff) == 'false' :
                    print("no luck")
        
    def createCSV(self, symb, TUPLES):
        file = symb + ".csv"
        with open(file, 'w', newline='') as csvFile:
              writer = csv.writer(csvFile)
              writer.writerows(TUPLES)    
              print("created " + symb)
              
              File = open("DataFound.txt", "a")
              File.write("+" + symb + "+\n")
              
    def getJson(self,symbol,key, sentiments,cutoff_date):            
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+ symbol + '&outputsize=full&apikey=' + key
            data = requests.get(url)
            data.encoding = 'utf-8'
            data = data.json()
            #print(data)
            tups = []
            dates = []
            date_list = []
            
            if 'Time Series (Daily)' in data:
                for i in data['Time Series (Daily)']:
                    date_string = ""
                    dt = datetime.datetime.strptime(i, "%Y-%m-%d")
                    date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year)
                    dates.append(i)
                    date_list.append(date_string)
                    
                #File = open("DataFound.txt", "a")
                #File.write("+" + symbol + "+\n")                  
                count = 0
                a = dt.strptime(cutoff_date, "%m-%d-%Y")                
                for i in dates:
                    b = dt.strptime(i, "%Y-%m-%d")
                    if b > a :
                        date_polarity = 0.02
                        date_string = ""
                        dt = datetime.datetime.strptime(i, "%Y-%m-%d")
                        date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year)
                        
                        for j in sentiments:
                            if date_string == j[0] :
                                date_polarity = date_polarity + j[1]                                  
                        opens = data['Time Series (Daily)'][i]['1. open']
                        highs = data['Time Series (Daily)'][i]['2. high']
                        lows = data['Time Series (Daily)'][i]['3. low']
                        close = data['Time Series (Daily)'][i]['4. close']
                        volume = data['Time Series (Daily)'][i]['5. volume']
                        tups.append(tuple((count, date_list[count], opens, highs, lows, close, volume, float(date_polarity)))) 
                        count = count + 1
                tups.reverse()
                self.createCSV(symbol, tups)
                return True
            
            else:
               print("error with ticker symbol " + symbol)
               return False
               

def correctness(predict, actual):
    length = len(predict)
    i = 0
    correct = 0
    while (i < length):
        if actual[i] == predict[i]:
            correct = correct + 1
        i = i + 1
    return(correct / i)
           
            
            

class Engine:
    def __init__(self, master):

        self.master = master
        self.master.title("Muhlenberg Stock Predictor")
        
        self.stock = ""
        
        self.canvas = Canvas(self.master, width=self.master.winfo_screenwidth(), height= self.master.winfo_screenheight()-20, border = 0)
        self.canvas.pack()
        
        swin = ScrolledWindow(self.canvas, width=self.master.winfo_screenwidth(), height= self.master.winfo_screenheight()-20)
        swin.pack()
        self.win = swin.window
        
        self.master1 = Canvas(self.win, width=700, height=700, border = 0)
        self.master1.pack()
        
        self.master2 = Canvas(self.win, width=700, height=700, border = 0)
        self.master2.pack()
        
        self.master3 = Canvas(self.win, width=700, height=700, border = 0)
        self.master3.pack()
        
        self.master4 = Canvas(self.win, width=700, height=700, border = 0)
        self.master4.pack()
        
        self.label1 = Label(self.master1, text= " \n What stock do you want us to predict? \n " , font=("Times New Roman", 20))
        self.label1.pack()
        
        self.textField = Entry(self.master1, width = 50, font = "Helvetica")
        self.textField.pack()
        
        #self.b1 = Button(self.master, text="GO!!!", command=self.quarry, activebackground = "crimson")
        self.b1 = Button(self.master1, text="GO!!!",command = self.stock_lookup, activebackground = "crimson")
        self.b1.pack()
        
        self.label = Label(self.master2, text= "Top Results", height = 2)

        self.Result1 = Button(self.master2, text= "", border = 0, command = self.callback1)
        self.Result2 = Button(self.master2, text= "", border = 0, command = self.callback1)
        self.Result3 = Button(self.master2, text= "", border = 0, command = self.callback1)
        self.Result4 = Button(self.master2, text= "", border = 0, command = self.callback1)
        self.Result5 = Button(self.master2, text= "", border = 0, command = self.callback1)
        self.faile = Label(self.master2, text= "", border = 0)
        
        self.verification = Label(self.master3, text= "", border = 0)
        self.prediction = Label(self.master3, text= "", border = 0)
        self.findings = Label(self.master3, text= "", border = 0)

        self.labelI = Label(self.master4, text="",   border = 0)
        self.sentiment1 = Label(self.master4, text= "", border = 0)
        self.sentiment2 = Label(self.master4, text= "", border = 0)
        self.sentiment3 = Label(self.master4, text= "", border = 0)
        self.sentiment4 = Label(self.master4, text= "", border = 0)
        
        
        
    def clearLabels(self):
        # all of these statements check to see if the element was packed and if it was, the element gets destroyed to make room for the next results
        
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
        #current_val is my reused regular expression
        
        current_val = re.compile(r'Current Value: .* \n')
        current_vals = current_val.findall(stud)
        value_of_stock = current_vals[0]
        
        current_val = re.compile(r'Prediction: .* \n')
        current_vals = current_val.findall(stud)
        prediction = current_vals[0]
        
        current_val = re.compile(r'Perecent_correct: .* \n')
        current_vals = current_val.findall(stud)
        percent_correct = current_vals[0]
        
        current_val = re.compile(r'Predicted Values: [^"]*\n Actual Values: \n')
        current_vals = current_val.findall(stud)
        predicted_vals = stud.split("]") 
    
        current_val = re.compile(r'Actual Values:[^"]* ---------')
        current_vals = current_val.findall(stud)
        actual_vals = stud.split("]") 
    
        cleaned_predicted = []
        cleaned_actual = []
        amount = []
        
        
        
        for r in range(len(predicted_vals)-1):
            cleaned_predicted.append(predicted_vals[r].strip()[1:])
            amount.append(r)
        for r in range(len(predicted_vals)-1):
            cleaned_actual.append(predicted_vals[r].strip()[1:])
            
        plt.plot(amount, cleaned_predicted, label = "predicted")
        plt.plot(amount, cleaned_actual, label = 'actual')
        plt.legend()
        plt.show()
        
        
        string = "This is what we concluded about " + stock + "."
        self.findings = Label(self.master3, text= string , border = 2, font=("Times New Roman", 14))
        self.findings.pack()
                    
        self.verification = Label(self.master3, text= "", border = 0, font=("Times New Roman", 14))
        self.verification.pack()
        val = float(percent_correct[19:-2].strip()) * 100
        val = float("{0:.2f}".format(val))
        self.verification['text'] = "We have been predicting movements in this stock with a " + str(val) + "% correctness"
                                  
        self.prediction = Label(self.master3, text= "", border = 0, font=("Times New Roman", 14))
        self.prediction.pack()
                      
        if "Increase" in prediction:
            self.prediction['text'] = "The stock is predicted to go on an upward trend!! \n \n"
        else:
            self.prediction['text'] = "The stock is predicted to go on an downward trend \n \n"
        
        i = 0
        for s in sentis:
            if "sentiment" in s:
                    if i == 0: 
                        string = " \n Here are some articles about " + stock + " \n "
                        self.labelI = Label(self.master4, text=string,   border = 0, font=("Times New Roman", 16))
                        self.labelI.pack()
                        
                        self.sentiment1 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14))
                        self.sentiment1.pack()
                        self.sentiment1['text'] = sentis[i][12:-2]
                    if i == 1: 
                        self.sentiment2 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14))
                        self.sentiment2.pack()
                        self.sentiment2['text'] = sentis[i][12:-2]
                    if i == 2: 
                        self.sentiment3 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14))
                        self.sentiment3.pack()
                        self.sentiment3['text'] = sentis[i] [12:-2]  
                    if i == 3: 
                        self.sentiment4 = Label(self.master4, text= "", border = 1, font=("Times New Roman", 14 ))
                        self.sentiment4.pack()
                        self.sentiment4['text'] = sentis[i][12:-2]
                    i = i + 1
        
 
    def run(self, stock):
            self.label.destroy()
            info = self.ticker_company(stock)
            stock = info[0]
            ticker = info[1]
            file = ticker.strip() + ".txt"
            exists = os.path.isfile(file)
            if exists:
                print("file exists")
                self.run_file(file)
            else:
                sent = Sentiment(stock,ticker)
                sentiments = sent.cleaned_data
                if csv_creator(ticker, sentiments, '12-10-2010'):
                    
                    sentis = sent.full_articles
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
                    

                    msft = CSV_Normalize()
                    msft.set_stock(ticker)
                    msft.set_normalized_input()
                    msft.set_normalized_output()
                    training_input = msft.get_training_input()
                    test_input = msft.get_testing_input()
                    training_output = [msft.get_training_output()]
                    test_output = msft.get_testing_output()
                    
                    neural_network = NeuralNetwork()
                    #print ("Random starting synaptic weights: ")
                    #print (neural_network.synaptic_weights)
                
                    neural_network.train(array(training_input), array(training_output).T, 100)
                    #neural_network.think(array(test_input[len(test_input)]))
                    
                    #print ("New synaptic weights after training: ")
                    #print (neural_network.synaptic_weights)
                    #print (guess)
                    #print (test_input[0])
                    #print (test_output[0])
                    
                    results = []
                    amount = []
                    actual = []
                
                    for n in range(len(test_output)):
                        results.append(neural_network.get_output(array(test_input[n])))
                        amount.append(n)
                
                    results_regular = []
                
                    for i in range(len(results)):
                        results_regular.append(msft.inverse(results[i]))
                    for o in range(len(test_output)):
                        actual.append(msft.inverse(test_output[o]))
                    
                    '''
                    fig = Figure(figsize=(3,6))
                    a = fig.add_subplot(111)
                    a.plot(amount, results_regular, color = "blue", label = "predicted")
                    a.plot(amount, actual, color = "red", label = "actual")
                    a.set_title("Forecasting Plot", fontsize = 10)
                    canvas = FigureCanvasTkAgg(fig, master = self.master3)
                    canvas.get_tk_widget().pack()
                    
                    
                    self.fig = Figure(figsize=(6,3))
                    a = self.fig.add_subplot(111)
                    a.plot(amount,results_regular, color = "blue", label = "predicted")
                    a.plot(amount, actual, color = "red", label = "actual")
                    a.set_title("Forecasting Plot", fontsize = 10)
                    self.canvas2 = Label(self.fig, master = self.master3)
                    self.canvas2.get_tk_widget().pack()
                    '''
                    
    
                    plt.plot(amount, results_regular, label = 'predicted')
                    plt.plot(amount, actual, label = 'actual')
                    plt.legend()
                    plt.show()
                    
                    
                    count = 0
                    temp = 0.0
                    tester = []
                    checker = []
                    #print("---------predicted-----------")
                    
                    for i in range(len(results_regular)):
                        #print (i[0])    
                        if count > len(results_regular)*.8:
                            #print("current = " + str(results_regular[i]) + "temp = " + str(temp))
                            temp = results_regular[i-1] 
                            if temp < results_regular[i]:
                                tester.append("Increase")
                            else:
                                tester.append("Decrease")
                          
                        count = count + 1
                        if count ==len(results_regular):
                            last_value_predicted = results_regular[i]
                            
                    count = 0
                    #print(tester)
                    #print("---------actual--------------")
                    for i in range(len(actual)):
                        #print (actual[i])
                        if count > len(actual)*.8:
                            temp = actual[i-1]
                            if temp < actual[i]:
                                checker.append("Increase")
                            else:
                                checker.append("Decrease")
                       
                        count = count + 1
                        if count ==len(actual):
                            last_value_actual = actual[i]
                    #print(checker)
                       
                    val = correctness(tester, checker)
                    
                    string = "This is what we concluded about " + stock + "."
                    self.findings = Label(self.master3, text= string , border = 2, font=("Times New Roman", 14))
                    self.findings.pack()
                    
                    self.verification = Label(self.master3, text= "", border = 0, font=("Times New Roman", 14))
                    self.verification.pack()
                    val = val * 100
                    val = float("{0:.2f}".format(val))
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
                        
    def stock_lookup(self):
        self.clearLabels()
        name = self.textField.get()
        self.label = Label(self.master, text= "Top Results \n ", height = 2)
        self.label.pack()
        #url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query=Apple&region=1&lang=en&callback=YAHOO.Finance.SymbolSuggest.ssCallback'
        url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query='+ name + '&region=1&lang=en&callback=YAHOO.Finance.SymbolSuggest.ssCallback'
        data = requests.get(url)
        #data.encoding = 'utf-8'
        data = data.text
        
        #data = data.replace("YAHOO.Finance.SymbolSuggest.ssCallback", "") #unnecessary syntax in return string
        
        company = re.compile(r'"name":"[^"]+"') # pattern to get name of company
        companys = company.findall(data)        # finds all of the company's names
        
        ticker = re.compile(r'"symbol":"[^"]+"')
        tickers = ticker.findall(data) 
    
    
        company_info = []
        company_info.append(tuple((0,0)))
        
        for i in range(len(companys)):      
            company_info.append((tuple(((companys[i][8:-1]),tickers[i][10:-1]))))
           
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
            if actual[i] == predict[i]:
                correct = correct + 1
        return(correct / len(predict))

def main():   
    root = Tk()
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    gui = Engine(root)
    root.mainloop()
main()