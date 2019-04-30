# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:53:39 2019

@author: kyleo
"""

from tkinter import Tk, Label, Button, Entry, LEFT, RIGHT, BOTTOM, Frame, Checkbutton, Canvas, PhotoImage
from PIL import ImageTk

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


nltk.download('stopwords')
stop_words = set(stopwords.words("english"))


class NeuralNetwork():
    from numpy import exp, array, random, dot
    def __init__(self):
        # We model a single neuron, with 4 input connections and 1 output connection.
        # We assign random weights to a 4 x 1 matrix, with values in the range 0 to 1
        # These weights will be adjusted based on the error
        
        self.synaptic_weights = array([[.40], [.40], [.40], [.2]])

    # The Cost Function shows us how our model is performing. We chose MSE to show how far each point
    # deviates from the actual point. The goal is to minimize this number.
    
    def cost_function(self, features, targets):

        #Find length of list
        N = len(targets)

        #Get list of predictions of the output using synaptic weights
        predictions = dot(features, self.synaptic_weights)
    
        # Find square error for entire matrix
        sq_error = (predictions - targets)**2

        # Return average squared error among predictions
        return 1.0/(2*N) * sq_error.sum()

    # The derivative of the MSE cost function.
    # This is the gradient of the MSE function that uses partial derivatives of each parameter.
    # It indicates how confident we are about the existing weight, and how much we have to change each weight.
    
    def update_weights(self,features, targets, lr):

        # Find predictions using w1x1 + w2x2 + w3x3...concept 
        predictions = dot(features, self.synaptic_weights)

        #Extract our features and put them into their own list
        x1 = features[:,0]
        x2 = features[:,1]
        x3 = features[:,2]
        x4 = features[:,3]

        # Use matrix cross product (*) to simultaneously
        # calculate the derivative for each weight (partial derivative)
        
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
            # And ultimately return the MSE
            
            output = self.think(training_set_inputs,training_set_outputs)

            # Update the weights by calling the function with a learning rate of .8.
            
            self.update_weights(training_set_inputs,training_set_outputs,.8)


    # The neural network thinks.
    
    def think(self, inputs, outputs):
        
        # Pass inputs through our neural network (our single neuron). And return MSE
        
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
        
        """
        
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
        
       
        """
        
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
                              
            sentiment_array.append(tuple((dates[count], polarity))) # an array of tuples with dates and the polarity of the article on that day is created
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
                        date_polarity = 0
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
                                   
def main(): 
    tickerSymbols = open("sp500.txt", "r")    #used to get info from the list of stocks we compiled
    stat_file = open("stat_file.txt", "a")
    #tickerSymbols = open("somestocks.txt", "r")      #used to test 5 stocks
    index = 0
    for tick in tickerSymbols:
        contents  = tick.split()
        ticker = contents[0]
        regex = "+" + ticker + "+"
        if regex not in open("DataFound.txt", "r").read() and index < 5:
                string = ""
                index = index + 1
                stock = ""
                count = 0
                while count < len(contents)-1:
                    count = count + 1
                    stock = stock + " " + contents[count]
                    stock = stock.strip()
                print("Company Name: " + stock)
                print("Ticker Symbol: " + ticker)
                
                
            
                sent = Sentiment(stock,ticker)
                sentiments = sent.cleaned_data
                articles = sent.full_articles
                i = 0
                
                if csv_creator(ticker, sentiments, '12-10-2010'):
                    
                    info_file = open(ticker + ".txt", "a")
                    info_file.write("Company Name: " + stock + " \n")
                    info_file.write("Ticker Symbol: " + ticker  + " \n")
                    
                    while i < len(articles):
                        if i < 5:
                            print("sentiment" + str(i+1) + ": " + articles[i].strip() + " \n")
                            print("---")
                            
                            info_file.write("sentiment" + str(i+1) + ": " + articles[i].strip() + " \n")
                            info_file.write("---")
                            
                        i = i + 1
                    
                    
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
                        
                    #plt.plot(amount, results, label = "predicted")
                    #plt.plot(amount, results_regular, label = 'predicted')
                    #plt.plot(amount, test_output, label = 'actual')
                    #plt.plot(amount, actual, label = 'actual')
                    #plt.legend()
                    #plt.show()
    
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
                                temp2 = "Increase"
                                tester.append(temp2)
                            else:
                                temp2 = "Decrease"
                                tester.append(temp2)
                        if count ==len(results_regular)-1:
                            last_value_predicted = temp2
                        count = count + 1            
                                 
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
                        
                        
                    val = correctness(tester, checker)
                    print("Current Value: " + str(actual[count-1]))
                    print("Prediction: " + last_value_predicted)
                    print("Perecent_correct: "+ str(val)) 
                    print("Predicted Values: \n")
                    
                    info_file.write("Current Value: " + str(actual[count-1]) + " \n")
                    info_file.write("Prediction: " + last_value_predicted + " \n")
                    info_file.write("Perecent_correct: "+ str(val) + " \n") 
                    info_file.write("Predicted Values: \n ")
                    
                    for i in results_regular:
                        print(i)
                        info_file.write(str(i))
                    print("Actual Values: \n" )
                    info_file.write("Actual Values: \n ")
                    for i in actual:
                        print(i)
                        info_file.write(str(i))
                    
                    info_file.write("\n ---------")
                    info_file.close()
                    msft.clear_lists()
                    
                    string = ticker + " " + str(actual[count-1]) + " " + last_value_predicted + " \n"
                    stat_file.write(string)
                    
                    
                    del(msft)
                    del(neural_network)
                    del(training_input)
                    del(test_input)
                    del(training_output)
                    del(test_output)
                    
                    index = index + 1
    tickerSymbols.close()
    stat_file.close()

    
main()
