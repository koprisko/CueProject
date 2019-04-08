#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:37:27 2019

@author: brianheckman
"""
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
import LoadCSV_V2
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

class Fetcher:
    import calendar as cal
    import pandas as pd
    import datetime as dt
    import re
    import requests
    api_url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=%s&events=%s&crumb=%s"
    def __init__(self, ticker, start, end=None, interval="1d"):
        """Initializes class variables and formats api_url string"""
        self.ticker = ticker.upper()
        self.interval = interval
        self.cookie, self.crumb = self.init()
        self.start = int(cal.timegm(dt.datetime(*start).timetuple()))

        if end is not None:
            self.end = int(cal.timegm(dt.datetime(*end).timetuple()))
        else:
            self.end = int(time.time())

    def init(self):
        """Returns a tuple pair of cookie and crumb used in the request"""
        url = 'https://finance.yahoo.com/quote/%s/history' % (self.ticker)
        r = requests.get(url)
        txt = r.content
        cookie = r.cookies['B']
        pattern = re.compile('.*"CrumbStore":\{"crumb":"(?P<crumb>[^"]+)"\}')

        for line in txt.splitlines():
            m = pattern.match(line.decode("utf-8"))
            if m is not None:
                crumb = m.groupdict()['crumb']
                crumb = crumb.replace(u'\\u002F', '/')
        return cookie, crumb  # return a tuple of crumb and cookie

    def getData(self, events):
        """Returns a list of historical data from Yahoo Finance"""
        if self.interval not in ["1d", "1wk", "1mo"]:
            raise ValueError("Incorrect interval: valid intervals are 1d, 1wk, 1mo")

        url = self.api_url % (self.ticker, self.start, self.end, self.interval, events, self.crumb)

        data = requests.get(url, cookies={'B':self.cookie})
        content = StringIO(data.content.decode("utf-8"))
        return pd.read_csv(content, sep=',')

    def getHistorical(self, events='history'):
        """Returns a list of historical price data from Yahoo Finance"""
        return self.getData('history')

    def getDividends(self):
        """Returns a list of historical dividends data from Yahoo Finance"""
        return self.getData('div')

    def getSplits(self):
        """Returns a list of historical splits data from Yahoo Finance"""
        return self.getData('split')

    def getDatePrice(self):
        """Returns a DataFrame for Date and Price from getHistorical()"""
        return self.getHistorical().ix[:,[0,4]]

    def getDateVolume(self):
        """Returns a DataFrame for Date and Volume from getHistorical()"""
        return self.getHistorical().ix[:,[0,6]]
    
    
    

class NeuralNetwork():
    from numpy import exp, array, random, dot
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 4 input connections and 1 output connection.
        # We assign random weights to a 4 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return ((1 / (1 + exp(-x))))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return (x * (1 - x))

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
    
class Sentiment:
    def __init__(self, stock, ticker, startdate, enddate):
        #start and end dates in format YYYYMMDD as a string
        key = "Q41EjWTOsr2VA3y4UaXDqMmpWg6aNbDr"
        info = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q=" + stock
        info = info + "&begin_date=" + startdate
        info = info + "&end_date=" + enddate
        info = info + "&subject:Stocks and Bonds&api-key=" + key
        
        r = requests.get(info)
        r.encoding = 'utf-8'
        
        data = self.collector(r.text,stock)
        
        r = requests.get("https://feeds.finance.yahoo.com/rss/2.0/headline?s=" + ticker + "&region=US&lang=en-US")
        r.encoding = 'utf-8'
        info2 = r.text
        data2 = self.yahooCollector(info2) 
                
        count = 0    
        self.total_percent = 0
        if data:
            for d in data: 
                #if d != 0:
                   count = count + 1
                   self.total_percent = self.total_percent + self.check_polar(d)
               
        if data2:
            for d in data2: 
                #if d != 0:
                   count = count + 1
                   self.total_percent = self.total_percent + self.check_polar(d)
            
        self.total_percent = self.total_percent / count
        print("total_positive_sentiment from " + str(count) + " articles = " + str(self.total_percent))
        
        
    def yahooCollector(self, data):
        irrelevant = re.compile(r'<title>Yahoo! Finance:.* News</title>')
        data = re.sub(irrelevant, "", data)
        
        irrelevant = re.compile(r'<description>Latest Financial News for.*</description>')
        data = re.sub(irrelevant, "", data)
        
        title = re.compile(r'<title>.*</title>')
        titles = title.findall(data)
        
        description = re.compile(r'<description>.*</description>')
        descriptions = description.findall(data)
        #print(r.text)
        good = True
        
        sentiment_array = []
        while good == True:
            polarity = 0
            #print("-------------------")
            if titles:
                temp = titles.pop()
                #print("title = " + temp[7:-8])
                Sa = self.yahooClassifier(temp)
                polarity = Sa + polarity
                
            if descriptions:
                temp = descriptions.pop()
                #print("description = " + temp[13: -14])
                Sa = self.yahooClassifier(temp)
                polarity = polarity + Sa
            else:
                good = False
            
            polarity = self.check_polar(polarity)
            sentiment_array.append(polarity)
        

        return sentiment_array

    def clean_headlines(self, info):
        for i in info:
            self.classifier(i[18:-3])
            
    def clean_snippets(self, info):
        for i in info:
            self.classifier(i[11:-1])
            
    def clean_mains(self, info):
        for i in info:
            self.classifier(i[8:-1])       
            
    def clean_paragraphs(self, info):
        for i in info:
            self.classifier(i[18:-3])        
            
               
    def clean_headline(self, info):
            return(info[18:-3])
            
    def clean_snippet(self, info):
            return(info[11:-1])
            
    def clean_main(self, info):
            return(info[8:-1])  
            
    def clean_paragraph(self, info):
            return(info[18:-3])
    
    def check_polar(self, pols):
        neg = -.25
        pos = .5
        if pols < pos and pols > neg:
            return 0
        if pols >= pos:
            return 1
        if pols <= neg:
            return -1
    
            
    def collector(self, stud,stock):
        
        head_Line = re.compile(r'"print_headline":"[^"]+","')
        headers = head_Line.findall(stud)
        stud = re.sub(head_Line, "", stud)
        #clean_headlines(headers)
        
        snips = re.compile(r'"snippet":"[^"]+"')
        snippets = snips.findall(stud)
        stud = re.sub(snips, "", stud)
        #clean_snippets(snippets)
        
        main = re.compile(r'"main":"[^"]+"')
        main_articles = main.findall(stud)
        stud = re.sub(main, "", stud)
        #clean_mains(main_articles)
        
        paragraph = re.compile(r'"lead_paragraph":"[^"]+","')
        paragraphs = paragraph.findall(stud)
        stud = re.sub(paragraph, "", stud)
        
        #clean_paragraphs(paragraphs)
        #print(len(paragraphs))
        #print(len(main_articles))
        #print(len(snippets))
        #print(len(headers))
    
        sentiment_array = []
        
        while main_articles:
            #print ("main article:")
            ma = self.classifier((main_articles.pop())) # while already checks to make sure there is one
            count = 1.0
            polarity = ma
            
            if headers:
                #print("header:")
                he = self.classifier(self.clean_paragraph(headers.pop()))
                count = count + 1
                polarity = polarity + he
                
            if paragraphs:    
                #print("paragraph:")
                pa = self.classifier(self.clean_paragraph(paragraphs.pop()))
                count = count + 1
                polarity = polarity + pa
                
            if snippets:
                #print("snippet:")
                sn = self.classifier(self.clean_snippet(snippets.pop()))
                count = count + 1
                polarity = polarity + sn
                    
                
            
            
            #print ("This message has a total sentiment of " + str(total_sentiment) + "%" )
            #print("--------------NEW MESSAGE------------------")
            sentiment_array.append(polarity) 
        return(sentiment_array)   
                
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
        
    def yahooClassifier(self, info):
        #info = self.clean_up(info)
        sentiment = self.sentiment_analyzer_scores(info)
        #print(sentiment)
        #print(info + " " + str(sentiment))
        return(sentiment)

class CSV_Normalize:
    import csv
    stock = ""
    
    close_prices = []
    high_prices = []
    
    prev_prices = []
    
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

        self.close_prices = self.close_prices[1:-1]
        self.high_prices = self.high_prices[1:-1]
        self.prev_prices = self.prev_prices[1:-1]

        for m in range(len(self.close_prices)):
            self.close_prices[m] = float(self.close_prices[m])
        for n in range(len(self.high_prices)):
            self.high_prices[n] = float(self.high_prices[n])
        for pp in range(len(self.prev_prices)):
            self.prev_prices[pp] = float(self.prev_prices[pp])
    
    
        #Set Min and Max values for normalization

        for p in range(len(self.close_prices)):
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
        
    def set_normalized_input(self):
        self.set_input()
        for i1 in range(len(self.close_prices)):
            self.normalized_close.append((self.close_prices[i1] - self.min_close)/(self.max_close - self.min_close))
        
        for i2 in range(len(self.high_prices)):
            self.normalized_high.append((self.high_prices[i2] - self.min_high)/(self.max_high - self.min_high))
            
    
        for i4 in range(len(self.prev_prices)):
            self.normalized_prev.append((self.prev_prices[i4] - self.min_prev)/(self.max_prev - self.min_prev))
            
    def get_input(self):
        return (list(zip(self.close_prices,self.high_prices,self.prev_prices)))
    
    def get_nomralized_input(self):
        return (list(zip(self.normalized_close,self.normalized_high,self.normalized_prev)))
    
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
        for i in range(len(self.normalized_close)):
            temp_list = [self.normalized_close[i],self.normalized_high[i],self.normalized_prev[i]]
            self.inputs.append(temp_list)
        train_end = int(.7*len(self.inputs))
        self.training_inputs = self.inputs[0:train_end]
    
    def get_testing_input(self):
        self.set_testing_input()
        return self.testing_inputs
    
    def set_testing_input(self):
        train_end = int(.7*len(self.inputs))
        self.testing_inputs = self.inputs[train_end:]
    
    def get_training_output(self):
        self.set_training_output()
        return self.training_outputs
    def set_training_output(self):
        train_end = int(.7*len(self.normalized_open))
        self.training_outputs = self.normalized_open[0:train_end]
    def get_testing_output(self):
        self.set_testing_output()
        return self.testing_outputs
    def set_testing_output(self):
        train_end = int(.7*len(self.normalized_open))
        self.testing_outputs = self.normalized_open[train_end:]
        
        
        
    

    
        
    
def main():
    stock = input("What is the stock?")
    ticker = input("What is the ticker?")
    data = Fetcher(ticker, [2016,1,1], [2016,4,8],'1d').getHistorical()
    data.to_csv(ticker + '.csv')
    print (data)
    msft = CSV_Normalize()
    msft.set_stock(ticker)
    msft.set_normalized_input()
    msft.set_normalized_output()
    training_input = msft.get_training_input()
    test_input = msft.get_testing_input()
    training_output = [msft.get_training_output()]
    test_output = msft.get_testing_output()
    
    sent = Sentiment(stock,ticker,"20160101","20160208")
    print (sent.total_percent)
    

    
    neural_network = NeuralNetwork()
    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)
    
    neural_network.train(array(training_input), array(training_output).T, 100)
    guess = (neural_network.think(array(test_input[0])))
    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)
    print (guess)
    print (test_input[0])
    print (test_output[0])
    
    results = []
    amount = []
    actual = []
    
    for n in range(len(test_output)):
        results.append(neural_network.think(array(test_input[n])))
        amount.append(n)
    
    results_regular = []
    
    for i in range(len(results)):
        results_regular.append(msft.inverse(results[i]))
    for o in range(len(test_output)):
        actual.append(msft.inverse(test_output[o]))
        
    #plt.plot(amount, results, label = "predicted")
    plt.plot(amount, results_regular, label = 'predicted')
    #plt.plot(amount, test_output, label = 'actual')
    plt.plot(amount, actual, label = "actual")
    plt.legend()
    plt.show()

    
main()  
    
    