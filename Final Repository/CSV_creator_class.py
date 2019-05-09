# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:06:15 2019

@author: kyleo
"""
import requests
import datetime as dt
import datetime
import csv
import statistics


"""This class is responsible for creating a csv file for a specific ticker symbol. This accesses the alpha vantage api to get historic data
and appends that sentiment values that were mined in the sentiment class. The sentiment class returns the sentiment scores for a particular day, and 
the csv creator is responsible for appending that data into a csv that is then opened in csv normalize and fed into the nerual network"""


class csv_creator:
    def __init__(self, ticker, sentiments, cutoff): 
        
        #we were able to get multiple api keys so that we have more chances at getting data
        key2=  "52O3GXSWQOEBGMOT" 
        key=  "0F9TBPXWF5YV5392"
        key3 =   "S8SGZ63ZVTYFOKV0"
        
        # this is block that continually checks keys for a certain stock ticker symbol
        # if none work then we move to the next stock
        self.verify = False
        if self.getJson(ticker,key, sentiments,cutoff) == True :
            self.verify = True
        else:
            if self.getJson(ticker,key2, sentiments,cutoff) == True :
                self.verify = True
            else:
                if self.getJson(ticker,key3, sentiments,cutoff) == False :
                    self.verify = False
                else:
                    self.verify = True
        
    def createCSV(self, symb, TUPLES):
        
        #opens a file and appends all of the tuples to it
        file = symb + ".csv"
        with open(file, 'w', newline='') as csvFile:
              writer = csv.writer(csvFile)
              writer.writerows(TUPLES)    
              #print("created " + symb)
              
              
    def getJson(self,symbol,key, sentiments,cutoff_date):      
            # makes the api call to alpha data
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+ symbol + '&outputsize=full&apikey=' + key
            data = requests.get(url)
            data.encoding = 'utf-8'
            data = data.json() # makes the json a dictionary
            #print(data)
            
            # arrays that store information during the data stripping process
            tups = [] #array of tuples to be appended to the csv file
            dates = [] #stores the dates so we can navigate the dictionary using them
            date_list = [] #used to store dates in the syntax that we want them
            
            if 'Time Series (Daily)' in data: # makes sure we got valid information
                for i in data['Time Series (Daily)']: # navigates the inportatnt part of the dictionary
                    # used to format the dates in the way that we want to combine dates
                    date_string = ""
                    dt = datetime.datetime.strptime(i, "%Y-%m-%d")
                    date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year)
                    dates.append(i) 
                    date_list.append(date_string)
                    
                count = 0 #used to index date information
                
                
                a = dt.strptime(cutoff_date, "%m-%d-%Y") # stores date for comparision of the cutoff date        
                for i in dates: # navigates the dates for the dictionary
                    b = dt.strptime(i, "%Y-%m-%d") # stores the date for comaprison
                    if b > a : # this is used to append infomrmation that occured after the cutoff date specified when user is passed
                        date_polarity = 0.0
                        date_string = ""
                        dt = datetime.datetime.strptime(i, "%Y-%m-%d")
                        date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year) #used to easily compare dates
                        
                        for j in sentiments: #goes through all of the sentiments that we stored in sentiments
                            if date_string == j[0] : #sees if the dates are the same
                                date_polarity = date_polarity + j[1]  #combines sentiment values if they occured on the same day   
                                
                        #gets the data from the disctionary that we want to include in our csv
                        opens = data['Time Series (Daily)'][i]['1. open']
                        highs = data['Time Series (Daily)'][i]['2. high']
                        lows = data['Time Series (Daily)'][i]['3. low']
                        close = data['Time Series (Daily)'][i]['4. close']
                        volume = data['Time Series (Daily)'][i]['5. volume']
                        #creates a tuple with the information we want about a day
                        tups.append(tuple((count, date_list[count], opens, highs, lows, close, volume, float(date_polarity)))) 
                        count = count + 1 #dae_list gets updated it has the proper date format that we want
                # we want older data first and new data last to be fed into the neural network
                tups.reverse()
                self.createCSV(symbol, tups)
                return True 
            else:
               print("error with ticker symbol " + symbol)
               return False