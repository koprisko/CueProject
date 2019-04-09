# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:30:08 2019

@author: kyleo
"""

import urllib
import json
import urllib.request

import requests
import json
import csv


#import tensorflow as tf
#from tensorflow.keras import layers

#url required to call microsoft data with API key daily results:  https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&outputsize=full&apikey=52O3GXSWQOEBGMOT
#url required to call USD data with monthly results : https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol=EUR&to_symbol=USD&apikey=52O3GXSWQOEBGMOT
#url required to create a easy search engine with their information: https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=BA&apikey=52O3GXSWQOEBGMOT


def createCSV(symb, TUPLES):
    file = symb + ".csv"
    with open(file, 'w', newline='') as csvFile:
          writer = csv.writer(csvFile)
          writer.writerows(TUPLES)    
          print("created " + symb)

def getJson(symbol,key):
            """
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+ symbol + '&outputsize=full&apikey=' + key
            data = requests.get(url)
            data.encoding = 'utf-8'
            data = data.json()
            
            tups = []
            dates = []
            
            if 'Time Series (Daily)' in data:
                for i in data['Time Series (Daily)']:
                    dates.append(i)
                count = 1
                for i in dates:
                    opens = data['Time Series (Daily)'][i]['1. open']
                    highs = data['Time Series (Daily)'][i]['2. high']
                    lows = data['Time Series (Daily)'][i]['3. low']
                    close = data['Time Series (Daily)'][i]['4. close']
                    volume = data['Time Series (Daily)'][i]['5. volume']
                    count = count + 1
                    tups.append(tuple((count, i, opens, highs, lows, close, volume))) 
                
                    File = open("DataFound.txt", "a")
                    File.write("+" + symbol + "+\n")
            else:
               print("error with ticker symbol " + symbol)
               print(data) 
               
            createCSV(symbol, tups)
            
            """
            
            url ='https://financialmodelingprep.com/api/financials/income-statement/' + symbol + '?period=quarter&datatype=json'
            data = requests.get(url)
            data.encoding = 'utf-8'
            data = data.json()
            for i in data['financials']:
                print (i)
                dates = i['date']
                revenues = i['Revenue']
                revenue_growth = i['Revenue Growth']
                cost_of_revenue = i['Cost of Revenue']
                gross_profit = i['Gross Profit']
                rANDd_expenses = i['R&D Expenses']
                sgANDa_expense = i['SG&A Expense']
            
                

def main():   
    count = 0
    key2 =  "52O3GXSWQOEBGMOT"
    key =  "0F9TBPXWF5YV5392"
    key3 =   "S8SGZ63ZVTYFOKV0"
    
    tickerSymbols = open("tickersymbols.txt", "r")    #used to get info from the list of stocks we compiled
    #tickerSymbols = open("somestocks.txt", "r")      #used to test 5 stocks
    for tick in tickerSymbols:
        tick = tick.rstrip()
        regex = "+" + tick + "+"
        if regex not in open("DataFound.txt", "r").read() and count < 1:
                if count <= 5: 
                    count = count + 1
                    getJson(tick,key2)
                if count > 5 and count <= 10: 
                    count = count + 1
                    getJson(tick,key3)
                if count > 10:
                    getJson(tick,key)
                    count = count + 1
main()

