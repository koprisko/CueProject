# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:30:08 2019

@author: kyleo
"""

import urllib
import urllib.request


def getJson(symbol):
        with urllib.request.urlopen("https://financialmodelingprep.com/api/financials/income-statement/AAPL") as url:
            data = url.read().decode()
            print(data)

def main():  
    symbol = "AAPL"
    with urllib.request.urlopen("https://financialmodelingprep.com/api/financials/income-statement/" + symbol) as url:
            #data = json.loads(url.read().decode())
            data = url.read().decode()
            JSONFile = open(symbol + "_information.txt", "a")
            JSONFile.write(data)
    
    
    
    
    
    #tickerSymbols = open("tickersymbols.txt", "r")
    #tickerSymbols = open("somestocks.txt", "r")
    #for tick in tickerSymbols:
     #   tick = tick.rstrip()
        #regex = "+" + tick + "+"
        #if regex not in open("DataFound.txt", "r").read() and count < 15:
     #   getJson(tick)

main()

