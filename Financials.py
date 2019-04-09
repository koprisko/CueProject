#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:41:49 2019

@author: brianheckman
"""
import requests
import csv
import pandas as pd
class Financials:
    ticker = ""
    def set_ticker(self,ticker):
        self.ticker = ticker
    def write_bs(self):
        url = "https://financialmodelingprep.com/api/financials/balance-sheet-statement/" + self.ticker + "?period=quarter&datatype=json"
        r = requests.get(url)
        with open(self.ticker + "_bs.json", 'wb') as output:
            output.write(r.content)
    def write_is(self):
        url = "https://financialmodelingprep.com/api/financials/income-statement/" + self.ticker + "?period=quarter&datatype=json"
        r = requests.get(url)
        with open(self.ticker + "_is.json", 'wb') as output:
            output.write(r.content)
    def write_cf(self):
        url = "https://financialmodelingprep.com/api/financials/cash-flow-statement/" + self.ticker + "?period=quarter&datatype=json"
        r = requests.get(url)
        with open(self.ticker + "_cf.json", 'wb') as output:
            output.write(r.content)
        
def main():
    ticker = 'MSFT'
    msft = Financials()
    msft.set_ticker(ticker)
    msft.write_bs()
    msft.write_is()
    msft.write_cf()
    
main()
    
        
        