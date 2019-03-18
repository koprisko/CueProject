# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:21:27 2019

@author: kyleo
"""

import re
import glob
import queue
import json
import csv


def cleanDate(date):
    date = date[:-4]    
    return(date)

def cleanClose(close):
    close = close[12:-1]
    return (float(close))

def cleanVolume(vol):
    vol = vol[13:-1]
    return(int(vol))
    
    
def createCSV(symb, TUPLES):
    file = symb + ".csv"
    with open(file, 'w', newline='') as csvFile:
          writer = csv.writer(csvFile)
          writer.writerows(TUPLES)    
          print("created " + symb)
        
def main():
    
    #regualr expressions used to get the data from the ALPHA vantage API
    date = re.compile(r'\d\d\d\d-\d\d-\d\d": {')  
    closingPrice = re.compile(r'4. close": "\d*.\d*"')
    volume = re.compile(r'5. volume": "\d*.\d*"')
    
    for filename in glob.glob('*PL_information.txt'):
        with open(filename, 'r') as myfile:
            
            #these are used so that the lists stay in order
            dateQ = queue.Queue()
            closeQ = queue.Queue()
            volQ = queue.Queue()
            
            #used to store info from diffrent parts of code
            TUPLES = []
            TUPLES.append(tuple(("DATE", "CLOSE", "VOLUME", "REVENUE", "TOTAL OPERATING EXPENSE", "Total Current Assets", "Accumulated Depreciation")))
            revTups = []
            fsTups = []
    
            #cuts off the end of the filename and grabs the symbol that we are using
            symb= filename[:-16]
            print("--------------------------------------------------------")
            print(symb)
            
            data =myfile.read().replace('\n', '')
            
            preData = re.split("<pre>", data) #splits the diffrent json files up
            
            #print(len(preData))
            #print(preData[0])  closing and volume numbers
            #print(preData[1])  income sheet
            #print(preData[3])  balance sheet
            
            #used the regex in the closing and volume
            dates = date.findall(preData[0])
            closingPrices = closingPrice.findall(preData[0])
            volumeList = volume.findall(preData[0])
            
            #print(len(preData))  
            
            if (len(preData) > 1) and ("Revenue" in preData[1]) and ("Total operating expenses" in preData[1]):  #makes sure the dataMiner worked and added another json 
                print("success")
                fsData = json.loads(preData[1])
                revs = fsData[symb]["Total operating expenses"]
                
                for x in revs:
                    info = int(fsData[symb]["Total operating expenses"][x])
                    info2 = int(fsData[symb]["Revenue"][x])
                    revTups.append(tuple((x, info2 , info )))
                #print(revTups[0][0])
                    
            if (len(preData) > 3):
                bsData = json.loads(preData[3])
                LENGTH = bsData[symb]["Total current assets"]
                for x in LENGTH:
                    info = int(bsData[symb]["Total current assets"][x])
                    info2 = int(bsData[symb]["Accumulated Depreciation"][x])
                    fsTups.append(tuple((x, info2 , info )))
                
            for price in closingPrices:
                cleanD = cleanClose(price)
                closeQ.put(cleanD)
                
            for time in dates:
                period = cleanDate(time)
                dateQ.put(period)
                
            for vols in volumeList:
                cleanVol = cleanVolume(vols)
                volQ.put(cleanVol)
                
            while not dateQ.empty():
                check = "F"
                time = dateQ.get()
                count2 = 0
                year = time[0:4]
                for x in revTups:
                    if (year == revTups[count2][0][0:4]) and len(fsTups) > 1:
                        tup = tuple((time, closeQ.get(), volQ.get(), revTups[count2][1], revTups[count2][2], fsTups[count2][1], fsTups[count2][2] ))
                        TUPLES.append(tup)
                        #print("match " + time + "=" + revTups[count2][0][0:4])
                        #print(tup)
                        #print("---------------------")
                        check = "T"
                    count2 = count2 + 1
                    
                if (check == "F"):
                        tup = tuple((time, closeQ.get(), volQ.get(), 0,0))  
                        TUPLES.append(tup)
                        #print("here" + time + "=" + revTups[count2][0][0:4])
                        #print(tup)
                        #print("----------------------")
                               
            createCSV(symb, TUPLES) 
                
main()