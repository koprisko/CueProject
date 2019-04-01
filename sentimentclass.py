# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:03:23 2019

API Key = https://api.nytimes.com/svc/search/v2/articlesearch.json?q=apple&api-key=Q41EjWTOsr2VA3y4UaXDqMmpWg6aNbDr
@author: kyleo
"""

#import urllib
#import urllib.request

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#from textblob import TextBlob

import re
import requests

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))


class NYTSentiment:
    def __init__(self, stock):
        key = "Q41EjWTOsr2VA3y4UaXDqMmpWg6aNbDr"
        info = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q=" + stock
        info = info + "&subject:Stocks and Bonds&api-key=" + key
        
        r = requests.get(info)
        r.encoding = 'utf-8'
        
        data = self.collector(r.text,stock)
        
        count = 0    
        self.total_percent = 0
        
        for d in data: 
            if d != 0:
               count = count + 1
               self.total_percent = self.total_percent + d
            
        self.total_percent = self.total_percent / count
        print("total_positive_sentiment from " + str(count) + " articles = " + str(self.total_percent))
        

    def clean_headlines(self, info):
        for i in info:
            self.classifier3(i[18:-3])
            
    def clean_snippets(self, info):
        for i in info:
            self.classifier3(i[11:-1])
            
    def clean_mains(self, info):
        for i in info:
            self.classifier3(i[8:-1])       
            
    def clean_paragraphs(self, info):
        for i in info:
            self.classifier3(i[18:-3])        
            
               
    def clean_headline(self, info):
            return(info[18:-3])
            
    def clean_snippet(self, info):
            return(info[11:-1])
            
    def clean_main(self, info):
            return(info[8:-1])  
            
    def clean_paragraph(self, info):
            return(info[18:-3])
    
    def check_polar(self, pols):
        if pols < .5 and pols > -.5:
            return 0
        if pols >= .5:
            return 1
        if pols <= -.5:
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
        
        image = re.compile(r'"image[^"]+"')
        stud = re.sub(image, "", stud)
        #images = image.findall(stud)
        #print(images)
        
        sentiment_array = []
        
        
        while main_articles:
            print ("main article:")
            ma = self.classifier3((main_articles.pop())) # while already checks to make sure there is one
            count = 1.0
            pos = 0.0
            neg = 0.0
            
            if ma > 0:
                pos = pos + 1
            if ma < 0:
                neg = neg + 1
            
            if headers:
                print("header:")
                he = self.classifier3(self.clean_paragraph(headers.pop()))
                count = count + 1
                if he > 0:
                    pos = pos + 1
                if he < 0:
                    neg = neg + 1
                
            if paragraphs:    
                print("paragraph:")
                pa = self.classifier3(self.clean_paragraph(paragraphs.pop()))
                count = count + 1
                if pa > 0:
                    pos = pos + 1
                if pa < 0:
                    neg = neg + 1
                    
            if snippets:
                print("snippet:")
                sn = self.classifier3(self.clean_snippet(snippets.pop()))
                count = count + 1
                if sn > 0:
                    pos = pos + 1
                if sn < 0:
                    neg = neg + 1
                    
            total_pos_sentiment = pos / count
            total_neg_sentiment = neg / count
            print ("This message returned positive " + str(total_pos_sentiment) + "% of the time." )
            print ("This message returned negative " + str(total_neg_sentiment) + "% of the time." )
            
             
            total_sentiment = total_pos_sentiment - total_neg_sentiment
            
            print ("This message has a total sentiment of " + str(total_sentiment) + "%" )
            print("--------------NEW MESSAGE------------------")
            sentiment_array.append(total_sentiment) 
        
        return(sentiment_array)
        #rank = re.compile(r'{"rank[^"]+}')
        #stud = re.sub(image, "", stud)
        #ranks = rank.findall(stud)
        #print(ranks)
        
        #print(stud)
        
    def classifier(info):
        print("---------------------------------------------------------")
        print(info)   
        ans = input("1 for good -- 0 for bad  -- 5 for neutral -- other to stop \n")
        try:  
            test3num = int(ans)
            ans = test3num
        except ValueError:
            print("not a number")
        if ans == 1:
            print("Review was sent to the good file\n")
            goodFile = open("goodInformation.txt", "a")
            goodFile.write(info + "\n")
            goodFile.close()
        if ans == 0:
            print("Review was sent to the bad file\n")
            badFile = open("badInformation.txt", "a")
            badFile.write(info + "\n")
            badFile.close()
        if ans == 5:
            print("Review was sent to the neutral file\n")
            badFile = open("neutralInformation.txt", "a")
            badFile.write(info + "\n")
            badFile.close()
        else:
            return 0
    
        
    def classifier2(info):
        tokenized_word=word_tokenize(info)
        
        filtered_sent=[]
        for w in tokenized_word:
            if w not in stop_words and w.isalpha():
                filtered_sent.append(w)
        print("Filterd Sentence:",filtered_sent)
        i = 0
        while(i < len(filtered_sent) -1):
            #print (filtered_sent[i])
            i = i + 1
            check = i % 2
            if i == 0:
                temp1 = filtered_sent[i-1]
                temp2 = filtered_sent[i]
                print(temp1 + " " + temp2)
    
            elif check == 0:
                temp1 = filtered_sent[i-1]
                temp2 = filtered_sent[i]
                print(temp1 + " " + temp2)
                
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
    
    
    
    def classifier3(self, info):
        #info = self.clean_up(info)
        sentiment = self.sentiment_analyzer_scores(info)
        print(info + " " + str(sentiment))
        return(self.check_polar((sentiment)))
                   
def main():
    stock = "Amazon stock"
    data = NYTSentiment(stock)  
    print(data.total_percent)
main()
