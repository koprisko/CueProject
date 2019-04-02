# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:03:23 2019

API Key = https://api.nytimes.com/svc/search/v2/articlesearch.json?q=apple&api-key=Q41EjWTOsr2VA3y4UaXDqMmpWg6aNbDr
@author: kyleo
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re
import requests

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))


class Sentiment:
    def __init__(self, stock, ticker):
        key = "Q41EjWTOsr2VA3y4UaXDqMmpWg6aNbDr"
        info = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q=" + stock
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
                   
def main():
    stock = "Kellogg stock"
    ticker = "K"
    data = Sentiment(stock, ticker)  
    print(data.total_percent)
main()
