# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:04:22 2019

@author: kyleo
"""

import datetime
import re
import requests

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))


"""This class is the class responsible for making calls to the article apis that we found. They are New York Times, Yahoo Finance,
 and Stock News API. The data is cleaned and sentiment values are appended to the date that the article found was published. You can access the sentiment values
 by calling the sentiment using the appropriate paramters of a stock and its ticker symbol, and then referencing the object created .cleaned_data
 This data is used to feed to our csv_creator and eventually gets fed into the nerual network"""

class Sentiment:
    def __init__(self, stock, ticker):
        
        self.full_articles = [] #used to return articles information text to the main class
        self.cleaned_data = [] #used to store the sentiments values and dates of all of the receieved sentiments
        
        
        #this block initializes the url syntax nexessary for making a call to the New york times api
        key = "Q41EjWTOsr2VA3y4UaXDqMmpWg6aNbDr"
        info = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q=" + stock
        info = info + "&subject:Stocks and Bonds&api-key=" + key
        r = requests.get(info)
        r.encoding = 'utf-8'
        
        #the text returned by the nyt api is a json file that has had some incorrect formating
        #It gets sent to the collector method so that the data can be gathered
        data = self.collector(r.text)  #gets returned the sentiment score and data
        
        new_dates = self.fix_year(data) # dates get properly formatted and returned
        
        count = 0 #used to index the array
        for i in data:
            self.cleaned_data.append(tuple((new_dates[count],i[1]))) #a tuple is created containing the date and sentiment value
            count = count + 1
        
        
        #used to call the yahoo finance api. The return is an xml file format that gets sent to the yahoo collector to have its data collected
        r = requests.get("https://feeds.finance.yahoo.com/rss/2.0/headline?s=" + ticker + "&region=US&lang=en-US")
        r.encoding = 'utf-8'
        info2 = r.text
        data2 = self.yahooCollector(info2) #gets returned a list of tuples with dates and sentiment values
        
        new_dates = self.fix_year2(data2) # dates get properly formatted and returned
        
        count = 0 # used to index the list
        for i in data2:
            self.cleaned_data.append(tuple((new_dates[count],i[1])))
            count = count + 1
        
        
        #This block makes a call to stock news api the return is a json file
        url = "https://stocknewsapi.com/api/v1?tickers=" + ticker + "&items=30&fallback=true&token=oroav5z0e7ov2ohmszaggk4a9pqutz3gacvvjfvo"        
        response = requests.get(url)
        response.encoding = 'utf-8'
        response = response.json() #converts the json to a dictionary
            
        
        if 'data' in response:  # makes sure the response was valid and we didnt overuse the api
            for r in response['data']:      #allows for us to see all of the data
                article = r['title'] + " " + r['text'] #adds the two parts of the article together instead of summing them later
                date = r['date'][4:16].strip() # gets the data tagged date and strips the information in the date that we dont need
                dt = datetime.datetime.strptime(date, "%d %b %Y") # strips the date information in the syntax that it is currently in
                
                #allows for us to create a date that easily matches our histoircial data 
                date_string = ""
                date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year) 
                
                 #adds the date and sentiment value to a list with the other article information
                self.cleaned_data.append(tuple((date_string, self.classifier(article))))
        
        
    def fix_year(self, date):
        holder = []
        #goes through all of the dates and puts them into the same syntax that we use for histoical information
        for d in date:
            dt = datetime.datetime.strptime(d[0], "%Y-%m-%d")
            date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year)
            holder.append(date_string)
        return holder   #returns the cleaned dates
    
    def fix_year2(self, date):
        holder = []
        #goes through all of the dates and puts them into the same syntax that we use for histoical information
        for d in date:
            temp = d[0].strip()
            dt = datetime.datetime.strptime(temp, "%d %b %Y")
            date_string = str(dt.month) + "-" +  str(dt.day) + "-" +  str(dt.year)
            holder.append(date_string)
        return holder  #returns the cleaned dates
        
    def clean_yahoo_Dates(self, info):
            heads = []
            #used to strip unimportant date infomration from the date string
            for i in info:
                heads.append(i[13:-25])
            return heads #return a list of cleaned dates
        
        
    def yahooCollector(self, data):
        
        #this datta was not needed so we to it out so that it did not get picked up as a article title
        #we use a regular expression and substitute method to accomplish all of our parsing of this data
        irrelevant = re.compile(r'<title>Yahoo! Finance:.* News</title>')
        data = re.sub(irrelevant, "", data)
        
        #this datta was not needed so we to it out so that it did not get picked up as a article description
        irrelevant = re.compile(r'<description>Latest Financial News for.*</description>')
        data = re.sub(irrelevant, "", data)
        
        #regex to find all of the titles
        title = re.compile(r'<title>.*</title>')
        titles = title.findall(data)
        
        #regex to find all descriptions
        description = re.compile(r'<description>.*</description>')
        descriptions = description.findall(data)
        
        #finds all of the dates
        date = re.compile(r'<pubDate>.*</pubDate>')
        dates = date.findall(data)
        
        
        
        
        dates = self.clean_yahoo_Dates(dates) #cleans off of the pub date tags and returns a array of dates
        
        good = True
        sentiment_array = []
        count = -1 # needed so that we can index the dates when a new title is found
        while good == True:
            polarity = 0 # used to keep track of the sentiment value
            
            #if we found a title then we do the proper cleaning of the data by getting rid of unneeded syntax that 
            #was returned by the xml file
            if titles:
                count = count + 1 #increment count
                temp = titles.pop().replace('<title>', "").replace('</title>', "")
                temp = self.clean_up(temp) # used to get rid of unnesary words and punctuation
                temp = temp.replace("amp", "").replace("quot", "")
                Sa = self.classifier(temp) #stores the sentiment value to be summed the call to classifier did not work in the summation
                polarity = Sa + polarity
            #if we found a description then we do the proper cleaning of the data by getting rid of unneeded syntax that 
            #was returned by the xml file   
            elif descriptions:
                temp = descriptions.pop().replace('<description>', "").replace('</description>', "")
                temp = self.clean_up(temp) # used to get rid of unnesary words and punctuation
                temp = temp.replace("amp", "").replace("quot", "")
                Sa = self.classifier(temp)
                polarity = polarity + Sa #stores the sentiment value to be summed the call to classifier did not work in the summation
            else:
                good = False # a boolean for inconsistent date is thrown when proper info is not returned
            #if this iterations data was proper then we append the data to a list to be read in the main class
            if good == True:
                sentiment_array.append(tuple((dates[count],polarity))) #includes the dates and sentiment values
        return sentiment_array
        


    """These next functions are used to cut parts of the returned data that we do not want to givwe to
    The sentiment analyzer. the return is an array of cleaned data for the new york times data"""
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
            
    
            
    def collector(self, stud): # stud has all of the reponse text from the http call
                                              
        head_Line = re.compile(r'"print_headline":"[^"]+","') #creates the regex to find headlines
        headers = head_Line.findall(stud)                #puts all of the headlines into an array
        stud = re.sub(head_Line, "", stud)               #takes out all of the headlines from the response text so we dont grab it by chance
        headers  = self.clean_headlines(headers)         #gets rid of the unnessary text in the headlines
        
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
         
        
    # this is used for yahoos articles because yahoo returns some chars that have no meaning
    # this is only used for yahoo because it returns random symbols sometimes and they are caught here
    def clean_up(self, info):
        tokenized_word=word_tokenize(info)
        filtered_sent= ""
        for w in tokenized_word: #goes through ever word and makes sure it does not have punctuation in the middle. like "&9amp"
            if w not in stop_words and w.isalpha():
                filtered_sent = filtered_sent + " " + w
        return(filtered_sent)
            
    def sentiment_analyzer_scores(self, sentence):
        #initializes a new sentiment analyzer and sends it the sentiment score
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(sentence) #return is a list of keys
        return(score['compound']) #returns the compound score in the list
     
    def classifier(self, info):
        sentiment = self.sentiment_analyzer_scores(info) #sends the article to be analyzed
        return(sentiment)
