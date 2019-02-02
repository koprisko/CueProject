import requests

API_KEY = 'Z2YB8LM6G3GEB0YJ'
r = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&outputsize=full&apikey=' + API_KEY)
if (r.status_code == 200):
  print r.json()
  result = r.json()
  dataForAllDays = result['Time Series (Daily)']
  dataForSingleDate = dataForAllDays['2010-05-25']
  symbol = result['Meta Data']['2. Symbol']
  print symbol
  print dataForSingleDate['1. open']
  print dataForSingleDate['2. high']
  print dataForSingleDate['3. low']
  print dataForSingleDate['4. close']
  print dataForSingleDate['5. volume']    
