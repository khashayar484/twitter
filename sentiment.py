
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.window.rolling import Window
from sklearn import tree
import twint
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import nltk
from datetime import datetime, timedelta
from nltk import word_tokenize
import string
# from transformers import pipeline
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from binance.client import Client

def fetch_date(channel , mention , coin_name  , order = 1000 , save = True ,  since = 0):
  '''
  you can fetch twitter data with this function
  by save = True you can save file to .csv format
  '''
  c = twint.Config()

  c.Username = channel

  query = mention.replace(',' , ' OR ')

  c.Search = query

  if since !=0:
    c.Since = since
  else:
    c.Limit = order

  c.Store_csv = True
    
  twint.output.clean_lists()

  if save : c.Output = f"twit_{channel}_{coin_name}.csv"
  twint.run.Search(c)

def preprocessing(dataframe):
  df = dataframe.copy(deep = True)

  string.punctuation = string.punctuation + '➡️' + '—' + '$' + ',' + '#' + '💔' + '.' + '“”' 
  def punctuation_cleaning(text):
    text = re.sub(r'@[A-Za-z0-9]+' , '' , text)  # remove @mention
    text = re.sub(r'RT[\s]+' , '' , text)
    text = re.sub(r'https?:\/\/\S+' , '' , text) # remove the hyper link
    text= text.replace('"', '')                  # remove doble quotes
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text

  df['twt'] = df['tweet'].apply(punctuation_cleaning)

  ## convert text to lower case
  df['lower_txt'] = df['twt'].apply(lambda x : x.lower())

  ## remove punctuation 
  df['text_punc'] =  df['lower_txt'].apply(lambda x : "".join([char for char in x if char not in string.punctuation]) )

  ## tokenization
  stop_words = stopwords.words('english')
  df['stop_words'] = df['text_punc'].apply(lambda x : [char for char in x.split(' ') if char not in stop_words])

  ## remove stop words
  df['converter'] = df['stop_words'].apply(lambda x : " ".join([char for char in x]))

  ## stemming
  porter = PorterStemmer()
  df['stem'] = df['converter'].apply(lambda x : [porter.stem(word) for word in x.split(' ')])

  ## concatenating them
  df['out'] = df['stem'].apply(lambda x : " ".join([char for char in x]))

  result = df[['date' , 'tweet' , 'converter' , 'out' , 'replies_count' , 'retweets_count' , 'likes_count']] 
  result = result[::-1]

  return result

def NLP(dataframe , column_name):

  dp = dataframe.copy(deep = True)
  sentiments , scores = [] , []

  for i in range(0 , len(dp)):

      classifier = pipeline('sentiment-analysis')
      sentiment , score = list(classifier(dp.iloc[i][column_name])[0].values())[0] , list(classifier(dp.iloc[i][column_name])[0].values())[1]

      if i % 20 == 0 :
        print('--------------> here we are in idnex ' , i)
        print('tweet is ' , dp.iloc[i][column_name])
        print('sentiment is ' , sentiment)
        print('score is ' , score )
        print('---------------------------------------')

      sentiments.append(sentiment)
      scores.append(score)

  dp['sentiment'] = np.array(sentiments)
  dp['score'] = np.array(scores)

  return dp[['date' , 'sentiment' , 'score' , 'tweet' , column_name , 'replies_count' , 'retweets_count' , 'likes_count']]


def calc_weighted(dataframe):
    train_set = dataframe.copy(deep = True)
    train_set['datetime'] = pd.to_datetime(train_set['date'])
    first_date, last_date = train_set['datetime'].iloc[0] , train_set['datetime'].iloc[-1]
    train_set['sentiment_amount'] = train_set['sentiment'].apply(lambda x : -1 if x == 'NEGATIVE' else 1)
    train_set['mark'] = train_set['sentiment_amount'] * train_set['score']
    index, marks , dates , daily_mark , aggregate_daily_money = 0, [] , [] , [] , pd.DataFrame()
    for date, mark in zip(train_set['datetime'] , train_set['mark']):
        if date == first_date:
            marks.append(mark)
            if date == last_date and index == len(train_set) - 1:
                print('------------> for last_date ' ,  date , ' we have score ' , np.array(marks).sum()) 
                dates.append(date)
                daily_mark.append(np.array(marks).sum())
        else:
            dates.append(first_date)
            daily_mark.append(np.array(marks).sum())
            print('------------> for day ' ,  first_date , ' we have score ' , np.array(marks).sum()) 
            first_date = date
            marks = []
            marks.append(mark)
        index +=1
    aggregate_daily_money['daily_money'] = np.array(daily_mark)
    aggregate_daily_money['date'] = np.array(dates)
    print(aggregate_daily_money)

    return aggregate_daily_money

def weighted_average(dataframe , weighted_by):
  '''
  use weighted average by number of likes, number of retweet or number of replies
  '''
  train_set = dataframe.copy(deep = True)
  train_set['datetime'] = pd.to_datetime(train_set['date'])
  first_date, last_date = train_set['datetime'].iloc[0] , train_set['datetime'].iloc[-1]
  train_set['sentiment_amount'] = train_set['sentiment'].apply(lambda x : -1 if x == 'NEGATIVE' else 1)
  train_set['mark'] = train_set['sentiment_amount'] * train_set['score']
  index, marks, dates, daily_mark, likes, aggregate_daily_money = 0, [] ,[] ,[], [] , pd.DataFrame()
  for date, mark ,like in zip(train_set['datetime'] , train_set['mark'] , train_set[weighted_by]):
    if date == first_date:
      marks.append(mark)
      likes.append(like)
      if date == last_date and index == len(train_set) - 1:
        weighted_av = sum(x*y for x,y in zip(marks , likes)) / sum(likes)
        print('------------> for last_date ' ,  date , ' we have score ' , weighted_av)
        dates.append(date)
        daily_mark.append(weighted_av)
    else:
      dates.append(first_date)
      weighted_av = sum(x*y for x,y in zip(marks , likes)) / sum(likes)
      first_date = date
      print('------------> for day ' ,  first_date , ' we have score ' , weighted_av) 
      marks , likes = [] , []
      daily_mark.append(weighted_av), likes.append(like) , marks.append(mark)
    index +=1

  aggregate_daily_money['daily_money'] = np.array(daily_mark)
  aggregate_daily_money['date'] = np.array(dates)
  print(aggregate_daily_money)

  return aggregate_daily_money

def get_coins(coin  , start , end , save = False ):
  client = Client(api_key, api_secret)
  one = client.get_historical_klines(f'{coin}USDT' ,  Client.KLINE_INTERVAL_1DAY  ,start_str = start , end_str = end)

  df = pd.DataFrame(one)
  df.columns = ['Open time','Open','High','Low','Close','Volume' , 'Close time' , 'Quote asset volume' , 'Number of trades' ,\
                                          'Taker buy base asset volume' , 'Taker buy quote asset volume' , 'ignore']

  df['date'] = pd.to_datetime(df['Open time'] , unit= 'ms')
  df = df.set_index('Open time')

  df = df.iloc[:-1 , :]
  if save :   df.to_excel(f'{coin}.xlsx')

  return df

## -----------------------> TODO : add ploting both simple and cumulative plot 

def composition(list_ma , list_shift , dataframe , columns_list):  ## TODO : maybe by using EWMA we can get better result
  df = dataframe[columns_list]
  dt = pd.DataFrame()
  for ma in list_ma:
      dt[f'ma_{ma}'] = df['wma_news'].rolling(window = ma).mean()
      dt[f'cum_ma_{ma}'] = dt[f'ma_{ma}'] / dt[f'ma_{ma}'].iloc[ma]
      print('dt[ma] \n ' ,dt[f'cum_ma_{ma}'] )
      for shift in list_shift:
          dt[f'cum_ma{ma}_shift_{shift}'] = dt[f'cum_ma_{ma}'].shift(shift)

  dt['cum_Close'] = df['Close'] / df['Close'].iloc[0]
  print('--------> total is' , dt)

  return dt

def feature_selection(df , cut_off):
  '''
  selected_important features : 
  '''
  dataframe = df.dropna()
  model = RandomForestRegressor(n_estimators= 1500 , criterion='mse' , max_depth=10)
  #   model = XGBClassifier()
  x_train , y_train = dataframe.drop(columns = 'cum_Close') , dataframe[['cum_Close']]

  print('-----> X_train shape is ' , x_train.shape ,  ' y_train shape is ' , y_train.shape)
  sc = StandardScaler()
  y_train_scale = sc.fit_transform(y_train)
  model.fit(x_train , y_train_scale.ravel()) 

  importance = model.feature_importances_

  print('-------------------------> total features is ' , importance)
  scores , cols_index   = [] , []
  print('------------------------->  scoring')
  for num,score in enumerate(importance):
      # print('Feature: %0d, Score: %.5f' % (num,score))
      if score > cut_off:
          scores.append(score)
          cols_index.append(num)

  imp_df = x_train.iloc[: , cols_index]
  print('-------------- > imp_df columsn is  ' , imp_df.columns)
  print('-------------- > important score is ' , scores)
  imp_df = pd.concat([y_train , imp_df] , axis = 1)

  return imp_df


# df = pd.read_excel('composition.xlsx' , index_col = 0)


def OLS(dataframe):
  '''
  fit ols line to importance features from PCA and use p_values and t_test
  '''
  df = dataframe.dropna()
  print('-----------> first \n  ' , df)
#   df = ((df.shift(5) / df) -1 )*100
#   df = df.dropna()
  print('-----------> secend df is \n' , df)
  x,y = df.drop(columns = 'cum_Close') , df[['cum_Close']]
  x = sm.add_constant(x)
  results = sm.OLS(y , x).fit()
  print('----------> OLS result \n ' , results.summary())

# OLS(dataframe= pca_df)

## --------------> TODO : sentiment.xlsx must reverse 

def main():
  fetch_date(channel = 'BTCTN' , mention='bitcoin,BTC' , coin_name='BTC' , since = '2021-01-01')
  btc = pd.read_csv('twit_BTCTN_BTC.csv' , index_col = 0 )
  words_btc = preprocessing(btc)
  words_btc.to_excel('BTCTN_words_preprocessing.xlsx')
  df = pd.read_excel('BTCTN_words_preprocessing.xlsx' , index_col = 0)
  df = df.iloc[1000: , :]
  sentiment_btc = NLP(dataframe = df , column_name = 'converter')
  sentiment_btc.to_excel('sentiment_btc.xlsx')
  df = pd.read_excel('sentiment_btc.xlsx')
  df = df[::-1]
  aggregate = calc_weighted(df)
  print(aggregate)
  start_date , end_date = aggregate['date'].iloc[0].strftime("%Y-%m-%d") , aggregate['date'].iloc[-1].strftime("%Y-%m-%d")
  print('-----------> start date is ' , start_date , ' end s date is ' , end_date)
  get_coins(coin  = 'BTC' , start = start_date , end  = end_date)
