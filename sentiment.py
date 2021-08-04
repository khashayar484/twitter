
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import twint
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import nltk
from datetime import datetime, timedelta
import string
# from transformers import pipeline
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from binance.client import Client
import seaborn as sns

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
  '''
  return : 'date' , 'sentiment' , 'score' , 'tweet' , column_name , 'replies_count' , 'retweets_count' , 'likes_count'
  '''

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
                dates.append(date)
                daily_mark.append(np.array(marks).sum())
        else:
            dates.append(first_date)
            daily_mark.append(np.array(marks).sum())
            first_date = date
            marks = []
            marks.append(mark)
        index +=1
    aggregate_daily_money['daily_average_sentiment_news'] = np.array(daily_mark)
    aggregate_daily_money['date'] = np.array(dates)

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
  index, marks, dates, daily_mark, likes, aggregated_twit = 0, [] ,[] ,[], [] , pd.DataFrame()
  for date, mark ,like in zip(train_set['datetime'] , train_set['mark'] , train_set[weighted_by]):
    if date == first_date:
      marks.append(mark)
      likes.append(like)
      if date == last_date and index == len(train_set) - 1:
        weighted_av = sum(x*y for x,y in zip(marks , likes)) / sum(likes)
        dates.append(date)
        daily_mark.append(weighted_av)
    else:
      dates.append(first_date)
      weighted_av = sum(x*y for x,y in zip(marks , likes)) / sum(likes)
      first_date = date
      marks , likes = [] , []
      daily_mark.append(weighted_av), likes.append(like) , marks.append(mark)
    index +=1

  aggregated_twit[f'weighed_by_{weighted_by}'] = np.array(daily_mark)
  aggregated_twit['date'] = np.array(dates)

  return aggregated_twit

def get_coins(coin  , start , end , save = False ):
  '''
  here i get coins data from binance api, before it you need to register on binance.com and get api_key
  and api_secret to do this.
  '''
  # client = Client(api_key, api_secret)
  client = Client('UUH1nlQxWwMGVyQJysJcWeYZf9SdgfDzLJOldIC0QVrioFnqXRhdZKMZEh6hJGkx', 'Kwg2hIDyidwNt9BuBL22TazN2LrDDBq0bqzuQHzD1lmpgW6sLAQN8Ov2FWhyz4v3') 
  one = client.get_historical_klines(f'{coin}USDT' ,  Client.KLINE_INTERVAL_1DAY  ,start_str = start , end_str = end)

  df = pd.DataFrame(one)
  df.columns = ['Open time','Open','High','Low','Close','Volume' , 'Close time' , 'Quote asset volume' , 'Number of trades' ,\
                                          'Taker buy base asset volume' , 'Taker buy quote asset volume' , 'ignore']

  df['date'] = pd.to_datetime(df['Open time'] , unit= 'ms')
  df = df.set_index('Open time')

  df = df.iloc[:-1 , :]
  if save :   df.to_excel(f'{coin}.xlsx')

  return df

def composition(list_ma , list_shift , dataframe , corr_plot = False , pyplot = False ,  mode = 'cumulative'):  
  '''
  Create different implementations and see the correlation coefficient,
  quoted_column: wma_close, sma_news
  list_ma: list of moving averages
  mode: cumulative, moving_average
  '''
  df = dataframe.copy(deep = True)
  quoted_column = df.columns[1]
  composition_df, plot_df = pd.DataFrame(), pd.DataFrame()

  if mode == 'cumulative':
    composition_df['cum_Close'] = df['Close'] / df['Close'].iloc[0]
    for ma in list_ma:
        composition_df[f'ma_{ma}'] = composition_df[quoted_column].rolling(window = ma).mean()
        composition_df[f'cum_ma_{ma}'] = composition_df[f'ma_{ma}'] / composition_df[f'ma_{ma}'].iloc[ma]
        if pyplot: plot_df[f'cum_ma_{ma}'] = composition_df[f'cum_ma_{ma}'] 
        for shift in list_shift:
            composition_df[f'cum_ma{ma}_shift_{shift}'] = composition_df[f'cum_ma_{ma}'].shift(shift)
  
  if mode == 'moving_average':
    composition_df['Close'] = df['Close']
    for ma in list_ma:
      composition_df[f'ma_{ma}'] = df[quoted_column].rolling(window = ma).mean()
      if pyplot: plot_df[f'ma_{ma}'] = composition_df[f'ma_{ma}']
      for shift in list_shift:
        composition_df[f'ma_{ma}_shift_{shift}'] = composition_df[f'ma_{ma}'].shift(shift)
    
  if pyplot == True:
    sc = StandardScaler()
    plot_df['Close'] = df['Close']
    scale = sc.fit_transform(plot_df)
    scale = pd.DataFrame(scale, columns = plot_df.columns)
    for col in  scale.columns:
      plt.plot(scale[col] , label = col)
    plt.legend()
    plt.show()

  if corr_plot: 
    mn = MinMaxScaler()
    composition_df = composition_df.dropna()
    scale = mn.fit_transform(composition_df)
    scale = pd.DataFrame(scale, columns = composition_df.columns)
    corr = scale.corr()
    sns.heatmap(corr, annot=True , cmap= 'Blues')
    plt.title('--- correlation ---')
    plt.tight_layout()
    plt.show()

  return composition_df

def feature_selection(df , target_name , cut_off , mode = 'RandomForest'):
  '''
  selected_important features : select most important features with respect to target value
  this function apply Random
  '''
  dataframe = df.dropna()
  model = RandomForestRegressor(n_estimators= 1500 , criterion='mse' , max_depth=10)
  x_train , y_train = dataframe.drop(columns = target_name) , dataframe[[target_name]]

  print('-----> X_train shape is ' , x_train.shape ,  ' y_train shape is ' , y_train.shape)
  sc = StandardScaler()
  y_train_scale = sc.fit_transform(y_train)
  model.fit(x_train , y_train_scale.ravel()) 
  importance = model.feature_importances_
  scores , cols_index   = [] , []
  for num,score in enumerate(importance):
      # print('Feature: %0d, Score: %.5f' % (num,score))
      if score > cut_off:
          scores.append(score)
          cols_index.append(num)

  imp_df = x_train.iloc[: , cols_index]
  max_score, arg_max = max(scores), np.argmax(scores)
  most_imp = x_train[[imp_df.columns[arg_max]]]

  imp_df = pd.concat([y_train , imp_df] , axis = 1)
  most_imp = pd.concat([y_train , most_imp] , axis =1 )

  return imp_df, most_imp


def linear_regression(dataframe, target_column):
  df = dataframe.dropna()
  sc = StandardScaler()
  scale = sc.fit_transform(df)
  scale = pd.DataFrame(scale , columns= df.columns)
  x,y = scale.drop(columns = target_column) , scale[[target_column]]
  x = sm.add_constant(x)
  results = sm.OLS(y , x).fit()
  print('-----------> OLS result \n ' , results.summary())
  ## ----------> OLS on return 
  df = df.astype(float)
  pct = df.pct_change().dropna()
  x_r, y_r = pct.drop(columns = target_column) , pct[[target_column]]
  x_r = sm.add_constant(x_r)
  print('------> x_r shape ' , x_r.shape , ' y_r shape is ' , y_r.shape)
  results = sm.OLS(y_r , x_r).fit()
  print('-----------> OLS result based on return \n ' , results.summary())


def main(channel_name , coin , search_keywords , plot = False):
  # fetch_date(channel = channel_name, mention = search_keywords, coin_name = coin , since = '2021-01-01')
  # twit_df = pd.read_csv(f'twit_{channel_name}_{coin}.csv' , index_col = 0)
  # preprocess_df = preprocessing(twit_df)
  # sentiment_btc = NLP(dataframe = preprocess_df , column_name = 'converter')
  sentiment_btc = pd.read_excel('sentiment.xlsx')
  sentiment_btc = sentiment_btc[::-1]
  aggregated_twit =   weighted_average(sentiment_btc, weighted_by='retweets_count')
  start_date , end_date = aggregated_twit['date'].iloc[0].strftime("%Y-%m-%d") , aggregated_twit['date'].iloc[-1].strftime("%Y-%m-%d")
  coin_date = get_coins(coin  = coin , start = start_date , end  = end_date)
  aggregated_twit, coin_date = aggregated_twit.set_index('date'), coin_date.set_index('date')
  concatenate = pd.concat([coin_date[['Close']], aggregated_twit] , axis = 1)
  print(concatenate)
  ## you can also make different compositions of dataframe and see the correlation coefficient
  range_of_dataframe = composition(list_ma=[5,10,20] , list_shift=[5,10,20] , dataframe=concatenate, corr_plot=True, pyplot=True ,mode = 'moving_average' )
  features = feature_selection(df = range_of_dataframe, target_name='Close', cut_off=0.1)
  linear_regression(dataframe=features[1], target_column='Close')


main(channel_name= 'BTCTN' , coin = 'BTC' , search_keywords= 'bitcoin,BTC' , plot = True)