from io import UnsupportedOperation
from operator import index, pos
from os import stat
from matplotlib import scale
from numpy.core.defchararray import lower
from numpy.lib.function_base import insert
from numpy.lib.type_check import real
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.tile import cut
from pandas.core.window.rolling import Window
from scipy.sparse import data
from seaborn.categorical import pointplot
from seaborn.rcmod import reset_defaults
from sklearn import tree
from sklearn.utils import indices_to_mask
import twint
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import nltk


def fetch_date(channel_list , mention , coin_name  , order = 100 , save = False ,  since = 0):
  '''
  you can fetch twitter data with this function

  by save = True you can save file to .csv format
  '''
  for channel in channel_list:

    print('--------------------> channel is ' , channel)

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
    df = dataframe
    print('--------------> convert text to lower case ')
    df['lower_txt'] = df['twt'].apply(lambda x : x.lower())
    print(df[['twt' , 'lower_txt']])

    import string
    print(string.punctuation)


    print('---------------> remove punctuation')
    df['text_p'] =  df['lower_txt'].apply(lambda x : "".join([char for char in x if char not in string.punctuation]) )

    print(df[['lower_txt' , 'text_p']])

    print('---------------> tokenization ')
    from nltk import word_tokenize
    # nltk.download( )
    df['token_p'] = df['text_p'].apply(lambda x : word_tokenize(x))

    print(df[['text_p' , 'token_p']])

    print('--------------> stop words filtering')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    print('-------------> stop words corpus is \n'  ,stop_words)

    df['stop_p'] = df['token_p'].apply(lambda x : [char for char in x if char not in stop_words])
    print(df[['token_p' , 'stop_p']])

    print('--------------------------> stemming ')
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    df['stem'] = df['stop_p'].apply(lambda x : [porter.stem(word) for word in x ])
    print(df[['stop_p' , 'stem']])

    df['stem'].to_excel('stem.xlsx')
    print('-----> done go to hugging face')

    link = 'https://colab.research.google.com/drive/1oJVDUcGZ-CxUr2-lU0xAVIZ61mrY3dQH'

    ## ----------------------> using Huging Face BERT ! 

def NLP(dataframe):
    dp = dataframe

    sentiments , scores = [] , []
    for i in range(0 , len(dp)):
        print('--------------> here we are in idnex ' , i)
        print(dp.iloc[i , 0])
        classifier = pipeline('sentiment-analysis')
        
        sentiment , score = list(classifier(dp.iloc[i]['stem'])[0].values())[0] , list(classifier(dp.iloc[i]['stem'])[0].values())[1]
        print(classifier(dp.iloc[i]['stem']))
        print('tweet is ' , dp.iloc[i]['stem'])
        print('sentiment is ' , sentiment)
        print('score is ' , score )

        print('---------------------------------------')
        sentiments.append(sentiment)
        scores.append(score)

    dp['sentiment'] = np.array(sentiments)
    dp['score'] = np.array(scores)

    print(dp)

    # NLP(dataframe = df)


def SMA():
    pass   ## TODO : use whale tracker

def WMA():
    pass

def get_coin():
    pass