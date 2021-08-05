
# Twitter Sentiment Ananlysis

this repository implement sentiment analysis based on BERT Transformers on Twitter channels; by using function fetch_data(), you can get any coin from any twitter channels, and by filtering keywords, you can get relevant data from it, by using preprocessing() function, we can preprocess text by using nltk methods, here function NLP() get sentiment and score of text., by applying calc_weighted() functions we can aggregate score of sentiment based on a daily basis, Also, by applying weighted_average(), we can get the weighted average of sentiment based on replies_count, retweets_count, likes_count. I used a simple average here.

After that, by contacting crypto data and aggregated sentiment dataframe. We can see the relation between them; using the composition() function and setting different moving averages and shift lists, we can see the correlation coefficient between them and select the most appropriate one for our analysis.

### bitcoin close price and different moving average of sentiments.
<img src="https://user-images.githubusercontent.com/54494078/128258422-dd02ebea-0dc5-4f36-bf7d-5dcf06a2b132.PNG" width="800" height="600" align = 'center' >

### correlation plot 

<img src="https://user-images.githubusercontent.com/54494078/128258666-3e938586-99de-4826-9f1d-7c8fdca8befe.PNG" width="800" height="600" align = 'center' >

As you can see from the above picture, close price of bitcoin and ten days moving average of sentiment has 0.64 correlation coeffiecient and has 0.69 correlation coeffiecient by 20 days moving averages which shifted 20 days.

i use featureselection() function which applying RandomForest for selecting most relevant features with respet to target value. here result is 10 days moving average of sentiment.
i use this function for getting the most relevant feauters. at the end by using the linear_regression() we can get the coeffiencts of linear regression between them. the result is.

<img src="https://user-images.githubusercontent.com/54494078/128300271-1444439d-29cf-4496-8a92-e3e056f588f4.PNG" width="800" height="600" align = 'center' >
 
As you can see, the R_squre is 0.4 here, which means we can capture 40% of the variation of Closing price by ten days moving average of daily sentiments. And also, the coefficient of sentiment is 0.63 and is also significant by the p_value measure, but the intercept is not. 

## contribution

the are several things we can do:
use another indicator besides the sentiment like volume, hash rate, difficulty, etc.
Use another model than linear regression.
Use a longer time frame than here.
Use portfolio of crypto and check the relation to cryptocurrency sentiment analysis
use better text preprocessing before use transformers.
