
# Twitter Sentiment Ananlysis

this repository implement sentiment analysis based on BERT Transformers on Twitter channels; by using function fetch_data(), you can get any coin from any twitter channels, and by filtering keywords, you can get relevant data from it,
by using preprocessing() function, we can preprocess text by using nltk methods, here function NLP() get sentiment and score of text., by applying calc_weighted() functions we can aggregate score of sentiment based on a daily basis,
Also, by applying weighted_average(), we can get the weighted average of sentiment based on replies_count, retweets_count, likes_count. I used a simple average here.
After that, by contacting crypto data and aggregated sentiment dataframe. We can see the relation between them; using the composition() function and setting different moving averages and shift lists, we can see the correlation coefficient between them and select the most appropriate one for our analysis.

<img src="https://user-images.githubusercontent.com/54494078/128258422-dd02ebea-0dc5-4f36-bf7d-5dcf06a2b132.PNG" width="800" height="600" align = 'center' >

correlation plot 

<img src="https://user-images.githubusercontent.com/54494078/128258666-3e938586-99de-4826-9f1d-7c8fdca8befe.PNG" width="800" height="600" align = 'center' >

here featureselection() applying RandomForest for selecting most relevant features with respet to target value. here result is 10 days moving average of sentiment.

