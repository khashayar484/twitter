composition# Twitter Sentiment Ananlysis

this repositpry implemet sentiment analysis which based on BERT Transformers on twitter channels, by using function fetch_data(), you can get any coin from any twitter channels and by filtering keywords you can get relevant data from it,
here by using prepreprocessing() function we can preprocessing text by using nltk methods, here function NLP() get sentiment and score of text., by applying calc_weighted() functions we can aggregate score of sentiment based on daily basis,
also by applying weighted_average() we can get weighted average of sentiment based on replies_count, retweets_count, likes_count. i used  simple average here.
afetr that by concating crypto data and aggreagted sentiment dataframe.

for seeing the relation between them by using function composition() and creating different version of data for seeing correlation coeffiecent of them.

