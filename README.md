# Sentiment Classification from Tweeter feed about US airlines

## Used [Tweets dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) from Kaggle

- Developed using Scala with Spark MlLib library with pipelines for preprocessing, estimating and cross validation
- Deployed in AWS EMR Cluster as an [executable jar](https://github.com/chatterjeesubhajit/Twitter-Feed--Sentiment-Classification/blob/main/tweets_2.11-0.1.jar) file  
- Application takes input file path and output directory path as arguments
- Performs preprocessing like Tokenization, Stop words removal, Term Hashing
- Uses Logistic Regression Classifier with Parameter Grid and Cross validation to evaluate best model parameters 
