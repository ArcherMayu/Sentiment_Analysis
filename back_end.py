#!/usr/bin/env python
# coding: utf-8



# In[ ]:


import matplotlib.pyplot as plt
import nltk
import seaborn as sns

nltk.download('stopwords')
from matplotlib import style

style.use('ggplot')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from sklearn.metrics import accuracy_score, classification_report
import requests

import pandas as pd
import re
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk

nltk.download('punkt')

from sklearn.svm import LinearSVC


# In[ ]:


def fetch_data(term):  # f1
    twitter_data = []
    for a in range(0, 4):
        payload = {'api_key': '97c277c1add29b16733cd1aeaebaafe4', 'query': term, 'num': '80', 'page': a}
        response = requests.get('https://api.scraperapi.com/structured/twitter/search', params=payload)
        data = response.json()
        all_tweets = data["organic_results"]
        for tweet in all_tweets:
            twitter_data.append(
                {'ID': tweet['position'], 'User': tweet["title"], 'Tweet': tweet["snippet"], 'URL': tweet["link"]})
        return twitter_data


def data_processing(text):  # f2
    text = text.lower()
    text = re.sub(r"http\S+|www\S+https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


stemmer = PorterStemmer()


def stemming(text):  # f3
    text = text.lower()
    text = re.sub(r"http\S+|www\S+https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    text = " ".join(filtered_text)
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)


def polarity(text):  # f4
    return TextBlob(text).sentiment.polarity


def sentiment(label):  # f5
    if label == 0:
        return 'Neutral'
    elif label < 0:
        return 'Negative'
    elif label > 0:
        return 'Positive'


# In[ ]:


def analyze(term):  # function to call
    twitter_data = fetch_data(term)

    df = pd.DataFrame(twitter_data)
    df.to_json('scraped_tweets.json', orient='index')

    # df.head()

    # df.shape

    stemmer = PorterStemmer()

    # Perform text preprocessing on the training data
    stop_words = set(stopwords.words('english'))
    df['Tweet'] = df['Tweet'].apply(data_processing)
    df['Tweet'] = df['Tweet'].apply(stemming)

    # df.head()

    df = df.drop('URL', axis='columns')

    df.head()

    df['polarity'] = df['Tweet'].apply(polarity)

    # df.head(10)

    df['Sentiment'] = df['polarity'].apply(sentiment)

    # df.head(10)
    # df.tail(10)

    # df['Sentiment'].values

    fig = plt.figure(figsize=(10, 5))

    sns.countplot(x='Sentiment', data=df)
    # save the plot as png
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Count")
    plt.savefig("static/countplot.png")

    vect = CountVectorizer(ngram_range=(1, 2)).fit(df['Tweet'])
    x = df['Tweet']
    y = df['Sentiment']
    x = vect.transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4, random_state=42)
    xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

    logreg = LogisticRegression()
    logreg.fit(xtrain, ytrain)
    logreg_pred = logreg.predict(xtest)
    logreg_acc = accuracy_score(logreg_pred, ytest)

    accuracy = logreg_acc * 100  # return

    SVCmodel = LinearSVC()
    SVCmodel.fit(xtrain, ytrain)

    svc_pred = SVCmodel.predict(xtest)
    svc_acc = accuracy_score(svc_pred, ytest)
    # print(f"Accuracy: {svc_acc}")

    # print(confusion_matrix(ytest, svc_pred))
    print(classification_report(ytest, svc_pred))

    # count positive neutral and negative data
    count_1 = df['Sentiment'].value_counts()['Positive']  # positive data
    count_2 = df['Sentiment'].value_counts()['Neutral']  # neutral data
    count_3 = df['Sentiment'].value_counts()['Negative']  # negative data

    report = []
    try:
        report_str = classification_report(ytest, svc_pred, target_names=['Negative', 'Neutral', 'Positive'])  # return
    except:
        report_str = 0
    report.append(accuracy)
    report.append(report_str)
    report.append(count_1)
    report.append(count_2)
    report.append(count_3)

    return report

# In[ ]:
