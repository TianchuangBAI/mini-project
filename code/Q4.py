import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('reviews_bsample.csv')
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isnull(text):
        text = ''
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def analyze_text_column(column_name):
    df[column_name] = df[column_name].apply(preprocess_text)
    X = tfidf_vectorizer.fit_transform(df[column_name])
    y = df['job_category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for {column_name}:', accuracy)

    chi2_score, p_value = chi2(X, y)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    target_topic_index = np.argmax(chi2_score)
    target_topic = feature_names[target_topic_index]
    print(f'Topic most correlated with {column_name}:', target_topic)

    sorted_indices = np.argsort(chi2_score)[::-1]
    sorted_features = [feature_names[idx] for idx in sorted_indices]
    sorted_scores = chi2_score[sorted_indices]

    top_features = sorted_features[:20]
    top_scores = sorted_scores[:20]

    plt.figure(figsize=(10, 8))
    plt.barh(top_features, top_scores, color='skyblue')
    plt.xlabel('Chi-squared score')
    plt.ylabel('Top Features')
    plt.title(f'Top 20 Features by Chi-squared score for {column_name}')

    plt.savefig(f'{column_name}_analysis.png')
    plt.show()

tfidf_vectorizer = TfidfVectorizer()

text_columns_to_analyze = ['review_summary', 'review_advice', 'review_pros', 'review_cons']
for column in text_columns_to_analyze:
    analyze_text_column(column)
