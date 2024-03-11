import pandas as pd
df = pd.read_csv('reviews_bsample.csv')
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isnull(text):
        text = ''
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 去除停用词和词干化
    tokens = [ps.stem(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

df['review_pros'] = df['review_pros'].apply(preprocess_text)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classfication_report

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['review_pros'])
y = df['job_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

print(classification_report(y_test, y_pred))
