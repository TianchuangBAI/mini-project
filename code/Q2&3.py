from matplotlib import pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

def train_and_evaluate(df, column_name):
    df[column_name] = df[column_name].apply(preprocess_text)
    X = TfidfVectorizer().fit_transform(df[column_name])
    y = df['job_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    nb_classifier = MultinomialNB()
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'fit_prior': [True, False]
    }

    grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters for {column_name}:", grid_search.best_params_)

    # Use the best parameters classifier for prediction
    best_classifier = grid_search.best_estimator_
    y_pred = best_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy with best parameters for {column_name}:', accuracy)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_classifier.classes_, yticklabels=best_classifier.classes_)
    plt.xlabel('Predicted value')
    plt.ylabel('Actual value')
    plt.title(f'Confusion Matrix for {column_name}')
    plt.savefig(f'confusion_matrix_{column_name}.png')  # Save confusion matrix as an image
    plt.close()

    class_report = classification_report(y_test, y_pred, target_names=best_classifier.classes_)
    print(class_report)

    with open(f'classification_report_{column_name}.txt', 'w') as f:
        f.write(class_report)

df = pd.read_csv('reviews_bsample.csv')

columns_to_predict = ['review_summary', 'review_advice', 'review_pros', 'review_cons']
for column in columns_to_predict:
    train_and_evaluate(df, column)
