from matplotlib import pyplot as plt
import pandas as pd
df = pd.read_csv('reviews_bsample.csv')
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sns
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['review_pros'])
y = df['job_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = MultinomialNB()
# 定义参数网格
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
    'fit_prior': [True, False]
}

# 使用网格搜索和交叉验证
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)

# 使用最佳参数的分类器进行预测
best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy with best parameters:', accuracy)

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_classifier.classes_, yticklabels=best_classifier.classes_)
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.title('Confusion Matrix')
plt.show()

# 输出分类报告
class_report = classification_report(y_test, y_pred, target_names=best_classifier.classes_)
print(class_report)
