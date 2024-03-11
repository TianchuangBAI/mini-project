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

# 读取数据
df = pd.read_csv('reviews_bsample.csv')

# 下载nltk数据
nltk.download('stopwords')
nltk.download('punkt')

# 初始化词干提取器和停用词列表
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# 预处理文本
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

# 使用TF-IDF向量化文本
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['review_pros'])
y = df['job_category']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练朴素贝叶斯分类器
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# 预测
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# 使用卡方检验计算特征与目标变量之间的相关性
chi2_score, p_value = chi2(X, y)

feature_names = tfidf_vectorizer.get_feature_names_out()


# 获取与收入/工资/薪水最相关的主题
target_topic_index = np.argmax(chi2_score)
target_topic = tfidf_vectorizer.get_feature_names_out()[target_topic_index]
print('Topic most correlated with income/wage/salary:', target_topic)

sorted_indices = np.argsort(chi2_score)[::-1]
sorted_features = [feature_names[idx] for idx in sorted_indices]
sorted_scores = chi2_score[sorted_indices]

top_features = sorted_features[:20]
top_scores = sorted_scores[:20]

plt.figure(figsize=(10, 8))
plt.barh(top_features, top_scores, color='skyblue')
plt.xlabel('Chi-squared score')
plt.ylabel('Top Features')
plt.title('Top 20 Features by Chi-squared score')
plt.show()
