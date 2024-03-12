import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


nltk.download('stopwords')
nltk.download('punkt')

file_path = 'reviews_bsample.csv'  
data = pd.read_csv(file_path)

def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = ' '.join([word for word in text.split() if not word.startswith("http")])
    text = ' '.join([word for word in text.split() if not word.startswith("#")])
    text = ' '.join([word for word in text.split() if not word.startswith("@")])
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

data['review_summary'] = data['review_summary'].apply(preprocess_text)
data['review_advice'] = data['review_advice'].apply(preprocess_text)
data['review_pros'] = data['review_pros'].apply(preprocess_text)
data['review_cons'] = data['review_cons'].apply(preprocess_text)

processed_file_path = 'processed_reviews.csv'  
data.to_csv(processed_file_path, index=False)
