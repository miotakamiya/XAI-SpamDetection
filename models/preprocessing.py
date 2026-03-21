
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def preprocess_list(texts):
    return [preprocess(t) for t in texts]

def create_vectorizer():
    return TfidfVectorizer(max_features=5000)

def fit_transform(texts):
    texts = preprocess_list(texts)
    vectorizer = create_vectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def transform(texts, vectorizer):
    texts = preprocess_list(texts)
    return vectorizer.transform(texts)

def save_vectorizer(vectorizer, path="vectorizer.pkl"):
    joblib.dump(vectorizer, path)

def load_vectorizer(path="vectorizer.pkl"):
    return joblib.load(path)
