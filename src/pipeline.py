# src/pipeline.py

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Định nghĩa pipeline
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr_model', LogisticRegression(max_iter=1000))
])