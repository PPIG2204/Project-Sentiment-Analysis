# src/preprocessing.py

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from . import config # Dấu "." để import từ cùng package 'src'

# Khởi tạo một lần duy nhất để tiết kiệm tài nguyên
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

def clean_text(text):
    """Hàm làm sạch văn bản thô."""
    # Xoá thẻ HTML
    text = re.sub('<.*?>', '', text)
    # Chuyển về chữ thường
    text = text.lower()
    # Tách từ (tokenize)
    tokens = tokenizer.tokenize(text)
    # Loại bỏ stopwords, lemmatize và chỉ giữ lại từ chứa chữ cái
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in config.CUSTOM_STOPWORDS and re.fullmatch(r'[a-z]+', word)
    ]
    return ' '.join(filtered_tokens)