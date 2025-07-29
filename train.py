# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib # Dùng để lưu và tải model
import os
import nltk

from src import config
from src import preprocessing
from src import pipeline as pl

def run_training():
    """Hàm chính để chạy toàn bộ quá trình huấn luyện."""
    print("Bắt đầu quá trình huấn luyện...")
    
    # 1. Tải dữ liệu
    df = pd.read_csv(config.DATA_PATH)
    df.dropna(subset=['review', 'sentiment'], inplace=True)
    
    # 2. Làm sạch dữ liệu
    print("Đang làm sạch dữ liệu...")
    df['clean_review'] = df['review'].apply(preprocessing.clean_text)
    
    # 3. Chuẩn bị dữ liệu và chia tập train/test
    X = df['clean_review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"Dữ liệu đã được chia: {len(X_train)} mẫu train, {len(X_test)} mẫu test.")
    
    # 4. Huấn luyện pipeline
    print("Đang huấn luyện mô hình Logistic Regression...")
    pl.sentiment_pipeline.fit(X_train, y_train)
    print("Huấn luyện hoàn tất.")
    
    # 5. Đánh giá mô hình
    y_pred = pl.sentiment_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy trên tập test: {accuracy:.4f}")
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_test, y_pred))
    
    # 6. Lưu mô hình
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(pl.sentiment_pipeline, config.MODEL_PATH)
    print(f"\nMô hình đã được lưu tại: {config.MODEL_PATH}")

if __name__ == "__main__":
    # Tải các gói NLTK cần thiết
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    run_training()