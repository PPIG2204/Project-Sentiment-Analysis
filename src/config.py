# src/config.py

from nltk.corpus import stopwords

# --- ĐƯỜNG DẪN FILE ---
DATA_PATH = "data/IMDB Dataset.csv"
MODEL_PATH = "models/logistic_regression_v1.pkl" # .pkl là định dạng phổ biến để lưu model sklearn

# --- CÁC THAM SỐ TIỀN XỬ LÝ ---
# Những từ này không mang nhiều ý nghĩa cảm xúc trong bối cảnh review phim
CUSTOM_STOPWORDS = set(stopwords.words('english')).union({
    'movie', 'film', 'one', 'would', 'like', 'really', 'also', 'even',
    'get', 'see', 'much', 'could', 'story', 'characters', 'time',
    'good', 'bad', 'br', 'everything', 'nothing'
})

# --- CÁC THAM SỐ MÔ HÌNH ---
RANDOM_STATE = 42
TEST_SIZE = 0.2