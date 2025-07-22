from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Dữ liệu văn bản của chúng ta
documents = [
    "Scikit-learn là một thư viện học máy tuyệt vời.",
    "Học máy là một lĩnh vực rất thú vị và có nhiều ứng dụng.",
    "Python là ngôn ngữ phổ biến cho học máy và khoa học dữ liệu.",
    "Scikit-learn hỗ trợ nhiều thuật toán học máy khác nhau.",
]

# 1. Khởi tạo TfidfVectorizer
# Bạn có thể điều chỉnh các tham số như stop_words (loại bỏ các từ phổ biến như 'là', 'và')
# Nếu bạn làm việc với tiếng Việt, bạn có thể cần một danh sách stop_words tiếng Việt.
# Hiện tại, chúng ta sẽ để mặc định hoặc chỉ ra stop_words của tiếng Anh nếu muốn thử.
# Với tiếng Việt, tốt nhất là tự định nghĩa stop_words hoặc bỏ qua tham số này nếu chỉ thử nghiệm.
vectorizer = TfidfVectorizer()

# 2. Học từ vựng và biến đổi tài liệu thành ma trận TF-IDF
# .fit_transform() thực hiện hai việc:
#    - fit: Học tất cả các từ duy nhất trong 'documents' và gán cho chúng một chỉ số.
#    - transform: Tính toán giá trị TF-IDF cho mỗi từ trong mỗi tài liệu.
tfidf_matrix = vectorizer.fit_transform(documents)

# 3. Xem kết quả (tùy chọn)

# Lấy danh sách các từ (tên đặc trưng/feature names)
feature_names = vectorizer.get_feature_names_out()

# Chuyển ma trận TF-IDF sang DataFrame của pandas để dễ nhìn hơn
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print("Ma trận TF-IDF:")
print(df_tfidf)

# Bạn có thể xem hình dạng của ma trận (số tài liệu, số từ duy nhất)
print(f"\nKích thước ma trận TF-IDF: {tfidf_matrix.shape}")
print(f"Có {tfidf_matrix.shape[0]} tài liệu và {tfidf_matrix.shape[1]} từ/đặc trưng duy nhất.")