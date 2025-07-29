# Project: Phân Tích Cảm Xúc Review Phim (Sentiment Analysis)

Đây là dự án cuối kỳ cho môn học Lập trình cho AI, thực hiện bởi nhóm BTL 01. Mục tiêu của dự án là xây dựng một mô hình Machine Learning có khả năng phân loại các đoạn văn bản review phim thành hai nhãn: **Tích cực (Positive)** và **Tiêu cực (Negative)**.

---

## Mục Lục
* [Công Cụ](#công-cụ)
* [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
* [Cài Đặt & Hướng Dẫn Sử Dụng](#cài-đặt--hướng-dẫn-sử-dụng)
* [Cách Chạy Dự Án](#cách-chạy-dự-án)
* [Kết Quả](#kết-quả)
* [Hướng Phát Triển](#hướng-phát-triển)
* [Thành Viên Nhóm](#thành-viên-nhóm)

---

## Công Cụ
Dự án được xây dựng chủ yếu bằng ngôn ngữ Python và các thư viện Khoa học Dữ liệu phổ biến:

*   **Ngôn ngữ:** `Python 3.x`
*   **Thao tác dữ liệu:** `Pandas`
*   **Xử lý ngôn ngữ tự nhiên (NLP):** `NLTK`, `Scikit-learn`
*   **Trực quan hóa dữ liệu:** `Matplotlib`, `Seaborn`, `WordCloud`
*   **Mô hình Machine Learning:** `Scikit-learn` (Logistic Regression, Naive Bayes, SVM)
*   **Quản lý phiên bản:** `Git` & `GitHub`

---

## Cấu Trúc Thư Mục
Dự án được tổ chức theo một cấu trúc chuyên nghiệp để đảm bảo tính module hóa và dễ bảo trì:

```
Project-Sentiment-Analysis/
├── data/
│   └── (Chứa file IMDB Dataset.csv sau khi tải về)
├── models/
│   └── (Chứa file model .pkl sau khi huấn luyện)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   └── pipeline.py
├── .gitignore
├── README.md
├── requirements.txt
├── run_eda.py
└── train.py
```

*   **`data/`**: Thư mục chứa dữ liệu thô. (Thư mục này không được đẩy lên GitHub).
*   **`models/`**: Thư mục lưu trữ các mô hình đã được huấn luyện.
*   **`src/`**: Chứa toàn bộ source code của dự án, được chia thành các module logic.
*   **`run_eda.py`**: Script để chạy quá trình Phân tích Khám phá Dữ liệu (EDA) và tạo các biểu đồ.
*   **`train.py`**: Script chính để huấn luyện, đánh giá và lưu mô hình.
*   **`requirements.txt`**: File chứa danh sách các thư viện cần thiết để chạy dự án.

---

## Cài Đặt & Hướng Dẫn Sử Dụng
Để chạy dự án trên máy của bạn, vui lòng làm theo các bước sau:

**1. Clone Repository**
```bash
git clone https://github.com/PPIG2204/Project-Sentiment-Analysis.git
cd Project-Sentiment-Analysis
```

**2. Tạo Môi Trường Ảo (Khuyến khích)**
```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

**3. Cài Đặt Các Thư Viện Cần Thiết**
Chạy lệnh sau để cài đặt tất cả các gói trong file `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Lưu ý: Nếu bạn chưa có file `requirements.txt`, hãy tạo nó bằng lệnh: `pip freeze > requirements.txt`)*

**4. Tải Dữ Liệu**
Dự án sử dụng bộ dữ liệu IMDb Dataset từ Kaggle.
*   **Cách 1 (Thủ công):**
    1.  Tải dữ liệu từ link sau: [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
    2.  Sau khi tải về, giải nén và đặt file `IMDB Dataset.csv` vào thư mục `data/`.

*   **Cách 2 (Tự động - Sắp tới):** Chạy script để tự động tải dữ liệu (cần cài đặt Kaggle API).

---

## Cách Chạy Dự Án

*   **Để xem các phân tích và biểu đồ trực quan:**
    ```bash
    python run_eda.py
    ```
*   **Để huấn luyện mô hình từ đầu:**
    ```bash
    python train.py
    ```
    Sau khi chạy, mô hình sẽ được lưu vào thư mục `models/`.

---

## Kết Quả
Sau quá trình huấn luyện và đánh giá trên tập kiểm tra (test set), mô hình **Logistic Regression** đã đạt được kết quả rất tốt:

| Chỉ số       | Giá trị  |
|--------------|----------|
| **Accuracy** | ~89.7%   |
| **F1-Score (Positive)** | 0.90     |
| **F1-Score (Negative)** | 0.89     |

*(Các chỉ số này có thể được xem chi tiết hơn khi chạy file `train.py`)*

---

## Hướng Phát Triển
Dự án có thể được cải thiện và mở rộng trong tương lai với các hướng sau:
*   [ ] Tinh chỉnh siêu tham số (Hyperparameter Tuning) bằng `GridSearchCV` để tối ưu hóa mô hình.
*   [ ] Thử nghiệm các mô hình Deep Learning như LSTM, GRU.
*   [ ] Sử dụng các mô hình ngôn ngữ lớn đã được huấn luyện trước (pre-trained models) như BERT để cải thiện độ chính xác.
*   [ ] Xây dựng một giao diện web đơn giản bằng `Streamlit` hoặc `Flask` để người dùng có thể demo sản phẩm.

---

## Thành Viên 
*   **Nguyễn Đức Hoàng Nam** - [24105787] - Link GitHub của Nam: https://github.com/NamNguyen-phenikaa
*   **Nguyễn Trương Phước** - [24100153] - Link GitHub của Phước: https://github.com/PPIG2204
