import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Tải các dữ liệu cần thiết
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Đọc dữ liệu
df = pd.read_csv("IMDB Dataset.csv")
df.dropna(subset=['review', 'sentiment'], inplace=True)

# Stopwords mở rộng
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words.union({
    'movie', 'film', 'one', 'would', 'like', 'really', 'also', 'even',
    'get', 'see', 'much', 'could', 'story', 'characters', 'time',
    'good', 'bad', 'br', 'everything', 'nothing'
})

# Khởi tạo công cụ NLP
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')  # tách từ theo chữ cái/số

# Hàm làm sạch văn bản
def clean_text(text):
    text = re.sub('<.*?>', '', text)  # Xoá thẻ HTML
    text = text.lower()  # Chuyển về chữ thường
    tokens = tokenizer.tokenize(text)  # Tách từ (tokenize)
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in custom_stopwords and re.fullmatch(r'[a-z]+', word)
    ]
    return ' '.join(filtered_tokens)

# Áp dụng làm sạch
df['clean_review'] = df['review'].apply(clean_text)
df['review_length'] = df['clean_review'].apply(lambda x: len(x.split()))

# Thống kê sơ bộ
print(df['sentiment'].value_counts())
print(df['review_length'].describe())

# Ghép từ theo sentiment
positive_text = ' '.join(df[df['sentiment'] == 'positive']['clean_review'])
negative_text = ' '.join(df[df['sentiment'] == 'negative']['clean_review'])

# WordCloud
positive_wc = WordCloud(width=1000, height=500, max_words=100,
                        background_color='white', stopwords=custom_stopwords,
                        collocations=False).generate(positive_text)

negative_wc = WordCloud(width=1000, height=500, max_words=100,
                        background_color='black', stopwords=custom_stopwords,
                        collocations=False, colormap='Reds').generate(negative_text)

# Trực quan hóa
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

sns.countplot(x='sentiment', data=df, ax=axs[0, 0])
axs[0, 0].set_title("Phân phối nhãn sentiment", fontsize=14)

sns.histplot(df['review_length'], bins=50, kde=True, ax=axs[0, 1])
axs[0, 1].set_title("Độ dài review (số từ)", fontsize=14)

axs[1, 0].imshow(positive_wc, interpolation='bilinear')
axs[1, 0].axis('off')
axs[1, 0].set_title("Word Cloud - Positive", fontsize=14)

axs[1, 1].imshow(negative_wc, interpolation='bilinear')
axs[1, 1].axis('off')
axs[1, 1].set_title("Word Cloud - Negative", fontsize=14)

plt.tight_layout()
plt.show()

#vector hoa van ban
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_bow = vectorizer.fit_transform(df['clean_review'])
print(X_bow.toarray())
#dung trong TH muon xem 1 vai du lieu ban dau
"""
print(X_bow[:5].toarray())
print(X_bow.shape)
"""
print(vectorizer.get_feature_names_out())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score # Example metric

# Assuming 'X' is your text data and 'y' is your labels
# 1. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. & 3. Sử dụng Pipeline để gộp Vectorizer và Model
# Naive Bayes Pipeline
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb_model', MultinomialNB())
])
nb_pipeline.fit(X_train, y_train)
nb_predictions = nb_pipeline.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_predictions)}")

# Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr_model', LogisticRegression(max_iter=1000)) # max_iter often needed for convergence
])
lr_pipeline.fit(X_train, y_train)
lr_predictions = lr_pipeline.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_predictions)}")

# SVM Pipeline
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm_model', SVC())
])
svm_pipeline.fit(X_train, y_train)
svm_predictions = svm_pipeline.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, svm_predictions)}")