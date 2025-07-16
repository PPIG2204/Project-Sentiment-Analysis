import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import re

# Tải stopwords nếu chưa có
nltk.download('stopwords')
from nltk.corpus import stopwords

# Đọc file CSV (cách đơn giản)
df = pd.read_csv("IMDB Dataset.csv")

# Tùy chỉnh stopwords mở rộng
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words.union({
    'movie', 'film', 'one', 'would', 'like', 'really', 'also', 'even', 'get',
    'see', 'much', 'could', 'story', 'characters', 'time', 'good', 'bad'
})

# Hàm làm sạch
def clean_text(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return ' '.join([word for word in text.split() if word not in custom_stopwords])

# Làm sạch và tính độ dài
df['clean_review'] = df['review'].apply(clean_text)
df['review_length'] = df['review'].apply(lambda x: len(x.split()))

# Tách theo sentiment
positive_text = ' '.join(df[df['sentiment'] == 'positive']['clean_review'])
negative_text = ' '.join(df[df['sentiment'] == 'negative']['clean_review'])

# WordCloud tối ưu
positive_wc = WordCloud(width=1000, height=500, max_words=100,
                        background_color='white', stopwords=custom_stopwords,
                        collocations=False).generate(positive_text)

negative_wc = WordCloud(width=1000, height=500, max_words=100,
                        background_color='black', stopwords=custom_stopwords,
                        collocations=False, colormap='Reds').generate(negative_text)

# Trực quan hóa
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Phân phối nhãn
sns.countplot(x='sentiment', data=df, ax=axs[0, 0])
axs[0, 0].set_title("Phân phối nhãn sentiment", fontsize=14)

# Độ dài review
sns.histplot(df['review_length'], bins=50, kde=True, ax=axs[0, 1])
axs[0, 1].set_title("Độ dài review (số từ)", fontsize=14)

# WordCloud Positive
axs[1, 0].imshow(positive_wc, interpolation='bilinear')
axs[1, 0].axis('off')
axs[1, 0].set_title("Word Cloud - Positive", fontsize=14)

# WordCloud Negative
axs[1, 1].imshow(negative_wc, interpolation='bilinear')
axs[1, 1].axis('off')
axs[1, 1].set_title("Word Cloud - Negative", fontsize=14)

plt.tight_layout()
plt.show()
