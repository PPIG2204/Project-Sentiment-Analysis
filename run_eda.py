# run_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from src import config
from src import preprocessing

print("Bắt đầu quá trình Phân tích Khám phá Dữ liệu (EDA)...")

# 1. Đọc dữ liệu
df = pd.read_csv(config.DATA_PATH)
df.dropna(subset=['review', 'sentiment'], inplace=True)

# 2. Áp dụng làm sạch (chỉ cần cho WordCloud và phân tích độ dài)
print("Làm sạch dữ liệu để phân tích...")
df['clean_review'] = df['review'].apply(preprocessing.clean_text)
df['review_length'] = df['clean_review'].apply(lambda x: len(x.split()))

# 3. Thống kê sơ bộ
print("\nPhân phối các nhãn:")
print(df['sentiment'].value_counts())
print("\nThống kê độ dài review (sau khi làm sạch):")
print(df['review_length'].describe())

# 4. Chuẩn bị text cho WordCloud
positive_text = ' '.join(df[df['sentiment'] == 'positive']['clean_review'])
negative_text = ' '.join(df[df['sentiment'] == 'negative']['clean_review'])

# 5. Tạo WordCloud
print("Đang tạo WordCloud...")
positive_wc = WordCloud(width=1000, height=500, max_words=100,
                        background_color='white', stopwords=config.CUSTOM_STOPWORDS,
                        collocations=False).generate(positive_text)

negative_wc = WordCloud(width=1000, height=500, max_words=100,
                        background_color='black', stopwords=config.CUSTOM_STOPWORDS,
                        colormap='Reds', collocations=False).generate(negative_text)

# 6. Trực quan hóa tất cả
print("Đang vẽ biểu đồ...")
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Phân Tích Khám Phá Dữ Liệu IMDb", fontsize=20)

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

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nEDA hoàn tất!")