import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Excel dosyalarını oku
kelimeler = pd.read_excel("unique_words.xlsx", header=None)
metinler = pd.read_excel("preprocessed_dataset.xlsx", header=None)

# Kelime listesini oluştur
kelime_listesi = kelimeler[0].tolist()

# TF-IDF vektörleyicisini (TfidfVectorizer) oluştur
tfidf_vectorizer = TfidfVectorizer(vocabulary=kelime_listesi, use_idf=True, smooth_idf=True)

# Metinlerdeki kelimelerin TF-IDF matrisini oluştur
tfidf_matrix = tfidf_vectorizer.fit_transform(metinler[0])

# TF-IDF matrisini yazdır
print(tfidf_matrix)
