import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Excel dosyalarını oku
kelimeler = pd.read_excel("unique_words.xlsx", header=None)
metinler = pd.read_excel("preprocessed_dataset.xlsx", header=None)
etiketler = pd.read_excel("overall.xlsx", header=None)  # Başlıkları dahil etme

# Metin ve etiketleri birleştir
data = pd.concat([metinler, etiketler], axis=1)

# TF-IDF vektörleyicisini oluştur
tfidf_vectorizer = TfidfVectorizer(vocabulary=kelimeler[0].tolist(), use_idf=True, smooth_idf=True, lowercase=False)

# Metinlerdeki kelimelerin TF-IDF matrisini oluştur
tfidf_matrix = tfidf_vectorizer.fit_transform(data.iloc[:, 0])  # Metinler ilk sütunda olduğu için

# Eğitim ve test veri kümelerini ayır
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, data.iloc[:, 1], test_size=0.2, random_state=42)

# Naive Bayes sınıflandırıcısını oluştur ve eğit
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)

# Test veri kümesi üzerinde sınıflandırıcıyı değerlendir
y_pred = naive_bayes_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)