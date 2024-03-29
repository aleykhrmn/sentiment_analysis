import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Veri setlerini yükleme
kelimeler = pd.read_excel("unique_words.xlsx")
metinler = pd.read_excel("preprocessed_dataset.xlsx")
yildizlar = pd.read_excel("overall.xlsx")

# Matris boyutlarını belirleme
sutun = kelimeler.shape[0]
satir = metinler.shape[0]

# Tüm verileri temsil etmek için bir numpy matrisi oluşturma
all_data = np.zeros((satir, sutun + 1))

# Veri hazırlığı için kelimeler listesini oluşturma
kelime_listesi = kelimeler.iloc[:, 0].tolist()

# Metin verilerini ve duygu puanlarını numpy matrisine dönüştürme
for i in range(satir):
    k = metinler.iloc[i, 0].split(" ")
    for j in range(len(k)):
        if k[j] in kelime_listesi:
            all_data[i, kelime_listesi.index(k[j])] = 1
    all_data[i, -1] = yildizlar.iloc[i, 0]

# Veri kümesini eğitim ve test setlerine bölmek
X_train, X_test, y_train, y_test = train_test_split(all_data[:, :-1], all_data[:, -1], test_size=0.2, random_state=42)

# Logistic Regression modelini oluşturmak ve eğitmek
model = LogisticRegression()
model.fit(X_train, y_train)

# Modeli test verisi üzerinde değerlendirmek
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
