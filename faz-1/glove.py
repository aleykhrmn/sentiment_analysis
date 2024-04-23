import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Birleşik veri dosyasının yolu
file_path = 'data.xlsx'

# Hata kontrolü ve dosyanın varlığını kontrol etme
if os.path.exists(file_path):
    print("Dosya başarıyla açıldı. Veriler okunuyor...")

    # Excel dosyasını oku
    df = pd.read_excel(file_path)

    # GloVe vektörlerinin yolu
    glove_path = 'glove.6B.300d.txt'

    # GloVe vektörlerini yükleyin
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Metin verilerini GloVe vektörleriyle temsil etme
    embedding_dim = 300

    X = np.zeros((len(df), embedding_dim))
    for i, text in enumerate(df['reviewText']):
        words = text.split()
        vec = np.zeros(embedding_dim)
        for word in words:
            if word in embeddings_index:
                vec += embeddings_index[word]
        X[i] = vec

    # Duygu puanlarını hedef değişken olarak kullanma
    y = df['duygu']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    LR = LogisticRegression(max_iter=1000)
    LR.fit(X_train, y_train)

    # Support Vector Machine
    SVM = SVC()
    SVM.fit(X_train, y_train)

    # Decision Tree
    DT = DecisionTreeClassifier()
    DT.fit(X_train, y_train)

    # Random Forest
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)

    # Eğitim setlerinde model performanslarını değerlendirme
    lr_train_accuracy = LR.score(X_train, y_train)
    svm_train_accuracy = SVM.score(X_train, y_train)
    dt_train_accuracy = DT.score(X_train, y_train)
    rf_train_accuracy = RF.score(X_train, y_train)

    # Test setlerinde model performanslarını değerlendirme
    lr_test_accuracy = LR.score(X_test, y_test)
    svm_test_accuracy = SVM.score(X_test, y_test)
    dt_test_accuracy = DT.score(X_test, y_test)
    rf_test_accuracy = RF.score(X_test, y_test)

    # Sonuçları yazdırma
    print("Logistic Regression Eğitim Seti Accuracy:", lr_train_accuracy)
    print("Support Vector Machine Eğitim Seti Accuracy:", svm_train_accuracy)
    print("Decision Tree Eğitim Seti Accuracy:", dt_train_accuracy)
    print("Random Forest Eğitim Seti Accuracy:", rf_train_accuracy)

    print("\nLogistic Regression Test Set Accuracy:", lr_test_accuracy)
    print("Support Vector Machine Test Set Accuracy:", svm_test_accuracy)
    print("Decision Tree Test Set Accuracy:", dt_test_accuracy)
    print("Random Forest Test Set Accuracy:", rf_test_accuracy)

    # Eğitim setlerinde sınıflandırma raporu
    print("\nLogistic Regression Eğitim Seti Sınıflandırma Raporu:")
    lr_train_pred = LR.predict(X_train)
    print(classification_report(y_train, lr_train_pred, zero_division=1))

    print("\nSupport Vector Machine Eğitim Seti Sınıflandırma Raporu:")
    svm_train_pred = SVM.predict(X_train)
    print(classification_report(y_train, svm_train_pred, zero_division=1))

    print("\nDecision Tree Eğitim Seti Sınıflandırma Raporu:")
    dt_train_pred = DT.predict(X_train)
    print(classification_report(y_train, dt_train_pred, zero_division=1))

    print("\nRandom Forest Eğitim Seti Sınıflandırma Raporu:")
    rf_train_pred = RF.predict(X_train)
    print(classification_report(y_train, rf_train_pred, zero_division=1))

    # Test setlerinde sınıflandırma raporu
    print("\nLogistic Regression Test Seti Sınıflandırma Raporu:")
    lr_test_pred = LR.predict(X_test)
    print(classification_report(y_test, lr_test_pred, zero_division=1))

    print("\nSupport Vector Machine Test Seti Sınıflandırma Raporu:")
    svm_test_pred = SVM.predict(X_test)
    print(classification_report(y_test, svm_test_pred, zero_division=1))

    print("\nDecision Tree Test Seti Sınıflandırma Raporu:")
    dt_test_pred = DT.predict(X_test)
    print(classification_report(y_test, dt_test_pred, zero_division=1))

    print("\nRandom Forest Test Seti Sınıflandırma Raporu:")
    rf_test_pred = RF.predict(X_test)
    print(classification_report(y_test, rf_test_pred, zero_division=1))

else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")

