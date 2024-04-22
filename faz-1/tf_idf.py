import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Birleşik veri dosyasının yolu
file_path = 'data.xlsx'

# Hata kontrolü ve dosyanın varlığını kontrol etme
if os.path.exists(file_path):
    print("Dosya başarıyla açıldı. Veriler okunuyor...")

    # Excel dosyasını oku
    df = pd.read_excel(file_path)

    # TF-IDF vektörleştirici oluşturma
    tfidf_vectorizer = TfidfVectorizer()

    # TF-IDF matrisini oluşturma
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['reviewText'])

    # Duygu puanlarını hedef değişken olarak kullanma
    y = df['duygu']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)

    # Complement Naive Bayes modelini oluşturma ve eğitme
    CNB = ComplementNB()
    CNB.fit(X_train, y_train)

    # Gaussian Naive Bayes modelini oluşturma ve eğitme
    GNB = GaussianNB()
    GNB.fit(X_train.toarray(), y_train)  # GaussianNB, yoğun (dense) matris gerektirir

    # Bernoulli Naive Bayes modelini oluşturma ve eğitme
    BNB = BernoulliNB()
    BNB.fit(X_train, y_train)

    # Multinomial Naive Bayes modelini oluşturma ve eğitme
    MNB = MultinomialNB()
    MNB.fit(X_train, y_train)

    # Logistic Regression modelini oluşturma ve eğitme
    LR = LogisticRegression(max_iter=1000)
    LR.fit(X_train, y_train)

    # Decision Tree modelini oluşturma ve eğitme
    DT = DecisionTreeClassifier()
    DT.fit(X_train, y_train)

    # Random Forest modelini oluşturma ve eğitme
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    
    # Yapay Sinir Ağı modelini oluşturma ve eğitme
    ANN = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    ANN.fit(X_train, y_train)
    
    # SVM modelini oluşturma ve eğitme
    SVM = SVC()
    SVM.fit(X_train, y_train)

    # Eğitim setlerinde model performanslarını değerlendirme
    cnb_train_accuracy = CNB.score(X_train, y_train)
    gnb_train_accuracy = GNB.score(X_train.toarray(), y_train)  
    bnb_train_accuracy = BNB.score(X_train, y_train)
    mnb_train_accuracy = MNB.score(X_train, y_train)
    lr_train_accuracy = LR.score(X_train, y_train)
    dt_train_accuracy = DT.score(X_train, y_train)
    rf_train_accuracy = RF.score(X_train, y_train)
    ann_train_accuracy = ANN.score(X_train, y_train)
    svm_train_accuracy = SVM.score(X_train, y_train)

    # Test setlerinde model performanslarını değerlendirme
    cnb_test_accuracy = CNB.score(X_test, y_test)
    gnb_test_accuracy = GNB.score(X_test.toarray(), y_test)  
    bnb_test_accuracy = BNB.score(X_test, y_test)
    mnb_test_accuracy = MNB.score(X_test, y_test)
    lr_test_accuracy = LR.score(X_test, y_test)
    dt_test_accuracy = DT.score(X_test, y_test)
    rf_test_accuracy = RF.score(X_test, y_test)
    ann_test_accuracy = ANN.score(X_test, y_test)
    svm_test_accuracy = SVM.score(X_test, y_test)

    # Sonuçları yazdırma
    print("Complement Naive Bayes Eğitim Seti Accuracy:", cnb_train_accuracy)
    print("Gaussian Naive Bayes Eğitim Seti Accuracy:", gnb_train_accuracy)
    print("Bernoulli Naive Bayes Eğitim Seti Accuracy:", bnb_train_accuracy)
    print("Multinomial Naive Bayes Eğitim Seti Accuracy:", mnb_train_accuracy)
    print("Logistic Regression Eğitim Seti Accuracy:", lr_train_accuracy)
    print("Decision Tree Eğitim Seti Accuracy:", dt_train_accuracy)
    print("Random Forest Eğitim Seti Accuracy:", rf_train_accuracy)
    print("Yapay Sinir Ağı Eğitim Seti Accuracy:", ann_train_accuracy)
    print("\nComplement Naive Bayes Test Set Accuracy:", cnb_test_accuracy)
    print("Gaussian Naive Bayes Test Set Accuracy:", gnb_test_accuracy)
    print("Bernoulli Naive Bayes Test Set Accuracy:", bnb_test_accuracy)
    print("Multinomial Naive Bayes Test Set Accuracy:", mnb_test_accuracy)
    print("Logistic Regression Test Set Accuracy:", lr_test_accuracy)
    print("Decision Tree Test Set Accuracy:", dt_test_accuracy)
    print("Random Forest Test Set Accuracy:", rf_test_accuracy)
    print("Yapay Sinir Ağı Test Seti Accuracy:", ann_test_accuracy)
    print("SVM Eğitim Seti Accuracy:", svm_train_accuracy)
    print("SVM Test Seti Accuracy:", svm_test_accuracy)

    # Eğitim setlerinde sınıflandırma raporu
    print("\nComplement Naive Bayes Eğitim Seti Sınıflandırma Raporu:")
    cnb_train_pred = CNB.predict(X_train)
    print(classification_report(y_train, cnb_train_pred, zero_division=1))

    print("\nGaussian Naive Bayes Eğitim Seti Sınıflandırma Raporu:")
    gnb_train_pred = GNB.predict(X_train.toarray())  
    print(classification_report(y_train, gnb_train_pred, zero_division=1))

    print("\nBernoulli Naive Bayes Eğitim Seti Sınıflandırma Raporu:")
    bnb_train_pred = BNB.predict(X_train)
    print(classification_report(y_train, bnb_train_pred, zero_division=1))

    print("\nMultinomial Naive Bayes Eğitim Seti Sınıflandırma Raporu:")
    mnb_train_pred = MNB.predict(X_train)
    print(classification_report(y_train, mnb_train_pred, zero_division=1))

    print("\nLogistic Regression Eğitim Seti Sınıflandırma Raporu:")
    lr_train_pred = LR.predict(X_train)
    print(classification_report(y_train, lr_train_pred, zero_division=1))

    print("\nDecision Tree Eğitim Seti Sınıflandırma Raporu:")
    dt_train_pred = DT.predict(X_train)
    print(classification_report(y_train, dt_train_pred, zero_division=1))

    print("\nRandom Forest Eğitim Seti Sınıflandırma Raporu:")
    rf_train_pred = RF.predict(X_train)
    print(classification_report(y_train, rf_train_pred, zero_division=1))
    
    print("\nYapay Sinir Ağı Eğitim Seti Sınıflandırma Raporu:")
    ann_train_pred = ANN.predict(X_train)
    print(classification_report(y_train, ann_train_pred, zero_division=1))

    print("\nSVM Eğitim Seti Sınıflandırma Raporu:")
    svm_train_pred = SVM.predict(X_train)
    print(classification_report(y_train, svm_train_pred, zero_division=1))

    # Test setlerinde sınıflandırma raporu
    print("\nComplement Naive Bayes Test Seti Sınıflandırma Raporu:")
    cnb_test_pred = CNB.predict(X_test)
    print(classification_report(y_test, cnb_test_pred, zero_division=1))

    print("\nGaussian Naive Bayes Test Seti Sınıflandırma Raporu:")
    gnb_test_pred = GNB.predict(X_test.toarray()) 
    print(classification_report(y_test, gnb_test_pred, zero_division=1))

    print("\nBernoulli Naive Bayes Test Seti Sınıflandırma Raporu:")
    bnb_test_pred = BNB.predict(X_test)
    print(classification_report(y_test, bnb_test_pred, zero_division=1))

    print("\nMultinomial Naive Bayes Test Seti Sınıflandırma Raporu:")
    mnb_test_pred = MNB.predict(X_test)
    print(classification_report(y_test, mnb_test_pred, zero_division=1))

    print("\nLogistic Regression Test Seti Sınıflandırma Raporu:")
    lr_test_pred = LR.predict(X_test)
    print(classification_report(y_test, lr_test_pred, zero_division=1))

    print("\nDecision Tree Test Seti Sınıflandırma Raporu:")
    dt_test_pred = DT.predict(X_test)
    print(classification_report(y_test, dt_test_pred, zero_division=1))

    print("\nRandom Forest Test Seti Sınıflandırma Raporu:")
    rf_test_pred = RF.predict(X_test)
    print(classification_report(y_test, rf_test_pred, zero_division=1))
    
    print("\nYapay Sinir Ağı Test Seti Sınıflandırma Raporu:")
    ann_test_pred = ANN.predict(X_test)
    print(classification_report(y_test, ann_test_pred, zero_division=1))

    print("\nSVM Test Seti Sınıflandırma Raporu:")
    svm_test_pred = SVM.predict(X_test)
    print(classification_report(y_test, svm_test_pred, zero_division=1))

    # Tüm TF-IDF matrisini metin dosyasına kaydetme
    with open('tfidf_matrix.txt', 'w') as file:
        for row in tfidf_matrix.toarray():
            file.write(' '.join([str(elem) for elem in row]) + '\n')

    print("TF-IDF Matrisi başarıyla metin dosyasına kaydedildi.")

else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")
