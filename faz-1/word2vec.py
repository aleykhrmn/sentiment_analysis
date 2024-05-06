import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Birleşik veri dosyasının yolu
file_path = 'data.xlsx'

# Hata kontrolü ve dosyanın varlığını kontrol etme
if os.path.exists(file_path):
    print("Dosya başarıyla açıldı. Veriler okunuyor...")

    # Excel dosyasını oku
    df = pd.read_excel(file_path)

    # Word2Vec modelini eğitme
    documents = [review.split() for review in df['reviewText']]
    model = Word2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)

    # Metin belgelerini vektörleştirme
    X = np.array([np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(100)], axis=0) for doc in documents])
    y = df['duygu']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verilerin negatif değerlerini ele almak için Min-Max ölçeklendirme
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelleri oluşturma ve eğitme

    # Bernoulli Naive Bayes modelini oluşturun
    BNB = BernoulliNB()
    BNB.fit(X_train_scaled, y_train)

    # Complement Naive Bayes modelini oluşturun
    CNB = ComplementNB()
    CNB.fit(X_train_scaled, y_train)

    # Gaussian Naive Bayes modelini oluşturun
    GNB = GaussianNB()
    GNB.fit(X_train_scaled, y_train)
    
    # Naive Bayes modelini oluşturma ve eğitme
    nb_model = MultinomialNB()
    nb_model.fit(X_train_scaled, y_train)
    
    # Decision Tree modelini oluşturun
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train_scaled, y_train)

    # Lojistik Regresyon modelini oluşturun
    log_reg_model = LogisticRegression(max_iter=1000)  # Artırılmış iterasyon limiti
    log_reg_model.fit(X_train_scaled, y_train)

    # Random Forest modelini oluşturun
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Yapay Sinir Ağı modelini oluşturma ve eğitme
    ANN = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    ANN.fit(X_train, y_train)
    
    # SVM modelini oluşturma ve eğitme
    svm_model = SVC()  
    svm_model.fit(X_train_scaled, y_train)

    # Performans metriklerini hesaplama

    # Decision Tree için performans metrikleri
    dt_pred = dt_model.predict(X_test_scaled)
    dt_accuracy = metrics.accuracy_score(dt_pred, y_test)
    dt_train_pred = dt_model.predict(X_train_scaled)
    dt_train_accuracy = metrics.accuracy_score(dt_train_pred, y_train)

    # Lojistik Regresyon için performans metrikleri
    log_reg_pred = log_reg_model.predict(X_test_scaled)
    log_reg_accuracy = metrics.accuracy_score(log_reg_pred, y_test)
    log_reg_train_pred = log_reg_model.predict(X_train_scaled)
    log_reg_train_accuracy = metrics.accuracy_score(log_reg_train_pred, y_train)

    # Random Forest için performans metrikleri
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = metrics.accuracy_score(rf_pred, y_test)
    rf_train_pred = rf_model.predict(X_train_scaled)
    rf_train_accuracy = metrics.accuracy_score(rf_train_pred, y_train)

    # SVM için performans metrikleri
    svm_pred = svm_model.predict(X_test_scaled)
    svm_accuracy = metrics.accuracy_score(svm_pred, y_test)
    svm_train_pred = svm_model.predict(X_train_scaled)
    svm_train_accuracy = metrics.accuracy_score(svm_train_pred, y_train)

    ann_train_accuracy = ANN.score(X_train, y_train)
    ann_test_accuracy = ANN.score(X_test, y_test)
    
    # Bernoulli Naive Bayes için performans metrikleri
    bnb_pred = BNB.predict(X_test_scaled)
    bnb_accuracy = metrics.accuracy_score(bnb_pred, y_test)
    bnb_train_pred = BNB.predict(X_train_scaled)
    bnb_train_accuracy = metrics.accuracy_score(bnb_train_pred, y_train)
    
    # Complement Naive Bayes için performans metrikleri
    cnb_pred = CNB.predict(X_test_scaled)
    cnb_accuracy = metrics.accuracy_score(cnb_pred, y_test)
    cnb_train_pred = CNB.predict(X_train_scaled)
    cnb_train_accuracy = metrics.accuracy_score(cnb_train_pred, y_train)

    # Gaussian Naive Bayes için performans metrikleri
    gnb_pred = GNB.predict(X_test_scaled)
    gnb_accuracy = metrics.accuracy_score(gnb_pred, y_test)
    gnb_train_pred = GNB.predict(X_train_scaled)
    gnb_train_accuracy = metrics.accuracy_score(gnb_train_pred, y_train)

    # Multinomial Naive Bayes için performans metrikleri
    mnb_pred = nb_model.predict(X_test_scaled)
    mnb_accuracy = metrics.accuracy_score(mnb_pred, y_test)
    mnb_train_pred = nb_model.predict(X_train_scaled)
    mnb_train_accuracy = metrics.accuracy_score(mnb_train_pred, y_train)
    
    # Performans metriklerini yazdırma
    print("\nBernoulli Naive Bayes Test Seti Accuracy:", bnb_accuracy)
    print("Complement Naive Bayes Test Seti Accuracy:", cnb_accuracy)
    print("Gaussian Naive Bayes Test Seti Accuracy:", gnb_accuracy)
    print("Multinomial Naive Bayes Test Seti Accuracy:", mnb_accuracy)
    print("Logistic Regression Test Set Accuracy: ", log_reg_accuracy )
    print("Random Forest Test Set Accuracy: ", rf_accuracy)
    print("Decision Tree Test Set Accuracy: ",dt_accuracy)
    print("Yapay Sinir Ağı Test Seti Accuracy:", ann_test_accuracy)
    print("SVM Test Set Accuracy: ", svm_accuracy)
    
    # Bernoulli Naive Bayes sınıflandırma raporu
    bnb_report = classification_report(y_test, bnb_pred, zero_division=1)
    print("Bernoulli Naive Bayes Classification Report:")
    print(bnb_report)
    

    # Complement Naive Bayes sınıflandırma raporu
    cnb_report = classification_report(y_test, cnb_pred, zero_division=1)
    print("Complement Naive Bayes Classification Report:")
    print(cnb_report)
    
    # Multinomial Naive Bayes sınıflandırma raporu
    nb_report = classification_report(y_test, mnb_pred, zero_division=1)
    print("Multinomial Naive Bayes Classification Report:")
    print(nb_report)
    
    # Gaussian Naive Bayes sınıflandırma raporu
    gnb_report = classification_report(y_test, gnb_pred, zero_division=1)
    print("Gaussian Naive Bayes Classification Report:")
    print(gnb_report)

    # Logistic Regression sınıflandırma raporu
    log_reg_report = classification_report(y_test, log_reg_pred, zero_division=1)
    print("Logistic Regression Classification Report:")
    print(log_reg_report)
    
    # Random Forest sınıflandırma raporu
    rf_report = classification_report(y_test, rf_pred, zero_division=1)
    print("Random Forest Classification Report:")
    print(rf_report)
    
    # Decision Tree sınıflandırma raporu
    dt_report = classification_report(y_test, dt_pred, zero_division=1)
    print("Decision Tree Classification Report:")
    print(dt_report)

    print("\nYapay Sinir Ağı Test Seti Sınıflandırma Raporu:")
    ann_test_pred = ANN.predict(X_test)
    print(classification_report(y_test, ann_test_pred, zero_division=1))
    
    # SVM sınıflandırma raporu
    svm_report = classification_report(y_test, svm_pred, zero_division=1)
    print("SVM Classification Report:")
    print(svm_report)
    
    
else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")
