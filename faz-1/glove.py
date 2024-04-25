import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB, GaussianNB, BernoulliNB

# GloVe modelinin yolu
glove_file = 'glove.6B.100d.txt'

# GloVe modelini Word2Vec formatına dönüştürme ve yükleme
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

# Veri dosyasının yolu
file_path = 'data.xlsx'

# Hata kontrolü ve dosyanın varlığını kontrol etme
if os.path.exists(file_path):
    print("Dosya başarıyla açıldı. Veriler okunuyor...")

    # Excel dosyasını oku
    df = pd.read_excel(file_path)

    # Metin belgelerini vektörleştirme
    X = np.array([np.mean([glove_model[word] for word in doc.split() if word in glove_model.key_to_index] or [np.zeros(100)], axis=0) for doc in df['reviewText']])
    y = df['duygu']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verilerin negatif değerlerini ele almak için Min-Max ölçeklendirme
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelleri oluşturma ve eğitme

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
    ANN.fit(X_train_scaled, y_train)

    # SVM modelini oluşturun
    svm_model = SVC()
    svm_model.fit(X_train_scaled, y_train)

    # Naive Bayes modellerini oluşturun
    cnb_model = ComplementNB()
    cnb_model.fit(X_train_scaled, y_train)

    gnb_model = GaussianNB()
    gnb_model.fit(X_train_scaled, y_train)

    mnb_model = MultinomialNB()
    mnb_model.fit(X_train_scaled, y_train)

    bnb_model = BernoulliNB()
    bnb_model.fit(X_train_scaled, y_train)

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

    # Yapay Sinir Ağı için performans metrikleri
    ann_train_accuracy = ANN.score(X_train_scaled, y_train)
    ann_test_accuracy = ANN.score(X_test_scaled, y_test)

    # SVM için performans metrikleri
    svm_pred = svm_model.predict(X_test_scaled)
    svm_accuracy = metrics.accuracy_score(svm_pred, y_test)
    svm_train_pred = svm_model.predict(X_train_scaled)
    svm_train_accuracy = metrics.accuracy_score(svm_train_pred, y_train)

    # Naive Bayes için performans metrikleri
    cnb_pred = cnb_model.predict(X_test_scaled)
    cnb_accuracy = metrics.accuracy_score(cnb_pred, y_test)
    cnb_train_pred = cnb_model.predict(X_train_scaled)
    cnb_train_accuracy = metrics.accuracy_score(cnb_train_pred, y_train)

    gnb_pred = gnb_model.predict(X_test_scaled)
    gnb_accuracy = metrics.accuracy_score(gnb_pred, y_test)
    gnb_train_pred = gnb_model.predict(X_train_scaled)
    gnb_train_accuracy = metrics.accuracy_score(gnb_train_pred, y_train)

    mnb_pred = mnb_model.predict(X_test_scaled)
    mnb_accuracy = metrics.accuracy_score(mnb_pred, y_test)
    mnb_train_pred = mnb_model.predict(X_train_scaled)
    mnb_train_accuracy = metrics.accuracy_score(mnb_train_pred, y_train)

    bnb_pred = bnb_model.predict(X_test_scaled)
    bnb_accuracy = metrics.accuracy_score(bnb_pred, y_test)
    bnb_train_pred = bnb_model.predict(X_train_scaled)
    bnb_train_accuracy = metrics.accuracy_score(bnb_train_pred, y_train)

    # Performans metriklerini yazdırma
    print("Bernoulli Naive Bayes Test Set Accuracy: ", bnb_accuracy)
    print("Complement Naive Bayes Test Set Accuracy: ", cnb_accuracy)
    print("Gaussian Naive Bayes Test Set Accuracy: ", gnb_accuracy)
    print("Multinomial Naive Bayes Test Set Accuracy: ", mnb_accuracy)
    print("Logistic Regression Test Set Accuracy: ", log_reg_accuracy)
    print("Random Forest Test Set Accuracy: ", rf_accuracy)
    print("Decision Tree Test Set Accuracy: ", dt_accuracy)
    print("Yapay Sinir Ağı Test Seti Accuracy:", ann_test_accuracy)
    print("SVM Test Set Accuracy: ", svm_accuracy)

    
    print("\nBernoulli Naive Bayes Classification Report:")
    print(classification_report(y_test, bnb_pred, zero_division=1))
    
    print("\nComplement Naive Bayes Classification Report:")
    print(classification_report(y_test, cnb_pred, zero_division=1))

    print("\nGaussian Naive Bayes Classification Report:")
    print(classification_report(y_test, gnb_pred, zero_division=1))

    print("\nMultinomial Naive Bayes Classification Report:")
    print(classification_report(y_test, mnb_pred, zero_division=1))

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

    # Yapay Sinir Ağı sınıflandırma raporu
    print("\nYapay Sinir Ağı Test Seti Sınıflandırma Raporu:")
    ann_test_pred = ANN.predict(X_test_scaled)
    print(classification_report(y_test, ann_test_pred, zero_division=1))

    # SVM sınıflandırma raporu
    svm_report = classification_report(y_test, svm_pred, zero_division=1)
    print("SVM Classification Report:")
    print(svm_report)

else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")
