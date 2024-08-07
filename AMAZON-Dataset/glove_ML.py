import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics

# Birleşik veri dosyasının yolu
file_path = 'data.xlsx'

# Hata kontrolü ve dosyanın varlığını kontrol etme
if os.path.exists(file_path):
    print("Dosya başarıyla açıldı. Veriler okunuyor...")

    # Excel dosyasını oku
    df = pd.read_excel(file_path)

    # GloVe vektörlerinin yolu
    glove_path = 'glove.6B.100d.txt'

    # GloVe vektörlerini yükleyin
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Metin verilerini GloVe vektörleriyle temsil etme
    embedding_dim = 100

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

    # Bernoulli Naive Bayes modelini oluşturun ve eğitin
    nb_model = BernoulliNB()
    nb_model.fit(X_train, y_train)

    # Logistic Regression
    LR = LogisticRegression(max_iter=20000)
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

    # Artificial Neural Network (MLPClassifier)
    ANN = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20000)
    ANN.fit(X_train, y_train)

    # XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    XGB = xgb.XGBClassifier()
    XGB.fit(X_train, y_train_encoded)

    # K-Nearest Neighbors
    KNN = KNeighborsClassifier()
    KNN.fit(X_train, y_train)

    # LightGBM
    LGB = LGBMClassifier(verbose=0)  # verbose=0 ile içsel bilgilendirme mesajları bastırılmayacak
    LGB.fit(X_train, y_train)

    # Test setlerinde model performanslarını değerlendirme
    lr_test_accuracy = LR.score(X_test, y_test)
    svm_test_accuracy = SVM.score(X_test, y_test)
    dt_test_accuracy = DT.score(X_test, y_test)
    rf_test_accuracy = RF.score(X_test, y_test)
    ann_test_accuracy = ANN.score(X_test, y_test)
    xgb_test_accuracy = XGB.score(X_test, label_encoder.transform(y_test))
    knn_test_accuracy = KNN.score(X_test, y_test)
    lgb_test_accuracy = LGB.score(X_test, y_test)
    gnb_test_accuracy = nb_model.score(X_test, y_test)

    print("\nBernoulli Naive Bayes Test Set Accuracy:", gnb_test_accuracy)
    print("Logistic Regression Test Set Accuracy:", lr_test_accuracy)
    print("Random Forest Test Set Accuracy:", rf_test_accuracy)
    print("Decision Tree Test Set Accuracy:", dt_test_accuracy)
    print("Support Vector Machine Test Set Accuracy:", svm_test_accuracy)
    print("Artificial Neural Network Test Set Accuracy:", ann_test_accuracy)
    print("K-Nearest Neighbors Test Set Accuracy:", knn_test_accuracy)
    print("XGBoost Test Set Accuracy:", xgb_test_accuracy)
    print("LightGBM Test Set Accuracy:", lgb_test_accuracy)
    

    # Test setlerinde sınıflandırma raporu
    # Naive Bayes sınıflandırma raporu
    nb_pred = nb_model.predict(X_test)
    nb_report = classification_report(y_test, nb_pred, zero_division=1)
    print("Bernoulli Naive Bayes Classification Report:")
    print(nb_report)
    
    print("\nLogistic Regression Test Set Classification Report:")
    lr_test_pred = LR.predict(X_test)
    print(classification_report(y_test, lr_test_pred, zero_division=1))
    
    print("\nRandom Forest Test Set Classification Report:")
    rf_test_pred = RF.predict(X_test)
    print(classification_report(y_test, rf_test_pred, zero_division=1))

    print("\nDecision Tree Test Set Classification Report:")
    dt_test_pred = DT.predict(X_test)
    print(classification_report(y_test, dt_test_pred, zero_division=1))
    
    print("\nSupport Vector Machine Test Set Classification Report:")
    svm_test_pred = SVM.predict(X_test)
    print(classification_report(y_test, svm_test_pred, zero_division=1))

    print("\nArtificial Neural Network Test Set Classification Report:")
    ann_test_pred = ANN.predict(X_test)
    print(classification_report(y_test, ann_test_pred, zero_division=1))
    
    print("\nK-Nearest Neighbors Test Set Classification Report:")
    knn_test_pred = KNN.predict(X_test)
    print(classification_report(y_test, knn_test_pred, zero_division=1))

    print("\nXGBoost Test Set Classification Report:")
    xgb_test_pred = XGB.predict(X_test)
    print(classification_report(y_test, label_encoder.inverse_transform(xgb_test_pred), zero_division=1))

    print("\nLightGBM Test Set Classification Report:")
    lgb_test_pred = LGB.predict(X_test)
    print(classification_report(y_test, lgb_test_pred, zero_division=1))

else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")
