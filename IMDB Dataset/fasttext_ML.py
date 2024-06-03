import os
import pandas as pd
import numpy as np
from gensim.models import FastText
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

# Birleşik veri dosyasının yolu
file_path = 'data.xlsx'

# Hata kontrolü ve dosyanın varlığını kontrol etme
if os.path.exists(file_path):
    print("Dosya başarıyla açıldı. Veriler okunuyor...")

    # Excel dosyasını oku
    df = pd.read_excel(file_path)

    # FastText modelini eğitme
    documents = [review.split() for review in df['reviewText']]
    model = FastText(vector_size=100, window=5, min_count=1, workers=4)
    model.build_vocab(documents)
    model.train(documents, total_examples=len(documents), epochs=10)

    # Metin belgelerini vektörleştirme
    X = np.array([np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(100)], axis=0) for doc in documents])
    y = df['duygu']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Bernoulli Naive Bayes
    BNB = BernoulliNB()
    BNB.fit(X_train, y_train)

    # Test setlerinde model performanslarını değerlendirme
    lr_test_accuracy = LR.score(X_test, y_test)
    svm_test_accuracy = SVM.score(X_test, y_test)
    dt_test_accuracy = DT.score(X_test, y_test)
    rf_test_accuracy = RF.score(X_test, y_test)
    ann_test_accuracy = ANN.score(X_test, y_test)
    xgb_test_accuracy = XGB.score(X_test, label_encoder.transform(y_test))
    knn_test_accuracy = KNN.score(X_test, y_test)
    lgb_test_accuracy = LGB.score(X_test, y_test)
    bnb_test_accuracy = BNB.score(X_test, y_test)
    
    print("\nBernoulli Naive Bayes Test Set Accuracy:", bnb_test_accuracy)
    print("Logistic Regression Test Set Accuracy:", lr_test_accuracy)
    print("Random Forest Test Set Accuracy:", rf_test_accuracy)
    print("Decision Tree Test Set Accuracy:", dt_test_accuracy)
    print("Support Vector Machine Test Set Accuracy:", svm_test_accuracy)
    print("Artificial Neural Network Test Set Accuracy:", ann_test_accuracy)
    print("K-Nearest Neighbors Test Set Accuracy:", knn_test_accuracy)
    print("XGBoost Test Set Accuracy:", xgb_test_accuracy)
    print("LightGBM Test Set Accuracy:", lgb_test_accuracy)
    

    # Test setlerinde sınıflandırma raporu
    
    print("\nBernoulli Naive Bayes Test Seti Sınıflandırma Raporu:")
    bnb_test_pred = BNB.predict(X_test)
    print(classification_report(y_test, bnb_test_pred, zero_division=1))
    
    print("\nLogistic Regression Test Seti Sınıflandırma Raporu:")
    lr_test_pred = LR.predict(X_test)
    print(classification_report(y_test, lr_test_pred, zero_division=1))

    print("\nRandom Forest Test Seti Sınıflandırma Raporu:")
    rf_test_pred = RF.predict(X_test)
    print(classification_report(y_test, rf_test_pred, zero_division=1))
    
    print("\nDecision Tree Test Seti Sınıflandırma Raporu:")
    dt_test_pred = DT.predict(X_test)
    print(classification_report(y_test, dt_test_pred, zero_division=1))
    
    print("\nSupport Vector Machine Test Seti Sınıflandırma Raporu:")
    svm_test_pred = SVM.predict(X_test)
    print(classification_report(y_test, svm_test_pred, zero_division=1))

    print("\nArtificial Neural Network Test Seti Sınıflandırma Raporu:")
    ann_test_pred = ANN.predict(X_test)
    print(classification_report(y_test, ann_test_pred, zero_division=1))

    print("\nK-Nearest Neighbors Test Seti Sınıflandırma Raporu:")
    knn_test_pred = KNN.predict(X_test)
    print(classification_report(y_test, knn_test_pred, zero_division=1))
    
    print("\nXGBoost Test Seti Sınıflandırma Raporu:")
    xgb_test_pred = XGB.predict(X_test)
    print(classification_report(y_test, label_encoder.inverse_transform(xgb_test_pred), zero_division=1))

    print("\nLightGBM Test Seti Sınıflandırma Raporu:")
    lgb_test_pred = LGB.predict(X_test)
    print(classification_report(y_test, lgb_test_pred, zero_division=1))

else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")
