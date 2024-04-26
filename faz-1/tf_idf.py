import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

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

    # Etiketleri sayısal değerlere dönüştürme
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y_encoded, test_size=0.2, random_state=42)

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
    RF = RandomForestClassifier(n_estimators=100, random_state=42)
    RF.fit(X_train, y_train)
    
    # Yapay Sinir Ağı modelini oluşturma ve eğitme
    ANN = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    ANN.fit(X_train, y_train)
    
    # SVM modelini oluşturma ve eğitme
    SVM = SVC()
    SVM.fit(X_train, y_train)

    # KNN modelini oluşturma ve eğitme
    KNN = KNeighborsClassifier()
    KNN.fit(X_train, y_train)
    
    # XGBoost modelini oluşturma ve eğitme
    XGB = xgb.XGBClassifier()
    XGB.fit(X_train, y_train)
    
    # LightGBM modelini oluşturma ve eğitme
    LGB = lgb.LGBMClassifier(verbose=0)
    LGB.fit(X_train, y_train)

    # Test setlerinde model performanslarını değerlendirme
    mnb_test_accuracy = MNB.score(X_test, y_test)
    lr_test_accuracy = LR.score(X_test, y_test)
    dt_test_accuracy = DT.score(X_test, y_test)
    rf_test_accuracy = RF.score(X_test, y_test)
    ann_test_accuracy = ANN.score(X_test, y_test)
    svm_test_accuracy = SVM.score(X_test, y_test)
    knn_test_accuracy = KNN.score(X_test, y_test)
    xgb_test_accuracy = XGB.score(X_test, y_test)
    lgb_test_accuracy = LGB.score(X_test, y_test)

    # Sonuçları yazdırma
    print("Multinomial Naive Bayes Test Set Accuracy:", mnb_test_accuracy)
    print("Logistic Regression Test Set Accuracy:", lr_test_accuracy)
    print("Random Forest Test Set Accuracy:", rf_test_accuracy)
    print("Decision Tree Test Set Accuracy:", dt_test_accuracy)
    print("Yapay Sinir Ağı Test Seti Accuracy:", ann_test_accuracy)
    print("SVM Test Seti Accuracy:", svm_test_accuracy)
    print("KNN Test Seti Accuracy:", knn_test_accuracy)
    print("XGBoost Test Seti Accuracy:", xgb_test_accuracy)
    print("LightGBM Test Seti Accuracy:", lgb_test_accuracy)

    # Test setlerinde sınıflandırma raporu    

    print("\nMultinomial Naive Bayes Test Seti Sınıflandırma Raporu:")
    mnb_test_pred = MNB.predict(X_test)
    print(classification_report(y_test, mnb_test_pred, zero_division=1))

    print("\nLogistic Regression Test Seti Sınıflandırma Raporu:")
    lr_test_pred = LR.predict(X_test)
    print(classification_report(y_test, lr_test_pred, zero_division=1))

    print("\nRandom Forest Test Seti Sınıflandırma Raporu:")
    rf_test_pred = RF.predict(X_test)
    print(classification_report(y_test, rf_test_pred, zero_division=1))
    
    print("\nDecision Tree Test Seti Sınıflandırma Raporu:")
    dt_test_pred = DT.predict(X_test)
    print(classification_report(y_test, dt_test_pred, zero_division=1))

    print("\nYapay Sinir Ağı Test Seti Sınıflandırma Raporu:")
    ann_test_pred = ANN.predict(X_test)
    print(classification_report(y_test, ann_test_pred, zero_division=1))

    print("\nSVM Test Seti Sınıflandırma Raporu:")
    svm_test_pred = SVM.predict(X_test)
    print(classification_report(y_test, svm_test_pred, zero_division=1))

    print("\nKNN Test Seti Sınıflandırma Raporu:")
    knn_test_pred = KNN.predict(X_test)
    print(classification_report(y_test, knn_test_pred, zero_division=1))
    
    print("\nXGBoost Test Seti Sınıflandırma Raporu:")
    xgb_test_pred = XGB.predict(X_test)
    print(classification_report(y_test, xgb_test_pred, zero_division=1))
    
    print("\nLightGBM Test Seti Sınıflandırma Raporu:")
    lgb_test_pred = LGB.predict(X_test)
    print(classification_report(y_test, lgb_test_pred, zero_division=1))

else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")
