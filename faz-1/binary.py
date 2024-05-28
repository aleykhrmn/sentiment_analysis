import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# CSV dosyasını yükle
if os.path.exists("all_data.csv"):
    all_data = pd.read_csv("all_data.csv", index_col=0)
else:
    print("CSV dosyası bulunamadı. Lütfen 'all_data.csv' dosyasını oluşturun veya kontrol edin.")
    exit()

# Özellikler ve hedef değişkeni ayırma
X = all_data.iloc[:, :-1]  # Özellikler
y = all_data.iloc[:, -1]   # Hedef değişken (duygu puanları)

# Eğitim ve test verilerine ayırma işlemi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bernoulli Naive Bayes modelini oluşturun
BNB = BernoulliNB()
# Modeli eğitin
BNB.fit(X_train, y_train)

# Decision Tree modelini oluşturun
dt_model = DecisionTreeClassifier()
# Modeli eğitin
dt_model.fit(X_train, y_train)

# Lojistik Regresyon modelini oluşturun
log_reg_model = LogisticRegression()
# Modeli eğitin
log_reg_model.fit(X_train, y_train)

# Random Forest modelini oluşturun
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# Modeli eğitin
rf_model.fit(X_train, y_train)

# Support Vector Machine (SVM) modelini oluşturun ve eğitin
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Artificial Neural Network (ANN) modelini oluşturun ve eğitin
ann_model = MLPClassifier()
ann_model.fit(X_train, y_train)

# K-Nearest Neighbors (KNN) modelini oluşturun ve eğitin
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# XGBoost modelini oluşturun ve eğitin
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# LightGBM modelini oluşturun ve eğitin
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Bernoulli Naive Bayes için performans metrikleri
accuracy_score_bnb = metrics.accuracy_score(BNB.predict(X_test), y_test)
test_pred_bnb = BNB.predict(X_test)

# Decision Tree için performans metrikleri
dt_pred = dt_model.predict(X_test)
dt_accuracy = metrics.accuracy_score(dt_pred, y_test)

# Lojistik Regresyon için performans metrikleri
log_reg_pred = log_reg_model.predict(X_test)
log_reg_accuracy = metrics.accuracy_score(log_reg_pred, y_test)

# Random Forest için performans metrikleri
rf_pred = rf_model.predict(X_test)
rf_accuracy = metrics.accuracy_score(rf_pred, y_test)

# SVM için performans metrikleri
svm_pred = svm_model.predict(X_test)
svm_accuracy = metrics.accuracy_score(svm_pred, y_test)

# ANN için performans metrikleri
ann_pred = ann_model.predict(X_test)
ann_accuracy = metrics.accuracy_score(ann_pred, y_test)

# KNN için performans metrikleri
knn_pred = knn_model.predict(X_test)
knn_accuracy = metrics.accuracy_score(knn_pred, y_test)

# XGBoost için performans metrikleri
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = metrics.accuracy_score(xgb_pred, y_test)

# LightGBM için performans metrikleri
lgb_pred = lgb_model.predict(X_test)
lgb_accuracy = metrics.accuracy_score(lgb_pred, y_test)

# Performans metriklerini yazdırma
print("Bernoulli Naive Bayes Test Seti Doğruluğu:", bnb_test_accuracy)
print("Logistic Regression Test Seti Doğruluğu:", lr_test_accuracy)
print("Random Forest Test Seti Doğruluğu:", rf_test_accuracy)
print("Decision Tree Test Seti Doğruluğu:", dt_test_accuracy)
print("SVM Test Seti Doğruluğu:", svm_accuracy)
print("ANN Test Seti Doğruluğu:", ann_accuracy)
print("KNN Test Seti Doğruluğu:", knn_test_accuracy)
print("XGBoost Test Seti Doğruluğu:", xgb_test_accuracy)
print("LightGBM Test Seti Doğruluğu:", lgb_test_accuracy)
print("\n ")

# Sınıflandırma raporlarını yazdırma
print("Bernoulli Naive Bayes Classification Report for Test Set:")
print(classification_report(y_test, bnb_pred, zero_division=1))
print("Logistic Regression Classification Report for Test Set:")
print(classification_report(y_test, log_reg_pred, zero_division=1))
print("Random Forest Classification Report for Test Set:")
print(classification_report(y_test, rf_pred, zero_division=1))
print("Decision Tree Classification Report for Test Set:")
print(classification_report(y_test, dt_pred, zero_division=1))
print("SVM Classification Report for Test Set:")
print(classification_report(y_test, svm_pred, zero_division=1))
print("ANN Classification Report for Test Set:")
print(classification_report(y_test, ann_pred, zero_division=1))
print("KNN Classification Report for Test Set:")
print(classification_report(y_test, knn_pred, zero_division=1))
print("XGBoost Classification Report for Test Set:")
print(classification_report(y_test, xgb_pred, zero_division=1))
print("LightGBM Classification Report for Test Set:")
print(classification_report(y_test, lgb_pred, zero_division=1))
