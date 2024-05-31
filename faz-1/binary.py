import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# CSV dosyasını yükle
all_data = pd.read_csv('all_data.csv')

# Özellik isimlerini temizleme
cleaned_columns = [re.sub(r'[^\w]', '_', col) for col in all_data.columns]
all_data.columns = cleaned_columns

# Özellikler ve hedef değişkeni ayırma
X = all_data.iloc[:, :-1]  # Özellikler
y = all_data.iloc[:, -1]   # Hedef değişken (duygu puanları)

# Eğitim ve test verilerine ayırma işlemi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM modelini oluşturun ve eğitin
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Bernoulli Naive Bayes modelini oluşturun ve eğitin
BNB = BernoulliNB()
BNB.fit(X_train, y_train)

# Decision Tree modelini oluşturun ve eğitin
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Support Vector Machine (SVM) modelini oluşturun ve eğitin
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Artificial Neural Network (ANN) modelini oluşturun ve eğitin
ann_model = MLPClassifier()
ann_model.fit(X_train, y_train)

# Lojistik Regresyon modelini oluşturun ve eğitin
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Random Forest modelini oluşturun ve eğitin
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# K-Nearest Neighbors (KNN) modelini oluşturun ve eğitin
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# XGBoost modelini oluşturun ve eğitin
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Performans metriklerini hesaplama ve yazdırma
# LightGBM için performans metrikleri
lgb_pred = lgb_model.predict(X_test)
lgb_accuracy = metrics.accuracy_score(lgb_pred, y_test)

# Bernoulli Naive Bayes için performans metrikleri
bnb_pred = BNB.predict(X_test)
accuracy_score_bnb = metrics.accuracy_score(bnb_pred, y_test)

# Decision Tree için performans metrikleri
dt_pred = dt_model.predict(X_test)
dt_accuracy = metrics.accuracy_score(dt_pred, y_test)

# SVM için performans metrikleri
svm_pred = svm_model.predict(X_test)
svm_accuracy = metrics.accuracy_score(svm_pred, y_test)

# ANN için performans metrikleri
ann_pred = ann_model.predict(X_test)
ann_accuracy = metrics.accuracy_score(ann_pred, y_test)

# Lojistik Regresyon için performans metrikleri
log_reg_pred = log_reg_model.predict(X_test)
log_reg_accuracy = metrics.accuracy_score(log_reg_pred, y_test)

# Random Forest için performans metrikleri
rf_pred = rf_model.predict(X_test)
rf_accuracy = metrics.accuracy_score(rf_pred, y_test)

# KNN için performans metrikleri
knn_pred = knn_model.predict(X_test)
knn_accuracy = metrics.accuracy_score(knn_pred, y_test)

# XGBoost için performans metrikleri
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = metrics.accuracy_score(xgb_pred, y_test)

# Performans metriklerini yazdırma
print("Bernoulli Naive Bayes Test Set Accuracy:", accuracy_score_bnb)
print("Logistic Regression Test Set Accuracy:", log_reg_accuracy)
print("Random Forest Test Set Accuracy:", rf_accuracy)
print("Decision Tree Test Set Accuracy:", dt_accuracy)
print("Support Vector Machine Test Set Accuracy:", svm_accuracy)
print("Artificial Neural Network Test Set Accuracy:", ann_accuracy)
print("K-Nearest Neighbors Test Set Accuracy:", knn_accuracy)
print("XGBoost Test Set Accuracy:", xgb_accuracy)
print("LightGBM Test Set Accuracy:", lgb_accuracy)
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
