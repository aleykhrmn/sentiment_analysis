import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


# Verileri chunksize ile parça parça yükleme
chunksize = 1000  # Her seferinde 1000 satır okuyacak

# İlk chunk'u yükle ve özellik isimlerini temizle
chunk_iter = pd.read_csv('all_data.csv', chunksize=chunksize)
all_data = next(chunk_iter)

cleaned_columns = [re.sub(r'[^\w]', '_', col) for col in all_data.columns]
all_data.columns = cleaned_columns

# Geri kalan chunk'ları yükleyip temizlenmiş veri setine ekle
for chunk in chunk_iter:
    chunk.columns = cleaned_columns
    all_data = pd.concat([all_data, chunk])

# Özellikler ve hedef değişkeni ayırma
X = all_data.iloc[:, :-1]  # Özellikler
y = all_data.iloc[:, -1]   # Hedef değişken (duygu puanları)

# Eğitim ve test verilerine ayırma işlemi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Bernoulli Naive Bayes modelini oluşturun ve eğitin
BNB = BernoulliNB()
BNB.fit(X_train, y_train)
# Logistic Regression modelini oluşturun ve eğitin
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Random Forest modelini oluşturun ve eğitin
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Decision Tree modelini oluşturun ve eğitin
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

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

# Test seti üzerinde tahmin yapma
bnb_pred = BNB.predict(X_test)
log_reg_pred = log_reg_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
ann_pred = ann_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)


# Performans metriklerini hesaplama
accuracy_score_bnb = accuracy_score(bnb_pred, y_test)
bnb_classification_report = classification_report(y_test, bnb_pred, zero_division=1)

log_reg_accuracy = accuracy_score(log_reg_pred, y_test)
log_reg_classification_report = classification_report(y_test, log_reg_pred, zero_division=1)

dt_accuracy = accuracy_score(dt_pred, y_test)
dt_classification_report = classification_report(y_test, dt_pred, zero_division=1)

rf_accuracy = accuracy_score(rf_pred, y_test)
rf_classification_report = classification_report(y_test, rf_pred, zero_division=1)

svm_accuracy = accuracy_score(svm_pred, y_test)
svm_classification_report = classification_report(y_test, svm_pred, zero_division=1)

ann_accuracy = accuracy_score(ann_pred, y_test)
ann_classification_report = classification_report(y_test, ann_pred, zero_division=1)

knn_accuracy = accuracy_score(knn_pred, y_test)
knn_classification_report = classification_report(y_test, knn_pred, zero_division=1)

xgb_accuracy = accuracy_score(xgb_pred, y_test)
xgb_classification_report = classification_report(y_test, xgb_pred, zero_division=1)

lgb_accuracy = metrics.accuracy_score(lgb_pred, y_test)
lgb_classification_report = classification_report(y_test, lgb_pred, zero_division=1)

# Sonuçları yazdırma
print("Bernoulli Naive Bayes Test Set Accuracy:", accuracy_score_bnb)
print("Logistic Regression Test Set Accuracy:", log_reg_accuracy)
print("Decision Tree Test Set Accuracy:", dt_accuracy)
print("Random Forest Test Set Accuracy:", rf_accuracy)
print("Support Vector Machine Test Set Accuracy:", svm_accuracy)
print("Artificial Neural Network Test Set Accuracy:", ann_accuracy)
print("K-Nearest Neighbors Test Set Accuracy:", knn_accuracy)
print("LightGBM Test Set Accuracy:", lgb_accuracy)
print("XGBoost Test Set Accuracy:", xgb_accuracy)


print("Bernoulli Naive Bayes Classification Report for Test Set:")
print(bnb_classification_report)

print("LightGBM Classification Report for Test Set:")
print(lgb_classification_report)

print("Logistic Regression Classification Report for Test Set:")
print(log_reg_classification_report)

print("Support Vector Machine Classification Report for Test Set:")
print(svm_classification_report)

print("Artificial Neural Network Classification Report for Test Set:")
print(ann_classification_report)

print("Random Forest Classification Report for Test Set:")
print(rf_classification_report)

print("K-Nearest Neighbors Classification Report for Test Set:")
print(knn_classification_report)

print("Decision Tree Classification Report for Test Set:")
print(dt_classification_report)

print("XGBoost Classification Report for Test Set:")
print(xgb_classification_report)
