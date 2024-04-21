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

# Complement Naive Bayes modelini oluşturun
CNB = ComplementNB()
# Modeli eğitin
CNB.fit(X_train, y_train)

# Gaussian Naive Bayes modelini oluşturun
GNB = GaussianNB()
# Modeli eğitin
GNB.fit(X_train, y_train)

# Decision Tree modelini oluşturun
dt_model = DecisionTreeClassifier()
# Modeli eğitin
dt_model.fit(X_train, y_train)

# Lojistik Regresyon modelini oluşturun
log_reg_model = LogisticRegression()
# Modeli eğitin
log_reg_model.fit(X_train, y_train)

# Naive Bayes modelini oluşturma ve eğitme
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Random Forest modelini oluşturun
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# Modeli eğitin
rf_model.fit(X_train, y_train)

# Bernoulli Naive Bayes için performans metrikleri
accuracy_score_bnb = metrics.accuracy_score(BNB.predict(X_test), y_test)
train_accuracy_bnb = metrics.accuracy_score(BNB.predict(X_train), y_train)
train_pred_bnb = BNB.predict(X_train)
test_pred_bnb = BNB.predict(X_test)

# Complement Naive Bayes için performans metrikleri
accuracy_score_cnb = metrics.accuracy_score(CNB.predict(X_test), y_test)
train_accuracy_cnb = metrics.accuracy_score(CNB.predict(X_train), y_train)
train_pred_cnb = CNB.predict(X_train)
test_pred_cnb = CNB.predict(X_test)

# Gaussian Naive Bayes için performans metrikleri
accuracy_score_gnb = metrics.accuracy_score(GNB.predict(X_test), y_test)
train_accuracy_gnb = metrics.accuracy_score(GNB.predict(X_train), y_train)
train_pred_gnb = GNB.predict(X_train)
test_pred_gnb = GNB.predict(X_test)

# Decision Tree için performans metrikleri
dt_pred = dt_model.predict(X_test)
dt_accuracy = metrics.accuracy_score(dt_pred, y_test)
dt_train_pred = dt_model.predict(X_train)
dt_train_accuracy = metrics.accuracy_score(dt_train_pred, y_train)

# Lojistik Regresyon için performans metrikleri
log_reg_pred = log_reg_model.predict(X_test)
log_reg_accuracy = metrics.accuracy_score(log_reg_pred, y_test)
log_reg_train_pred = log_reg_model.predict(X_train)
log_reg_train_accuracy = metrics.accuracy_score(log_reg_train_pred, y_train)

# Naive Bayes için performans metrikleri
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
train_pred = nb_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

# Random Forest için performans metrikleri
rf_pred = rf_model.predict(X_test)
rf_accuracy = metrics.accuracy_score(rf_pred, y_test)
rf_train_pred = rf_model.predict(X_train)
rf_train_accuracy = metrics.accuracy_score(rf_train_pred, y_train)

# Performans metriklerini yazdırma
print("Bernoulli Naive Bayes Test Set Accuracy: {:.2f}%".format(accuracy_score_bnb * 100))
print("Complement Naive Bayes Test Set Accuracy: {:.2f}%".format(accuracy_score_cnb * 100))
print("Gaussian Naive Bayes Test Set Accuracy: {:.2f}%".format(accuracy_score_gnb * 100))
print("Decision Tree Test Set Accuracy: {:.2f}%".format(dt_accuracy * 100))
print("Logistic Regression Test Set Accuracy: {:.2f}%".format(log_reg_accuracy * 100))
print("Multinomial Naive Bayes Test Set Accuracy: {:.2f}%".format(accuracy * 100))
print("Random Forest Test Set Accuracy: {:.2f}%".format(rf_accuracy * 100))

print("\n ")

print("Bernoulli Naive Bayes Training Set Accuracy: {:.2f}%".format(train_accuracy_bnb * 100))
print("Complement Naive Bayes Training Set Accuracy: {:.2f}%".format(train_accuracy_cnb * 100))
print("Gaussian Naive Bayes Training Set Accuracy: {:.2f}%".format(train_accuracy_gnb * 100))
print("Decision Tree Training Set Accuracy: {:.2f}%".format(dt_train_accuracy * 100))
print("Logistic Regression Training Set Accuracy: {:.2f}%".format(log_reg_train_accuracy * 100))
print("Multinomial Naive Bayes Training Set Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Random Forest Training Set Accuracy: {:.2f}%".format(rf_train_accuracy * 100))

# Sınıflandırma raporlarını yazdırma

print("Bernoulli Naive Bayes Classification Report for Test Set:")
print(classification_report(y_test, test_pred_bnb, zero_division=1))  # zero_division parametresi ekleniyor
print("Bernoulli Naive Bayes Classification Report for Train Set:", )
print(classification_report(y_train, train_pred_bnb, zero_division=1))  # zero_division parametresi ekleniyor
print("Complement Naive Bayes Classification Report for Test Set:")
print(classification_report(y_test, test_pred_cnb, zero_division=1))  # zero_division parametresi ekleniyor
print("Complement Naive Bayes Classification Report for Train Set:")
print(classification_report(y_train, train_pred_cnb, zero_division=1))  # zero_division parametresi ekleniyor
print("Gaussian Naive Bayes Classification Report for Test Set :")
print(classification_report(y_test, test_pred_gnb, zero_division=1))  # zero_division parametresi ekleniyor
print("Gaussian Naive Bayes Classification Report for Train Set:")
print(classification_report(y_train, train_pred_gnb, zero_division=1))  # zero_division parametresi ekleniyor
print("Decision Tree Classification Report for Test Set:")
print(classification_report(y_test, dt_pred, zero_division=1))  # zero_division parametresi ekleniyor
print("Decision Tree Classification Report for Training Set:")
print(classification_report(y_train, dt_train_pred, zero_division=1))  # zero_division parametresi ekleniyor
print("Logistic Regression Classification Report for Test Set:")
print(classification_report(y_test, log_reg_pred, zero_division=1))  # zero_division parametresi ekleniyor
print("Logistic Regression Classification Report for Training Set:")
print(classification_report(y_train, log_reg_train_pred, zero_division=1))  # zero_division parametresi ekleniyor
print("Multinomial Naive Bayes Classification Report for Test Set:")
print(classification_report(y_train, train_pred, zero_division=1))  # zero_division parametresi ekleniyor
print("Multinomial Naive Bayes Classification Report for Train Set:")
print(classification_report(y_test, y_pred, zero_division=1))  # zero_division parametresi ekleniyor
print("Random Forest Classification Report for Test Set:")
print(classification_report(y_test, rf_pred, zero_division=1))  # zero_division parametresi ekleniyor
print("Random Forest Classification Report for Training Set:")
print(classification_report(y_train, rf_train_pred, zero_division=1))  # zero_division parametresi ekleniyor
