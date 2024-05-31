import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

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

# Veriyi uygun şekilde yeniden şekillendirme
X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# CNN modeli için
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping])

cnn_pred = (cnn_model.predict(X_test) > 0.5).astype(int)
cnn_accuracy = accuracy_score(y_test, cnn_pred)

print("CNN Test Seti Doğruluğu:", cnn_accuracy)
print("CNN Classification Report for Test Set:")
print(classification_report(y_test, cnn_pred, zero_division=1))

# RNN modeli için
rnn_model = Sequential()
rnn_model.add(SimpleRNN(128, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
rnn_model.add(Dense(1, activation='sigmoid'))
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping])

rnn_pred = (rnn_model.predict(X_test) > 0.5).astype(int)
rnn_accuracy = accuracy_score(y_test, rnn_pred)

print("RNN Test Seti Doğruluğu:", rnn_accuracy)
print("RNN Classification Report for Test Set:")
print(classification_report(y_test, rnn_pred, zero_division=1))

# LSTM modeli için
lstm_model = Sequential()
lstm_model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping])

lstm_pred = (lstm_model.predict(X_test) > 0.5).astype(int)
lstm_accuracy = accuracy_score(y_test, lstm_pred)

print("LSTM Test Seti Doğruluğu:", lstm_accuracy)
print("LSTM Classification Report for Test Set:")
print(classification_report(y_test, lstm_pred, zero_division=1))
