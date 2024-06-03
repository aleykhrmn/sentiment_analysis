import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

# CSV dosyasını yükleme
df = pd.read_csv('all_data.csv')

# Özellikler (features) ve etiketleri (labels) ayırma
X = df.drop('Sentiment', axis=1).values
y = df['Sentiment'].values

# Eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri normalize etme
X_train = X_train.astype('float32') / X_train.max()
X_test = X_test.astype('float32') / X_test.max()

# Verileri CNN, LSTM ve RNN girişine uygun hale getirme
X_train_exp = np.expand_dims(X_train, axis=2)
X_test_exp = np.expand_dims(X_test, axis=2)

# EarlyStopping callback'i
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# RNN Modeli Tanımlama ve Eğitme
rnn_model = Sequential()
rnn_model.add(SimpleRNN(64, input_shape=(X_train_exp.shape[1], 1)))
rnn_model.add(Dense(128, activation='relu'))
rnn_model.add(Dropout(0.5))
rnn_model.add(Dense(3, activation='softmax'))  # 3 sınıf olduğu için 'softmax' kullanıyoruz

rnn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

rnn_history = rnn_model.fit(X_train_exp, y_train, epochs=10, validation_data=(X_test_exp, y_test), 
                            batch_size=32, callbacks=[early_stopping])

# RNN Modeli Değerlendirme
rnn_test_loss, rnn_test_accuracy = rnn_model.evaluate(X_test_exp, y_test, verbose=2)
print(f'RNN Modeli Test Seti Doğruluğu: {rnn_test_accuracy}')

rnn_y_test_pred = rnn_model.predict(X_test_exp)
rnn_y_test_pred_classes = np.argmax(rnn_y_test_pred, axis=1)

print("\nRNN Modeli Test Seti Sınıflandırma Raporu:")
print(classification_report(y_test, rnn_y_test_pred_classes, zero_division=1))


# LSTM Modeli Tanımlama ve Eğitme
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_train_exp.shape[1], 1)))
lstm_model.add(Dense(128, activation='relu'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(3, activation='softmax'))  # 3 sınıf olduğu için 'softmax' kullanıyoruz

lstm_model.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

lstm_history = lstm_model.fit(X_train_exp, y_train, epochs=10, validation_data=(X_test_exp, y_test), 
                              batch_size=32, callbacks=[early_stopping])

# LSTM Modeli Değerlendirme
lstm_test_loss, lstm_test_accuracy = lstm_model.evaluate(X_test_exp, y_test, verbose=2)
print(f'LSTM Modeli Test Seti Doğruluğu: {lstm_test_accuracy}')

lstm_y_test_pred = lstm_model.predict(X_test_exp)
lstm_y_test_pred_classes = np.argmax(lstm_y_test_pred, axis=1)

print("\nLSTM Modeli Test Seti Sınıflandırma Raporu:")
print(classification_report(y_test, lstm_y_test_pred_classes, zero_division=1))


# CNN Modeli Tanımlama ve Eğitme
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_exp.shape[1], 1)))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(3, activation='softmax'))  # 3 sınıf olduğu için 'softmax' kullanıyoruz

cnn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

cnn_history = cnn_model.fit(X_train_exp, y_train, epochs=10, validation_data=(X_test_exp, y_test), 
                            batch_size=32, callbacks=[early_stopping])

# CNN Modeli Değerlendirme
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test_exp, y_test, verbose=2)
print(f'CNN Modeli Test Seti Doğruluğu: {cnn_test_accuracy}')

cnn_y_test_pred = cnn_model.predict(X_test_exp)
cnn_y_test_pred_classes = np.argmax(cnn_y_test_pred, axis=1)

print("\nCNN Modeli Test Seti Sınıflandırma Raporu:")
print(classification_report(y_test, cnn_y_test_pred_classes, zero_division=1))
