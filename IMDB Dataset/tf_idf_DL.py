import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Birleşik veri dosyasının yolu
file_path = 'data.xlsx'

# Hata kontrolü ve dosyanın varlığını kontrol etme
if os.path.exists(file_path):
    print("Dosya başarıyla açıldı. Veriler okunuyor...")

    # Excel dosyasını oku
    df = pd.read_excel(file_path)
    
    # Yalnızca metin ve duygu sütunlarını kullan
    texts = df['reviewText'].values
    labels = df['duygu'].values

    # Etiketleri sayısal değerlere dönüştürme
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Metin verilerini TF-IDF ile sayısal verilere dönüştürme
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    data = tfidf_matrix.toarray()

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)
    

    # CNN Modeli
    model_cnn = Sequential()
    model_cnn.add(Dense(128, input_shape=(5000,), activation='relu'))
    model_cnn.add(Dropout(0.2))
    model_cnn.add(Dense(256, activation='relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(1, activation='sigmoid'))
    model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # EarlyStopping callback'i oluşturma
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # CNN Modeli eğitme
    print("\nCNN Modeli Eğitiliyor...")
    history_cnn = model_cnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping])
    
    # CNN Modeli değerlendirme
    test_loss_cnn, test_accuracy_cnn = model_cnn.evaluate(X_test, y_test, verbose=2)
    y_pred_prob_cnn = model_cnn.predict(X_test)
    y_pred_classes_cnn = (y_pred_prob_cnn > 0.5).astype("int32")
    print(f'CNN Modeli Test Seti Doğruluğu: {test_accuracy_cnn}')
    print("CNN Modeli Test Seti Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred_classes_cnn, target_names=['0', '1'], zero_division=1))
    
    # RNN Modeli
    model_rnn = Sequential()
    model_rnn.add(Embedding(input_dim=len(tfidf_vectorizer.vocabulary_) + 1, output_dim=128, input_length=5000))
    model_rnn.add(SimpleRNN(128))
    model_rnn.add(Dense(1, activation='sigmoid'))
    model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # LSTM Modeli
    model_lstm = Sequential()
    model_lstm.add(Embedding(input_dim=len(tfidf_vectorizer.vocabulary_) + 1, output_dim=128, input_length=5000))
    model_lstm.add(LSTM(128))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # EarlyStopping callback'i oluşturma
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # RNN Modeli eğitme
    print("\nRNN Modeli Eğitiliyor...")
    history_rnn = model_rnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping])
    # RNN Modeli değerlendirme
    test_loss_rnn, test_accuracy_rnn = model_rnn.evaluate(X_test, y_test, verbose=2)
    y_pred_prob_rnn = model_rnn.predict(X_test)
    y_pred_classes_rnn = (y_pred_prob_rnn > 0.5).astype("int32")
    print(f'RNN Modeli Test Seti Doğruluğu: {test_accuracy_rnn}')
    print("RNN Modeli Test Seti Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred_classes_rnn, target_names=['0', '1'], zero_division=1))

    # LSTM Modeli eğitme
    print("\nLSTM Modeli Eğitiliyor...")
    history_lstm = model_lstm.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping])
    # LSTM Modeli değerlendirme
    test_loss_lstm, test_accuracy_lstm = model_lstm.evaluate(X_test, y_test, verbose=2)
    y_pred_prob_lstm = model_lstm.predict(X_test)
    y_pred_classes_lstm = (y_pred_prob_lstm > 0.5).astype("int32")
    print(f'LSTM Modeli Test Seti Doğruluğu: {test_accuracy_lstm}')
    print("LSTM Modeli Test Seti Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred_classes_lstm, target_names=['0', '1'], zero_division=1))

else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")
