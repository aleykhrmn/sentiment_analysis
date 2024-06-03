import os
import pandas as pd
import numpy as np
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
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
    labels_categorical = to_categorical(labels_encoded)

    # Metin verilerini sayısal verilere dönüştürme
    tokenizer = Tokenizer(num_words=5000)  # 5000 en sık kullanılan kelimeyi tut
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    # Tüm dizileri aynı uzunlukta olacak şekilde padding uygulama
    max_sequence_length = 200
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(data, labels_categorical, test_size=0.2, random_state=42)
    
    # FastText modelini eğitme
    documents = [text.split() for text in texts]
    fasttext_model = FastText(sentences=documents, vector_size=128, window=5, min_count=1, workers=4, sg=1)
    fasttext_model.train(documents, total_examples=len(documents), epochs=10)
    
    # Gömme matrisi oluşturma
    embedding_matrix = np.zeros((len(word_index) + 1, 128))
    for word, i in word_index.items():
        if word in fasttext_model.wv:
            embedding_matrix[i] = fasttext_model.wv[word]

    # CNN Modeli
    model_cnn = Sequential()
    model_cnn.add(Embedding(input_dim=len(word_index) + 1, 
                            output_dim=128, 
                            weights=[embedding_matrix], 
                            input_length=max_sequence_length, 
                            trainable=False))
    model_cnn.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Dropout(0.2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(256, activation='relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(3, activation='softmax'))
    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # EarlyStopping callback'i oluşturma
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # CNN Modeli eğitme
    print("\nCNN Modeli Eğitiliyor...")
    history_cnn = model_cnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping])
    # CNN Modeli değerlendirme
    test_loss_cnn, test_accuracy_cnn = model_cnn.evaluate(X_test, y_test, verbose=2)
    y_pred_prob_cnn = model_cnn.predict(X_test)
    y_pred_classes_cnn = np.argmax(y_pred_prob_cnn, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    print(f'CNN Modeli Test Seti Doğruluğu: {test_accuracy_cnn}')

    # Convert integer labels back to string labels
    reverse_label_mapping = label_encoder.classes_
    y_test_class_strings = np.array([reverse_label_mapping[label] for label in y_test_classes])
    y_pred_class_strings_cnn = np.array([reverse_label_mapping[label] for label in y_pred_classes_cnn])

    # Print classification report
    print("CNN Modeli Test Seti Sınıflandırma Raporu:")
    print(classification_report(y_test_class_strings, y_pred_class_strings_cnn, zero_division=1))
    
    
    # RNN Modeli
    model_rnn = Sequential()
    model_rnn.add(Embedding(input_dim=len(word_index) + 1, 
                            output_dim=128, 
                            weights=[embedding_matrix], 
                            input_length=max_sequence_length, 
                            trainable=False))
    model_rnn.add(SimpleRNN(units=128))
    model_rnn.add(Dropout(0.5))
    model_rnn.add(Dense(units=3, activation='softmax'))
    model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # EarlyStopping callback'i oluşturma
    early_stopping_rnn = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # RNN Modeli eğitme
    print("\nRNN Modeli Eğitiliyor...")
    history_rnn = model_rnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping_rnn])
    # RNN Modeli değerlendirme
    test_loss_rnn, test_accuracy_rnn = model_rnn.evaluate(X_test, y_test, verbose=2)
    y_pred_prob_rnn = model_rnn.predict(X_test)
    y_pred_classes_rnn = np.argmax(y_pred_prob_rnn, axis=1)
    print(f'RNN Modeli Test Seti Doğruluğu: {test_accuracy_rnn}')

    # Convert integer labels back to string labels
    y_pred_class_strings_rnn = np.array([reverse_label_mapping[label] for label in y_pred_classes_rnn])

    # Print classification report for RNN
    print("RNN Modeli Test Seti Sınıflandırma Raporu:")
    print(classification_report(y_test_class_strings, y_pred_class_strings_rnn, zero_division=1))
    
    # LSTM Modeli
    model_lstm = Sequential()
    model_lstm.add(Embedding(input_dim=len(word_index) + 1, 
                             output_dim=128, 
                             weights=[embedding_matrix], 
                             input_length=max_sequence_length, 
                             trainable=False))
    model_lstm.add(LSTM(units=128))
    model_lstm.add(Dense(3, activation='softmax'))
    model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # EarlyStopping callback'i oluşturma
    early_stopping_lstm = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # LSTM Modeli eğitme
    print("\nLSTM Modeli Eğitiliyor...")
    history_lstm = model_lstm.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping_lstm])
    # LSTM Modeli değerlendirme
    test_loss_lstm, test_accuracy_lstm = model_lstm.evaluate(X_test, y_test, verbose=2)
    y_pred_prob_lstm = model_lstm.predict(X_test)
    y_pred_classes_lstm = np.argmax(y_pred_prob_lstm, axis=1)
    print(f'LSTM Modeli Test Seti Doğruluğu: {test_accuracy_lstm}')

    # Convert integer labels back to string labels
    y_pred_class_strings_lstm = np.array([reverse_label_mapping[label] for label in y_pred_classes_lstm])

    # Print classification report for LSTM
    print("LSTM Modeli Test Seti Sınıflandırma Raporu:")
    print(classification_report(y_test_class_strings, y_pred_class_strings_lstm, zero_division=1))

else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin")
