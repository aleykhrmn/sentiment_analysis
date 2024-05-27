import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
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
else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")

# Modeli oluşturma
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_sequence_length))
model.add(SimpleRNN(units=128))
model.add(Dense(units=3, activation='softmax'))

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli özetle
model.summary()

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

# Test seti üzerinde doğruluk oranını hesaplama
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Seti Doğruluğu: {test_accuracy}')

# Test seti üzerinde sınıflandırma raporu
y_test_pred = model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

print("\nTest Seti Sınıflandırma Raporu:")
print(classification_report(y_test_true_classes, y_test_pred_classes, target_names=label_encoder.classes_, zero_division=1))
