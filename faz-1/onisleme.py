import pandas as pd
import numpy as np
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Dosya yolu
file_path = 'Dataset.xlsx'

# Hata kontrolü ve dosyanın varlığını kontrol etme
if os.path.exists(file_path):
    print("Dosya başarıyla açıldı. Veriler okunuyor...")

    # Excel dosyasını oku
    df = pd.read_excel(file_path)

    # Boş satırları bul ve çıkar
    df.dropna(inplace=True)

    # Önişleme adımları
    def preprocess_text(text):
        # Küçük harfe dönüştürme
        text = text.lower()
        # Sayıların kaldırılması
        text = re.sub(r'\d+', '', text)
        # Noktalama işaretlerinin kaldırılması
        text = re.sub(r'[^\w\s]', '', text)
        # Stopwords'lerin kaldırılması
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        # Kelimelerin köklerinin bulunması (Stemming)
        stemmer = PorterStemmer()
        stemmed_text = [stemmer.stem(word) for word in filtered_text]
        return ' '.join(stemmed_text)

    # Yeni bir DataFrame oluştur
    preprocessed_data = pd.DataFrame(columns=df.columns)

    print("Önişleme adımına başlanıyor...")

    # DataFrame üzerinde döngü
    for index, row in df.iterrows():
        # Önişleme yap
        preprocessed_row = row.copy()
        for column in df.columns:
            preprocessed_row[column] = preprocess_text(str(row[column]))
        # Önişlenmiş satırı yeni DataFrame'e ekle
        preprocessed_data = pd.concat([preprocessed_data, pd.DataFrame([preprocessed_row])], ignore_index=True)

    print("Önişleme tamamlandı.")

    # Önişlenmiş veriyi yeni bir Excel dosyasına kaydet
    preprocessed_data.to_excel('preprocessed_dataset.xlsx', index=False)

    print("Kelimeler bulunuyor...")
    # Tüm farklı kelimeleri bul
    unique_words = set()

    # Iterate over each cell in the preprocessed DataFrame
    for column in preprocessed_data.columns:
        for cell in preprocessed_data[column]:
            words = word_tokenize(cell)
            unique_words.update(words)

    # Convert set to list for easier manipulation
    unique_words_list = list(unique_words)

    # Create a DataFrame from the unique words
    unique_words_df = pd.DataFrame(unique_words_list, columns=['Unique Words'])

    # Save the unique words to a new Excel file
    unique_words_df.to_excel('unique_words.xlsx', index=False)

    print("Kelime bulma işlemi tamamlandı.")

else:
    print("Dosya bulunamadı veya açılamadı. Lütfen dosya yolunu kontrol edin.")
