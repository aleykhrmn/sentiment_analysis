import pandas as pd
import numpy as np
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from scipy.sparse import lil_matrix

# Dosya yolu
file_path = 'Dataset.xlsx'

try:
    # Dosyayı pandas ile oku
    df = pd.read_excel(file_path)
    print("Dosya başarıyla açıldı. Veriler okunuyor...")

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

    # Tüm veriyi önişle
    print("Önişleme adımına başlanıyor...")
    df_preprocessed = df.applymap(lambda x: preprocess_text(str(x)))
    print("Önişleme tamamlandı.")

    # Önişlenmiş veriyi yeni bir Excel dosyasına kaydet
    df_preprocessed.to_excel('preprocessed_dataset.xlsx', index=False)
    print("Önişlenmiş veri 'preprocessed_dataset.xlsx' adlı dosyaya kaydedildi.")

    print("Kelimeler bulunuyor...")
    # Tüm farklı kelimeleri bul
    unique_words = set()
    for column in df_preprocessed.columns:
        for cell in df_preprocessed[column]:
            words = word_tokenize(cell)
            unique_words.update(words)

    # Set'i liste olarak dönüştürme (daha kolay işlem yapmak için)
    unique_words_list = list(unique_words)

    # Unique kelimeleri DataFrame'e dönüştürme
    unique_words_df = pd.DataFrame(unique_words_list, columns=['Unique Words'])

    # Unique kelimeleri yeni bir Excel dosyasına kaydetme
    unique_words_df.to_excel('unique_words.xlsx', index=False)
    print("Kelimeler 'unique_words.xlsx' adlı dosyaya kaydedildi.")

    # Verileri yükleme
    kelimeler = unique_words_df
    metinler = df_preprocessed
    yildizlar = pd.read_excel("overall.xlsx")

    # Veri boyutlarını al
    sutun = kelimeler.shape[0]
    satir = metinler.shape[0]

    print("Matris oluşturuluyor...")
    # Sparse matris oluşturma
    all_data = lil_matrix((satir, sutun + 1), dtype=np.float32)

    # Kelime listesini oluşturma
    kelime_listesi = list(kelimeler['Unique Words'])

    # Sparse matrisi doldurma
    for i in range(satir):
        k = metinler.iloc[i, 0].split(" ")
        for word in k:
            if word in kelime_listesi:
                all_data[i, kelime_listesi.index(word)] = 1

    # Duygu puanlarını ekleme ve pozitif, negatif veya nötr olarak işaretleme
    for i in range(satir):
        duygu_puani = yildizlar.iloc[i, 0]
        if duygu_puani > 1:
            all_data[i, -1] = 1  # Pozitif
        elif duygu_puani < 1:
            all_data[i, -1] = 0  # Negatif

    print("Matris oluşturuldu.")

    # Sparse matrisi DataFrame'e dönüştürme
    all_data_df = pd.DataFrame.sparse.from_spmatrix(all_data, columns=kelime_listesi + ['Sentiment'])

    # DataFrame'i bir CSV dosyasına kaydetme
    all_data_df.to_csv('all_data.csv', index=False)
    print("Veri 'all_data.csv' adlı dosyaya kaydedildi.")

    # Birleştirme işlemi
    birlesik_veri = pd.merge(metinler, yildizlar, left_index=True, right_index=True)

    # Yeni sütun ekleme işlemi
    birlesik_veri['duygu'] = np.where(birlesik_veri['overall'] == 0, '0', np.where(birlesik_veri['overall'] == 1, '1', np.nan))

    # Birleştirilmiş veriyi Excel'e yazdırma
    birlesik_veri.to_excel("data.xlsx", index=False)
    print("Önişlenmiş veri ve duygular 'data.xlsx' adlı dosyaya kaydedildi.")

except FileNotFoundError:
    print("Dosya bulunamadı.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
