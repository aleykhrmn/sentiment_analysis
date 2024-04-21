import pandas as pd
import numpy as np
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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

    # Preprocessed DataFrame'deki her bir hücre üzerinde döngü
    for column in preprocessed_data.columns:
        for cell in preprocessed_data[column]:
            words = word_tokenize(cell)
            unique_words.update(words)

    # Set'i liste olarak dönüştürme (daha kolay işlem yapmak için)
    unique_words_list = list(unique_words)

    # Unique kelimeleri DataFrame'e dönüştürme
    unique_words_df = pd.DataFrame(unique_words_list, columns=['Unique Words'])

    # Unique kelimeleri yeni bir Excel dosyasına kaydetme
    unique_words_df.to_excel('unique_words.xlsx', index=False)

    print("Kelime bulma işlemi tamamlandı.")

    # Verileri yükleme
    kelimeler = pd.read_excel("unique_words.xlsx")
    metinler = pd.read_excel("preprocessed_dataset.xlsx")
    yildizlar = pd.read_excel("overall.xlsx")

    # Veri boyutlarını al
    sutun = kelimeler.shape[0]
    satir = metinler.shape[0]
    
    print("Matris oluşturuluyor...")
    # Tüm veri matrisini oluşturma
    all_data = np.zeros((satir, sutun + 1))

    # Kelime listesini oluşturma
    kelime_listesi = list(kelimeler['Unique Words'])

    # Matrisi oluşturma
    for i in range(satir):
        k = metinler.iloc[i, 0].split(" ")
        for j in range(len(k)):
            if k[j] in kelime_listesi:
                all_data[i, kelime_listesi.index(k[j])] = 1
                

    # Duygu puanlarını ekleme
    all_data[:, -1] = yildizlar.iloc[:, 0]
    
    print("Matris oluşturuldu.")
    
    # NumPy dizisini DataFrame'e dönüştürme
    all_data_df = pd.DataFrame(all_data, columns=kelime_listesi + ['Duygu Puanı'])

    # DataFrame'i ekrana yazdırma
    print(all_data_df)

    # DataFrame'i bir CSV dosyasına kaydetme
    all_data_df.to_csv('all_data.csv', index=False)

    print("Veri CSV dosyasına kaydedildi.")


    saved_df = pd.read_csv('all_data.csv')
    print("Kaydedilen CSV dosyasının boyutu:", saved_df.shape)
    print(saved_df.head())  # İlk birkaç satırı gösterme

else:
    print("Dosya bulunamadı.")
