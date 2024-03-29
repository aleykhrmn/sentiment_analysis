import pandas as pd
import numpy as np

kelimeler = pd.read_excel("unique_words.xlsx")
metinler = pd.read_excel("preprocessed_dataset.xlsx")
yildizlar = pd.read_excel("overall.xlsx")
sutun = kelimeler.shape[0]
satir = metinler.shape[0]
all_data = np.zeros((satir, sutun + 1))
print(yildizlar.iloc[0, 0])
print(kelimeler.iloc[0, 0])
print(metinler.iloc[1, 0])
print(kelimeler.iloc[sutun - 1, 0])
kelime_listesi = []
i = 0
while i < sutun:
  kelime_listesi.append(kelimeler.iloc[i, 0])
  i += 1

for i in range(satir):
  k = metinler.iloc[i, 0].split(" ")
  for j in range(len(k)):
    if k[j] in kelime_listesi:
      all_data[i, kelime_listesi.index(k[j])] = 1

all_data[:, -1] = yildizlar.iloc[:, 0]
print(all_data)


