#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from tkinter import Tk, filedialog

# Muat model-model
model1 = load_model('C:\\Users\\fajar\\savemodel_daun.h5')
model2 = load_model('C:\\Users\\fajar\\savemodel_daun2.h5')

# Inisialisasi class_labels
class_labels = ['healty', 'early_blight', 'late_blight']

Tk().withdraw()
file_path = filedialog.askopenfilename(title="Pilih file gambar", filetypes=[("File gambar", "*.png;*.jpg;*.jpeg")])

if not file_path:
    print("Tidak ada file yang dipilih. Keluar.")
    exit()

img = cv2.imread(file_path)

img = cv2.resize(img, (224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Lakukan prediksi dengan model-model
prediction1 = model1.predict(img_tensor)
prediction2 = model2.predict(img_tensor)
ensemble_prediction = (prediction1 + prediction2) / 2  # Ambil rata-rata prediksi dari model-model

class_index = np.argmax(ensemble_prediction[0])
class_label = class_labels[class_index]

confidence_score = ensemble_prediction[0][class_index]

print(f"Kelas: {class_label}")
print(f"Skor Kepercayaan: {confidence_score * 100:.2f}%")

if class_label == 'healty':
    print('Daun Ini Sehat')
elif class_label == 'early_blight':
    print('Daun ini Sedikit Layu')
elif class_label == 'late_blight':
    print('Daun Ini Kering')
else:
    print('Objek tidak diketahui')

resized_frame = cv2.resize(img, (800, 800))

label_texts = [f"{class_labels[i]}: {ensemble_prediction[0][i] * 100:.2f}%" for i in range(len(class_labels))]

y_position = 60

for label_text in label_texts:
    cv2.putText(resized_frame, label_text, (10, y_position), cv2.FONT_HERSHEY_COMPLEX,
                1.0, (0, 0, 0), 2, cv2.LINE_AA)
    y_position += 30  

cv2.imshow('Hasil Klasifikasi Gambar', resized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




