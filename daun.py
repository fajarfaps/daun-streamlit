import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os


# Muat model-model
model1 = load_model('C:\\Users\\fajar\\savemodel_daun.h5')
model2 = load_model('C:\\Users\\fajar\\savemodel_daun2.h5')

# Inisialisasi class_labels
class_labels = ['healty', 'early_blight', 'late_blight']
# Buat tautan ke direktori yang sesuai
home_directory = "C:\\Users\\fajar\\daun.py"  # Ganti dengan jalur direktori yang sesuai
about_us_directory = "C:\\Users\\fajar\\daun.py"  # Ganti dengan jalur direktori yang sesuai



# Tambahkan konten utama
add_selectbox = st.sidebar.selectbox(
    "Machine for Healty",
    ("Home", "About Us")
)

if add_selectbox == "Home":
    st.write("Selamat datang di halaman utama!")

if add_selectbox == "About Us":
    st.write("Selamat datang di halaman About Us.")
st.title('Aplikasi Pendeteksian Penyakit Daun')




# Tambahkan elemen unggah gambar
uploaded_image = st.file_uploader("Unggah gambar daun", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Proses gambar yang diunggah
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Lakukan prediksi dengan model-model
    prediction1 = model1.predict(img_array)
    prediction2 = model2.predict(img_array)
    ensemble_prediction = (prediction1 + prediction2) / 2  # Ambil rata-rata prediksi dari model-model

    class_index = np.argmax(ensemble_prediction[0])
    class_label = class_labels[class_index]
    confidence_score = ensemble_prediction[0][class_index]

    # Tampilkan hasil prediksi
    st.image(uploaded_image, caption='Gambar yang diunggah', use_column_width=True)
    st.write(f'Kelas: {class_label}')
    st.write(f'Skor Kepercayaan: {confidence_score * 100:.2f}%')

    if class_label == 'healty':
        st.write('Daun Ini Sehat')
    elif class_label == 'early_blight':
        st.write('Daun ini Sedikit Layu')
    elif class_label == 'late_blight':
        st.write('Daun Ini Kering')
    else:
        st.write('Objek tidak diketahui')
