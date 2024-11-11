import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('My_cnn_model.h5')

def process_image(img):
    img=img.resize((30,30))
    img=np.array(img)#numpy dizisine dönüştürdü
    img=img/255.0
    img=np.expand_dims(img,axis=0)#Görüntüye yeni bir boyut ekler ve onu dört boyutlu hale getirir
    return img

st.title("Sıtma Resmi Sınıflandırma Sıtma:") #Ana başlık
st.write("Resim seç ve model kanser olup olmadığını tahmin etsin")#Alt Başlık

file=st.file_uploader('Bir resim seç',type=['jpg','jpeg','png'])

if file is not None: #Dosyanın boş olmadığını kontrol eder
    img=Image.open(file).convert('RGB')#Resmi açar
    st.image(img,caption='Yüklenen resim')#Resmi okur
    image=process_image(img)#Görüntüyü işler
    prediction=model.predict(image)#Model ile tahmin yapar
    predicted_class = np.argmax(prediction, axis=1)#Tahmini sınıfa dönüştürür
    class_names=['Sıtma Değil','Sıtma']#Sınıf isimlerini tanımlar
    st.write(class_names[predicted_class[0]])#Tanımlanmış sınıfları yazdırır