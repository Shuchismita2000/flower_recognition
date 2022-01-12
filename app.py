import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model=load_model('FlowerClassification.h5')

image=st.file_uploader('UPLOAD ANY IMAGE OF FLOWER (Here You can upload any of Daisy,Dandelion,Rose,Sunflower ,Tulip )', type=['png', 'jpg'], accept_multiple_files=False)
flower=Image.open(image)
st.image(flower)

img = cv2.imread(image.name,cv2.IMREAD_COLOR)
img = cv2.resize(img, (150,150))
a=np.array(img)
a=a/255
a=a.reshape(1,150,150,3)
p=model.predict(a)
z=np.argmax(p[0])
label={0:'Daisy',1:'Dandelion',2:'Rose',3:'Sunflower',4:'Tulip'}
label[z]

