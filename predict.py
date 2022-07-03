from typing import final
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import os
from pathlib import Path
import shutil
import warnings
# %%

warnings.filterwarnings("ignore")

# Function to predict the result
def prediction(image):
    model = load_model('models\knee_xray.h5')

    x = tf.io.read_file(image)
    x = tf.io.decode_image(x,channels=3) 
    x = tf.image.resize(x,[299,299])
    x = tf.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    preds = model.predict(x)

    return preds

# Streamlit part starts
st.title("Knee Replacement Prediction App")
st.header("A Deep Learning technique to predict the knee replacement chances")
st.set_option('deprecation.showPyplotGlobalUse', False)

result = ["Minimal", "Healthy", "Moderate" ,"Doubtful", "Severe"]

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    directory = "uploaded_file"
    path = os.path.join(os.getcwd(), directory)
    p = Path(path)
    if not p.exists():
        os.mkdir(p)
    with open(os.path.join(path, uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer()) 
    
    file_location = os.path.join(path, uploaded_file.name)  
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.header("Classifying ......................................................")
    
    label = prediction(file_location)
    final_prediction = "Your knee condition is " + str(result[np.argmax(label)])
    st.subheader(final_prediction)
    shutil.rmtree(path)
        
    data = {}
    class_prediction_compi =  np.argmax(label)
    label = label[0]
    for i in range(len(result)):
        data[result[i]] = label[i]
    
    # Ploting of result comparision graph
    keys = list(data.keys())
    values = list(data.values())
        
    fig = plt.figure(figsize = (10, 5))
    plt.bar(keys, values, color ='maroon',width = 0.4)
    plt.xlabel("Category of prediction")
    plt.ylabel("Probability")
    plt.title("Prediction Graph")
    plt.plot()
    plt.show()
    plt.savefig('Plot.png')
    graph_file = Image.open("Plot.png")
    st.write("")
    st.header("** Prediction Graph **")
    st.image(graph_file, caption='Prediction Graph', use_column_width=True)

    # Showing ROC graph
    accurage_file = Image.open("Accuracy.png")
    st.write("")
    st.header("** Accuracy Graph **")
    st.image(accurage_file, caption='Accuracy Graph', use_column_width=True)
