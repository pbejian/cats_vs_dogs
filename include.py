import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import streamlit as st
import pickle 


def espace(n):
    """
    Cette fonction ne renvoie rien mais affiche n lignes vides
    dans une application streamlit.
    """
    for _ in range(n):
        st.write("")
    return None


def load_image(image_file):
    """
    Ouvre le fichier passé en argument et le transforme en une image PIL.
    """
    img = Image.open(image_file)
    return img


def prediction(filename):
    """
    Renvoie la chaîne de caractère "chien" ou "chat" en fonction de la
    prédiction faite par le modèle CNN. On rappelle que ce modèle déjà
    entraîné est stocké dans le fichier 'my_model.pkl'.
    """
    model_filename = 'my_model.pkl'
    loaded_model = pickle.load(open(model_filename, 'rb'))
    img = image.load_img(filename, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x/255.
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    result = loaded_model.predict(images)    
    result = result[0][0]
    if round(result) == 0:
        return "chat 🐱"
    elif round(result) == 1:
        return "chien 🐶"
    else:
        return "Erreur"
