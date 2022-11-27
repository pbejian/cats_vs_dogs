#-------------------------------------------------------------------------------
# Importation des modules 

import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import include as inc


#-------------------------------------------------------------------------------
# Application principale

st.title("Chient ou chat ?")
st.write("""
    ### Un exemple de classification binaire à l'aide d'un CNN

    Il s'agit d'une classification binaire réalisée avec un réseau de neurones 
    convolutif permettant de reconnaître les chiens et les chats.
    
    La base d'images (contenant environ 20000 fichiers) est une version modifiée 
    d'une base d'images de Kaggle. Le notebook Jupyter ayant servi à l'entraînement 
    du modèle est probablement la propriété de Coursera (même si j'ai modifié un grand
    nombre de choses dans le notebook d'origine). 

    Ainsi, pour des raisons de copyright,  ni la base d'images, ni le notebook de création et 
    d'entrainment du modèle ne figurent dans mon dépôt GitHub. Mais dans le fichier
    README.md figure malgré tout les caractéristiques du modèle CNN choisi.        
""")

inc.espace(1)

st.write("""
    #####  ➡️ Choix d'une image
""")

image_file = st.file_uploader(label="Choisir une image (de chient ou de chat).", type=["png","jpg","jpeg"])

inc.espace(1)

if image_file is not None:
    st.write("""
        #####  ➡️ Voici votre image
    """)
    img = inc.load_image(image_file)
    st.image(img, width=600)
    result = inc.prediction(image_file)
    inc.espace(1)    
    st.write(f"""
        #####  ➡️ Voici la prédiction
        Il semblerait que l'animal soit un **{result}**.
    """)

#-------------------------------------------------------------------------------
# Conclusion avec le lien vers les sources sur GitHub

st.markdown("""
    <hr>
""", unsafe_allow_html=True)
inc.espace(2)
st.write("""
📝 Sources de l'application :
[https://github.com/pbejian/cats_vs_dogs](https://github.com/pbejian/cats_vs_dogs)
""")
#-------------------------------------------------------------------------------