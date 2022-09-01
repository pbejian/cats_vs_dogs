# Chien ou chat ?

L'application est exécutable en ligne à l'adresse suivante :

🚀 [https://pbejian-cats-vs-dogs-app-556gra.streamlitapp.com/](https://pbejian-cats-vs-dogs-app-556gra.streamlitapp.com/).


Il s'agit d'une classification binaire réalisée avec un réseau de neurones 
convolutif permettant de reconnaître des photos de chiens ou de chats.
    
La base d'images d'entraînement (contenant environ 20000 fichiers) est une 
version modifiée par [Cousera](https://www.coursera.org/) d'une base d'images 
de [Kaggle](https://www.kaggle.com/).  Le notebook Jupyter ayant 
servi à l'entraînement du modèle est probablement la propriété de Coursera 
(même si j'ai modifié un grand nombre de choses dans le notebook d'origine). 

Ainsi, pour des raisons de copyright, ni la base d'images, ni le notebook 
de création et d'entrainment du modèle ne figurent dans ce dépôt GitHub. 
Malgré tout, voici les détails du modèle choisi :

```python
model = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),            
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
  ])

model.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']) 
```

**NB -** Pour les besoins de l'application interactive, le modèle déjà entraîné a été 
stocké dans un fichier "pickle" qui se trouve dans le dépôt.