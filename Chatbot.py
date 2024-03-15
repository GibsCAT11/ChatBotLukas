import random
import json
import pickle
import numpy as np
from keras.models import load_model

import nltk
from nltk.stem import WordNetLemmatizer

# Inicializar el lematizador de palabras de NLTK
lemmatizer = WordNetLemmatizer()

# Cargar las intenciones desde el archivo JSON
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Cargar las palabras, clases y modelo previamente guardados
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Función para limpiar y lematizar una oración
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Función para convertir una oración en una bolsa de palabras
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                # Establecer a 1 si la palabra está presente en la oración
                bag[i] = 1
    return np.array(bag)

# Función para predecir la clase (etiqueta) de una oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    # Encontrar el índice de la etiqueta con la probabilidad más alta
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

# Función para obtener una respuesta basada en la etiqueta predicha
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            # Seleccionar una respuesta aleatoria de las intenciones
            result = random.choice(i['responses'])
            break
    return result

# Bucle infinito para interactuar con el chatbot
while True:
    message = input("Usuario: ")
    # Predecir la etiqueta de la oración del usuario
    predicted_class = predict_class(message)
    
    # Obtener y mostrar la respuesta del chatbot
    response = get_response(predicted_class, intents)
    
    # Verificar si la respuesta es genérica y explorar otras opciones
    if response == "No entiendo":
        # Puedes personalizar esta parte para manejar respuestas específicas o consultas adicionales
        response = "No estoy seguro de cómo responder a eso. ¿Puedes ser más claro?"
    
    print("Chatbot:", response)