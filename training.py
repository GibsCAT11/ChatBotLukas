# Importar las bibliotecas necesarias
import random
import json
import pickle
import numpy as np

# Importar NLTK (Natural Language Toolkit) para procesamiento de lenguaje natural
import nltk
from nltk.stem import WordNetLemmatizer

# Importar componentes específicos de Keras para construir la red neuronal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Inicializar el lematizador de palabras de NLTK
lemmatizer = WordNetLemmatizer()

# Cargar los datos de intenciones desde un archivo JSON llamado 'intents.json'
intents = json.loads(open('intents.json').read())

# Descargar recursos adicionales de NLTK (tokenización, lematización, etc.)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializar listas para almacenar palabras, clases e información de los documentos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Iterar sobre las intenciones y patrones para construir el conjunto de datos
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar las palabras en los patrones
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Agregar el patrón y la etiqueta a la lista de documentos
        documents.append((word_list, intent["tag"]))
        # Agregar la etiqueta a la lista de clases si no está presente
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematizar y filtrar las palabras para obtener un conjunto único
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guardar las palabras y clases en archivos pickle para su reutilización
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inicializar listas para almacenar características (entrada) y etiquetas (salida)
X = []  # Features (entrada)
Y = []  # Labels (salida)

# Construir las características (bolsas de palabras) y etiquetas a partir de los documentos
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        # Usar 1 si la palabra está presente en el patrón, 0 de lo contrario
        bag.append(1) if word in word_patterns else bag.append(0)

    # Agregar la bolsa de palabras y la etiqueta a las listas X e Y, respectivamente
    X.append(bag)
    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1
    Y.append(output_row)

# Convertir las listas a arrays NumPy
X = np.array(X)
Y = np.array(Y)

# Imprimir los arrays resultantes
print("X:")
print(X)
print("\nY:")
print(Y)

# Dividir los datos en conjuntos de entrenamiento y prueba
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

train_x = X[:split_index]
train_y = Y[:split_index]

test_x = X[split_index:]
test_y = Y[split_index:]

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo con los datos de entrenamiento
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save("chatbot_model.h5")

