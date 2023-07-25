import joblib as jb
import numpy as np

model = jb.load('C:/ESPOCH/Mineria_Proyecto/ALGORITMOS_SL/produccion/models/Modelo0.291.pkl')

#X_test = np.array([9,9,7650,4,19.75308642,2320,131767,14200,71.3,33])  # Debe dar un valor 0
X_test = np.array([41, 1.205882, 10269, 10, 4.451451, 1297, 83849 , 10322,  0.0, 46])  # Debe dar un valor 1


# Obtener las probabilidades de las clases
probabilities = model.predict(X_test.reshape(1, -1))

# Definir un umbral para decidir si es 0 o 1
threshold = 0.5

# Redondear la probabilidad de la clase 1 a 0 o 1 según el umbral
prediction = (probabilities[0] > threshold).astype(int)

if prediction == 1:
    print('Nivel de toxicidad Alto:', prediction)
elif prediction == 0:
    print('Nivel de toxicidad Bajo', prediction)
