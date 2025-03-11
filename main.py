# main.py
from keras.datasets import imdb
from keras import models
from keras import layers
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from scr import vectorize_sequences
import numpy as np

# 1. Cargar los datos de IMDB
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 2. Preprocesamiento: One-hot encoding
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Convertir las etiquetas a tipo float32
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 3. Preparar los datos de validación
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 4. Definir el modelo
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Visualizar la estructura del modelo (opcional)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# 5. Compilar el modelo
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. Entrenar el modelo
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,  # Reducido a 4 epochs para evitar overfitting
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 7. Evaluar el modelo
results = model.evaluate(x_test, y_test)
print("Resultados de la evaluación:", results)

# 8. Visualizar el historial de entrenamiento (pérdida y precisión)
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.subplot(1, 2, 2)
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 9. Usar el modelo para hacer predicciones
predictions = model.predict(x_test[:2])
print("Predicciones para las primeras dos revisiones:", predictions)
