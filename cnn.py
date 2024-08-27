import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

# Configurações globais
img_height = 224
img_width = 224
batch_size = 32
data_dir = "./leaveDataset"
epochs = 25

# Divisão de dataset para treino (70%), validação (15%) e teste (15%)
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,  # 30% para validação e teste juntos
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_and_test_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Separando validação (15%) e teste (15%) dos 30% restantes
val_batches = int(0.5 * len(val_and_test_data))  # 50% dos 30% para validação

val_data = val_and_test_data.take(val_batches)  # Pegando a primeira metade para validação
test_data = val_and_test_data.skip(val_batches)  # Pegando a segunda metade para testes

# Obter o número de classes (categorias)
num_classes = len(train_data.class_names)
print(f"Num classes: {num_classes}")
print(train_data.class_names)

# Construção do modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="model_checkpoint.weights.h5", 
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True
)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} ended. Accuracy: {logs['accuracy']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")
        with open("training_log.txt", "a") as f:
            f.write(f"Epoch {epoch + 1}: accuracy={logs['accuracy']:.4f}, val_accuracy={logs['val_accuracy']:.4f}\n")

# Treinamento do modelo
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[tensorboard_callback, checkpoint_callback, CustomCallback()]
)

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc:.2f}")

# Gráficos finais de acurácia e perda
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treinamento')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Acurácia de Treinamento e Validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda de Treinamento')
plt.plot(epochs_range, val_loss, label='Perda de Validação')
plt.legend(loc='upper right')
plt.title('Perda de Treinamento e Validação')
plt.show()

# Função para classificar uma imagem
def classify_single_image(model, img_path, class_names):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predição: {class_names[predicted_class[0]]}")
    plt.show()

    print(f"Classe prevista: {class_names[predicted_class[0]]}")
    print(f"Confiança: {predictions[0][predicted_class[0]]:.2f}")
    return predicted_class[0], predictions[0][predicted_class[0]]

# Função para realizar testes com várias imagens
def batch_classification(model, test_data, class_names):
    correct_predictions = 0
    total_confidence = 0.0
    total_images = 0

    for images, labels in test_data:
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        for i in range(len(images)):
            total_confidence += predictions[i][predicted_classes[i]]
            if predicted_classes[i] == labels[i]:
                correct_predictions += 1
            total_images += 1

    if total_images == 0:
        print("Nenhuma imagem foi encontrada no conjunto de teste.")
        return

    # Cálculo da precisão e confiança média
    accuracy = correct_predictions / total_images
    avg_confidence = total_confidence / total_images
    
    # Exibindo os resultados
    print(f"\nPrecisão: {accuracy * 100:.2f}%")
    print(f"Confiança média: {avg_confidence:.2f}")
    print(f"Total de predições corretas: {correct_predictions}")
    print(f"Total de imagens: {total_images}")
    print(f"Confiança total acumulada: {total_confidence:.2f}")

# Executar o teste batch
batch_classification(model, test_data, train_data.class_names)

