import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

# Configurações globais
img_height = 224
img_width = 224
data_dir = "./leaveDataset"
batch_size = 32

# Carregar o conjunto de dados para obter as classes
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Obter o número de classes (categorias)
num_classes = len(train_data.class_names)
print(f"Num classes: {num_classes}")
print(train_data.class_names)

# Reconstrução do modelo CNN
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

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Carregar os pesos salvos
model.load_weights("model_checkpoint.weights.h5")
print("Pesos do modelo carregados com sucesso.")

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
def batch_classification(model, class_names):
    test_images = [
        "./leaveDataset/Raspberry___healthy/image (1000).JPG",
        "./leaveDataset/Apple___Apple_scab/image (200).JPG",
        "./leaveDataset/Corn___Common_rust/image (350).JPG",
        "./leaveDataset/Tomato___Leaf_Mold/image (720).JPG"
    ]
    
    # Classes esperadas para as imagens acima 
    expected_classes = [24, 0, 9, 33]  
    
    correct_predictions = 0
    total_confidence = 0.0
    
    for i, img_path in enumerate(test_images):
        print(f"\nClassificando imagem: {img_path}")
        predicted_class, confidence = classify_single_image(model, img_path, class_names)
        total_confidence += confidence
        # Comparar a classe prevista com a classe esperada
        print(predicted_class)
        print(expected_classes[i])
        if predicted_class == expected_classes[i]:
            correct_predictions += 1

    # Cálculo da precisão
    accuracy = correct_predictions / len(test_images)
    avg_confidence = total_confidence / len(test_images)
    
    print(f"\nPrecisão: {accuracy * 100:.2f}%")
    print(f"Confiança média: {avg_confidence:.2f}")

# Executar o teste batch
batch_classification(model, train_data.class_names)
