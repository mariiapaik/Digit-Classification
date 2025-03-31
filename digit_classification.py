import sys
import io

# Nastavenie kódovania na UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Cesta k tréningovým dátam
train_dir = 'D:/nn/exam/train'

# Parametre obrázkov
img_height, img_width = 28, 28
batch_size = 32

# Vytvorenie ImageDataGenerator pre načítanie a augmentáciu dát
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3  # 30% dát bude použitých na validáciu a testovanie
)

# Načítanie tréningových dát
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',  # Načítanie v grayscale
    class_mode='sparse',
    subset='training'  # Tréningová množina
)

# Načítanie validačných dát
validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',  # Načítanie v grayscale
    class_mode='sparse',
    subset='validation'  # Validačná množina
)

# Použitie 50% validačnej množiny ako testovacej množiny
test_size = int(0.5 * validation_generator.samples)
validation_generator.samples = validation_generator.samples - test_size

# Načítanie testovacích dát
test_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    subset='validation',  # Použitie validačnej množiny na testovanie
    shuffle=False
)

# Návrh modelu
model = Sequential([
    Input(shape=(img_height, img_width, 1)),  # Grayscale obrázky
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 tried (čísla 0-9)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Tréning modelu
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

# Vykreslenie priebehu tréningu a validácie
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predikcia na testovacej množine
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Skutočné triedy
y_true = test_generator.classes

# Konfúzna matica
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

# Výpočet metrik
report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
print("Classification Report:\n", report)
