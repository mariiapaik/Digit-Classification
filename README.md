# 🧠 Handwritten Digit Classification with Keras

## 🚀 Overview

This Python script uses **TensorFlow Keras** to train a neural network for classifying grayscale images of handwritten digits (0–9). It includes image loading, model training, evaluation, and visualization of performance metrics.

---

## 📂 Dataset Structure

The training data should be placed in the following folder structure:
D:/nn/exam/train
Each subfolder should contain images of the corresponding digit.

# 🧠 Handwritten Digit Classification with Keras

## ⚙️ How It Works

# Data Loading & Augmentation
ImageDataGenerator(rescale=1./255, validation_split=0.3)

# Model Architecture
Input(shape=(28, 28, 1))
→ Flatten()
→ Dense(128, activation='relu')
→ Dropout(0.5)
→ Dense(64, activation='relu')
→ Dropout(0.5)
→ Dense(10, activation='softmax')

# Training
optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
epochs = 50

# Evaluation
- Accuracy/Loss curves (train & val)
- Confusion matrix
- Classification report (precision, recall, F1-score)
