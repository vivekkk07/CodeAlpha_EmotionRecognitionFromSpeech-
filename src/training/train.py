import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.mfcc import extract_mfcc
from src.models.cnn import create_cnn_model
from src.models.rnn_lstm import create_rnn_lstm_model

# Set parameters
BATCH_SIZE = 32
EPOCHS = 50
MODEL_TYPE = 'cnn'  # or 'rnn_lstm'
DATA_DIR = '../data/processed'
SAVE_MODEL_PATH = '../models/emotion_recognition_model.h5'

def train_model(model, train_data, train_labels):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
    return model

def main():
    # Load and preprocess data
    audio_files, labels = load_data(DATA_DIR)
    processed_data = preprocess_data(audio_files)
    mfcc_features = extract_mfcc(processed_data)

    # Create model
    if MODEL_TYPE == 'cnn':
        model = create_cnn_model(input_shape=mfcc_features.shape[1:])
    else:
        model = create_rnn_lstm_model(input_shape=mfcc_features.shape[1:])

    # Train model
    trained_model = train_model(model, mfcc_features, labels)

    # Save the trained model
    trained_model.save(SAVE_MODEL_PATH)
    print(f'Model saved to {SAVE_MODEL_PATH}')

if __name__ == '__main__':
    main()