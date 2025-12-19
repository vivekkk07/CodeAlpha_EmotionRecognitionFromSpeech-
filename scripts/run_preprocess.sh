#!/bin/bash

# This script automates the preprocessing of audio data for emotion recognition.

# Define directories
RAW_DATA_DIR="data/raw"
PROCESSED_DATA_DIR="data/processed"

# Create processed data directory if it doesn't exist
mkdir -p $PROCESSED_DATA_DIR

# Preprocess audio files
for file in $RAW_DATA_DIR/*.wav; do
    # Extract filename without extension
    filename=$(basename "$file" .wav)
    
    # Call the preprocessing script (assuming preprocess.py has a function called preprocess_audio)
    python src/data/preprocess.py --input "$file" --output "$PROCESSED_DATA_DIR/$filename_processed.wav"
done

echo "Preprocessing completed. Processed files are saved in $PROCESSED_DATA_DIR."