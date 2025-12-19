# README for Raw Data

This directory contains the raw audio data used for the emotion recognition project. The data is sourced from various publicly available datasets, which include recordings of human speech expressing different emotions.

## Data Sources

1. **RAVDESS**: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) is a validated multimodal dataset of emotional speech and song. It includes recordings of actors performing various emotions.

2. **TESS**: The Toronto emotional speech set (TESS) consists of recordings of actors speaking short phrases in different emotional tones, including happiness, sadness, anger, and fear.

3. **EMO-DB**: The Berlin Database of Emotional Speech (EMO-DB) contains recordings of German speakers expressing a range of emotions, providing a rich resource for emotion recognition tasks.

## Data Format

The audio files are typically in WAV format, sampled at 16 kHz, and contain single-channel (mono) audio. Each file is labeled according to the emotion it represents, which is crucial for training and evaluating the emotion recognition models.

## Usage

The raw data can be accessed and processed using the scripts provided in the `src/data` directory. Ensure to follow the preprocessing steps outlined in `src/data/preprocess.py` to prepare the data for model training.

## Note

Please ensure to comply with the licensing agreements of the datasets used in this project when utilizing the data for research or commercial purposes.