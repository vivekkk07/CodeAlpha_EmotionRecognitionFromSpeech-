import os
import librosa
import numpy as np

def load_audio_files_from_directory(directory):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            file_path = os.path.join(directory, filename)
            audio_files.append(file_path)
    return audio_files

def load_data(raw_data_dir):
    audio_files = load_audio_files_from_directory(raw_data_dir)
    audio_data = []
    for file in audio_files:
        signal, sr = librosa.load(file, sr=None)
        audio_data.append((signal, sr))
    return audio_data

def load_labels(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            filename, emotion = line.strip().split(',')
            labels[filename] = emotion
    return labels

def load_dataset(raw_data_dir, label_file):
    audio_data = load_data(raw_data_dir)
    labels = load_labels(label_file)
    return audio_data, labels