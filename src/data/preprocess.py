import os
import librosa
import numpy as np

def load_audio_file(file_path):
    """Load an audio file and return the audio time series and sample rate."""
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate

def normalize_audio(audio):
    """Normalize the audio signal to have zero mean and unit variance."""
    return (audio - np.mean(audio)) / np.std(audio)

def segment_audio(audio, segment_length=2.0, sample_rate=22050):
    """Segment the audio into overlapping segments."""
    segment_samples = int(segment_length * sample_rate)
    segments = []
    for start in range(0, len(audio), segment_samples // 2):
        end = start + segment_samples
        if end <= len(audio):
            segments.append(audio[start:end])
    return segments

def preprocess_audio(file_path):
    """Preprocess the audio file: load, normalize, and segment."""
    audio, sample_rate = load_audio_file(file_path)
    normalized_audio = normalize_audio(audio)
    segments = segment_audio(normalized_audio)
    return segments, sample_rate

def save_processed_data(segments, output_dir, base_filename):
    """Save processed audio segments to the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_dir, f"{base_filename}_segment_{i}.wav")
        librosa.output.write_wav(output_file, segment, sr=22050)