import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_audio(file_path):
    """
    Load an audio file and return the audio time series and sample rate.
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate

def visualize_waveform(audio, sample_rate):
    """
    Visualize the waveform of the audio signal.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def save_audio(file_path, audio, sample_rate):
    """
    Save the audio signal to a file.
    """
    librosa.output.write_wav(file_path, audio, sample_rate)