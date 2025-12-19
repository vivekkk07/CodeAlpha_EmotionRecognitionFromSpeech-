import numpy as np
import librosa

def compute_mfcc(audio_path, n_mfcc=13, sr=22050):
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCCs) from an audio file.

    Parameters:
    - audio_path: str, path to the audio file
    - n_mfcc: int, number of MFCCs to return
    - sr: int, sample rate for loading the audio file

    Returns:
    - mfccs: np.ndarray, array of shape (n_mfcc, T) where T is the number of frames
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    return mfccs

def compute_mfccs_for_directory(directory, n_mfcc=13, sr=22050):
    """
    Compute MFCCs for all audio files in a given directory.

    Parameters:
    - directory: str, path to the directory containing audio files
    - n_mfcc: int, number of MFCCs to return
    - sr: int, sample rate for loading the audio files

    Returns:
    - mfccs_dict: dict, mapping of audio file names to their MFCCs
    """
    mfccs_dict = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):  # Assuming audio files are in .wav format
            audio_path = os.path.join(directory, filename)
            mfccs = compute_mfcc(audio_path, n_mfcc, sr)
            mfccs_dict[filename] = mfccs
            
    return mfccs_dict