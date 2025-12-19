import numpy as np
import librosa
import torch
import torch.nn.functional as F
from src.models.cnn import CNNModel
from src.models.rnn_lstm import RNNLSTMModel

class EmotionPredictor:
    def __init__(self, model_type='cnn', model_path='path/to/your/model.pth'):
        self.model_type = model_type
        self.model = self.load_model(model_path)
        self.labels = ['happy', 'sad', 'angry', 'neutral']  # Update with your actual labels

    def load_model(self, model_path):
        if self.model_type == 'cnn':
            model = CNNModel()
        elif self.model_type == 'rnn_lstm':
            model = RNNLSTMModel()
        else:
            raise ValueError("Model type not recognized. Choose 'cnn' or 'rnn_lstm'.")
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def extract_features(self, audio_path):
        signal, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs

    def predict(self, audio_path):
        features = self.extract_features(audio_path)
        features = torch.tensor(features).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        with torch.no_grad():
            output = self.model(features)
            probabilities = F.softmax(output, dim=1)
            predicted_label = self.labels[torch.argmax(probabilities).item()]
            return predicted_label, probabilities.numpy()

if __name__ == "__main__":
    predictor = EmotionPredictor(model_type='cnn', model_path='path/to/your/model.pth')
    audio_file = 'path/to/your/audio/file.wav'
    label, probs = predictor.predict(audio_file)
    print(f'Predicted Emotion: {label}, Probabilities: {probs}')