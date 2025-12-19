# This file provides information about the models used in the project, including architecture details and performance metrics.

## Models for Emotion Recognition from Speech

This project implements two main types of deep learning models for recognizing emotions from speech audio: Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) layers.

### 1. Convolutional Neural Network (CNN)

- **Architecture**: The CNN model is designed to capture spatial hierarchies in the audio features extracted from the input audio signals. It consists of several convolutional layers followed by pooling layers, which help in reducing the dimensionality while preserving important features.
- **Input**: The input to the CNN model is typically a 2D representation of the audio signal, such as a spectrogram or MFCCs.
- **Output**: The output layer uses a softmax activation function to classify the audio into one of the predefined emotion categories (e.g., happy, sad, angry).

### 2. Recurrent Neural Network (RNN) with LSTM

- **Architecture**: The RNN model with LSTM layers is designed to capture temporal dependencies in the audio signals. LSTMs are particularly effective for sequence prediction problems due to their ability to remember long-term dependencies.
- **Input**: The input to the RNN model is a sequence of features extracted from the audio signal, such as MFCCs over time.
- **Output**: Similar to the CNN model, the output layer uses a softmax activation function for emotion classification.

### Performance Metrics

- **Accuracy**: The models are evaluated based on their accuracy in classifying emotions from the test dataset.
- **Confusion Matrix**: A confusion matrix is generated to visualize the performance of the models across different emotion classes.
- **F1 Score**: The F1 score is calculated to assess the balance between precision and recall for each emotion class.

### Conclusion

Both models are trained on datasets such as RAVDESS, TESS, or EMO-DB, and their performance is compared to determine the most effective approach for emotion recognition from speech. Further experiments can be conducted to fine-tune hyperparameters and improve model performance.