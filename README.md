# Emotion Recognition from Speech

This project aims to recognize human emotions from speech audio using deep learning and speech signal processing techniques. The primary emotions targeted in this project include happy, angry, and sad.

## Project Structure

The project is organized into the following directories and files:

- **data/**: Contains raw, processed, and external datasets.
  - **raw/**: Contains the original audio files used for training and evaluation.
  - **processed/**: Contains the processed audio data ready for model training and evaluation.
  - **external/**: Contains any external datasets or resources that may be used in the project.

- **notebooks/**: Contains Jupyter notebooks for data exploration and feature extraction.
  - **01-exploration.ipynb**: Exploratory data analysis, visualizing the dataset, and understanding the distribution of emotions.
  - **02-feature-extraction.ipynb**: Focuses on extracting features from the audio data, specifically Mel-Frequency Cepstral Coefficients (MFCCs).

- **src/**: Contains source code for data loading, preprocessing, feature extraction, model definitions, training, and inference.
  - **data/**: 
    - **load_data.py**: Functions to load the audio data from the specified directories.
    - **preprocess.py**: Functions for preprocessing the audio data, such as normalization and segmentation.
  - **features/**: 
    - **mfcc.py**: Functions to compute Mel-Frequency Cepstral Coefficients (MFCCs) from audio signals.
  - **models/**: 
    - **cnn.py**: Defines a Convolutional Neural Network (CNN) model for emotion recognition.
    - **rnn_lstm.py**: Defines a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers for emotion recognition.
  - **training/**: 
    - **train.py**: Contains the training loop for the models, including loss calculation and optimization.
  - **inference/**: 
    - **predict.py**: Functions for making predictions on new audio samples using the trained models.
  - **utils/**: 
    - **audio_utils.py**: Utility functions for audio processing, such as loading audio files and visualizing waveforms.

- **models/**: Contains information about the models used in the project, including architecture details and performance metrics.

- **experiments/**: Contains configuration settings for experiments, such as hyperparameters and dataset paths.
  - **config.yaml**: YAML file for experiment configurations.

- **scripts/**: Contains shell scripts for automating processes.
  - **run_preprocess.sh**: Automates the preprocessing of audio data.
  - **run_train.sh**: Automates the training process for the models.

- **requirements.txt**: Lists the Python dependencies required for the project.

- **environment.yml**: Used for creating a conda environment with the necessary packages.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd speech-emotion-recognition
   ```

2. Create a conda environment:
   ```
   conda env create -f environment.yml
   conda activate speech-emotion-recognition
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

- Use the Jupyter notebooks in the `notebooks/` directory for exploratory data analysis and feature extraction.
- The source code in the `src/` directory can be used for data loading, preprocessing, model training, and inference.
- Modify the `experiments/config.yaml` file to adjust hyperparameters and dataset paths as needed.

## Acknowledgments

This project utilizes datasets such as RAVDESS, TESS, and EMO-DB for training and evaluation.