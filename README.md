# Voice Command Recognition Project

## Overview

This project focuses on the design, implementation, and training of various machine learning models aimed at recognizing voice commands. The goal is to explore how different techniques can identify relevant patterns in audio signals and develop models capable of correctly classifying different voice commands recorded by various speakers under diverse conditions, such as background noise and different voice characteristics.

The main task is to classify short voice commands like "up", "left", "right", etc., from the Google Speech Commands dataset, ensuring robustness and reliability in real-world scenarios.

## Dataset

The dataset used for training the models is the [Google Speech Commands Dataset](https://www.tensorflow.org/datasets/community_catalog/huggingface/google_speech_commands), which contains short audio recordings (1-2 seconds) of simple voice commands spoken by various speakers. This dataset includes recordings with different accents, tones, and recording conditions, providing the necessary variety for training robust models.

## Preprocessing

To convert the audio data into a format suitable for model input, the following preprocessing steps were performed:

- Conversion of audio signals into spectrograms.
- Experimentation with three types of spectrograms: Linear Spectrograms, Mel Spectrograms, and Mel Frequency Cepstral Coefficients (MFCC).
- After evaluating the performance of different spectrogram types using a baseline model, Mel Spectrograms were chosen as they provided the best results.

Additionally, several hyperparameters of Mel Spectrograms (e.g., frame length, frame step, FFT length, and Mel bins) were fine-tuned for optimal performance.

## Models

Different model architectures were tested, combining Convolutional layers, Recurrent layers, and Transformers. These architectures were designed to extract local features and capture temporal dependencies in the audio signal.

### Model Architectures:
1. **Model 1:** Convolutional Layers Only
2. **Model 2:** Convolutional + Bi-directional Recurrent Layers
3. **Model 3:** Convolutional + Recurrent + Transformer Layers

### Results:
- Model 1: Accuracy = 89.78%
- Model 2: Accuracy = 89.90%
- Model 3: Accuracy = 90.32%

The final architecture was chosen based on the balance between performance and computational cost, selecting Model 2 (Convolutional + Bi-directional Recurrent Layers).

## Hyperparameter Tuning

A grid search was conducted to fine-tune the following hyperparameters:
- Number of filters in convolutional layers
- Number of units in LSTM layers
- Dense layer size
- Learning rate

The optimal configuration was found to be:
- `conv1_filters`: 32
- `lstm_units`: 96
- `dense_units`: 64
- `lr`: 0.001

## Data Augmentation

To further improve model generalization and reduce errors between similar-sounding words (e.g., "three" vs "tree"), the SpecAugment technique was applied. This technique involves the use of time and frequency masking on the Mel Spectrograms.

With SpecAugment, the following results were achieved:
- Training Accuracy: 91.83%
- Validation Accuracy: 97.22%
- Loss: 0.2925 (Training), 0.1233 (Validation)

## Conclusions

This project demonstrates that, in voice command recognition tasks (Keyword Spotting), the choice of data representation and the augmentation techniques play a crucial role in developing robust systems. By combining Mel Spectrograms with Convolutional and Recurrent layers, and employing data augmentation with SpecAugment, the model's performance was significantly improved, achieving a final validation accuracy of 97.22%.

The confusion between similar-sounding classes, such as "three" and "tree", remains an area for further improvement, which could be addressed by further enhancing the training data or refining the model architecture.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- librosa
- matplotlib
- scikit-learn
