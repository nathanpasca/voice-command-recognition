# Voice Command Recognition System
A Python-based voice command recognition system utilizing a CNN+LSTM deep learning model to classify audio commands ("yes," "no," "play," "stop") from a dataset of WAV files. This project was developed as part of a Voice and Image Recognition course mid project to demonstrate audio classification techniques using MFCC features.

## Prerequisites
Before running this system, make sure you have:
- Python 3.9+
- A labeled dataset (`command_dataset.zip`) containing WAV files organized in subdirectories (`yes`, `no`, `play`, `stop`)
- Required Python libraries (listed below)
- A working installation of FFmpeg for audio processing (if required by `librosa`)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/voice-command-recognition
cd voice-command-recognition
```

2. Install dependencies:
```bash
pip install tensorflow librosa numpy joblib
```

3. Prepare the dataset:
- Unzip the dataset file (`command_dataset.zip`) in the project directory to create a `command_dataset` folder with subdirectories `yes`, `no`, `play`, and `stop`, each containing WAV files.
```bash
unzip command_dataset.zip
```

4. Ensure FFmpeg is installed (if required by `librosa`):
- On Ubuntu: `sudo apt-get install ffmpeg`
- On macOS: `brew install ffmpeg`
- On Windows: Download and install FFmpeg from [ffmpeg.org](https://ffmpeg.org/) and add it to your system PATH.

## Usage
Run the script:
```bash
python3 voice_command_recognition.py
```

The system will:
- Load a pre-trained CNN+LSTM model (`cnn_lstm_command_model.h5`) and label encoder (`label_encoder.pkl`)
- Process audio files using MFCC (Mel-Frequency Cepstral Coefficients), delta, and delta-delta features
- Classify audio commands as "yes," "no," "play," or "stop" with confidence scores
- Save the trained model and label encoder for future use

Example usage for predicting a command:
```python
command, confidence = predict_command_cnn_lstm("test_yes.wav", model, label_encoder)
print(f"Predicted command: {command} with {confidence*100:.2f}% confidence")
```

Example output:
```
Audio length: 16000, Sample rate: 16000
Audio min: -0.1821, max: 0.2030, mean: 0.0001

Confidence scores:
no: 0.30%
play: 0.27%
stop: 0.60%
yes: 98.82%
Predicted command: yes with 98.82% confidence
```

## Note
- **Dataset**: The dataset (`command_dataset.zip`) must contain WAV files sampled at 16kHz, each approximately 1 second long, organized in subdirectories corresponding to the command labels (`yes`, `no`, `play`, `stop`).
- **Model**: The system assumes the presence of a pre-trained model (`cnn_lstm_command_model.h5`) and label encoder (`label_encoder.pkl`). If training from scratch, ensure the dataset is properly formatted and modify the script to include the training code.
- **Audio Preprocessing**: Audio files shorter than 1 second are padded, and longer files are truncated to ensure consistent input length.
- **Performance**: The model’s accuracy depends on the quality and diversity of the dataset. For production use, additional data augmentation and noise handling may be required.

## Academic Disclaimer
This project was developed as part of an academic mid project in Voice and Image Recognition. While it demonstrates audio classification concepts, it may require additional robustness, error handling, and optimization for production use.

## Acknowledgments
- [TensorFlow](https://www.tensorflow.org/)
- [Librosa](https://librosa.org/)
- [NumPy](https://numpy.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [FFmpeg](https://ffmpeg.org/)
