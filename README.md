# Real-Time Fight Detection System

## Overview

This project is a real-time fight detection system using deep learning. The system extracts frames from a dataset of fight and non-fight videos, processes them using ResNet152 for feature extraction, and trains an RNN model to classify the frames. Once trained, the model is used to analyze live video feeds to detect fights in real-time.

This project was developed as part of a Deep Learning course in the 3rd year of university by **İlkay Gökbudak**, **Ömer**, and **Berkant**.

## Features

- Extracts frames from fight and non-fight videos
- Uses ResNet152 for feature extraction
- Trains an RNN model on extracted features
- Performs real-time fight detection on live video feeds

## Technologies Used

- **Python**
- **TensorFlow/Keras** (for deep learning models)
- **ResNet152** (for feature extraction)
- **Recurrent Neural Networks (RNNs)** (for fight classification)
- **OpenCV** (for video processing)
- **NumPy, Pandas** (for data handling)

## File Structure

```
project_root/
│-- dataset/
│   ├── videos/
│   │   ├── fight/ (empty)
│   │   ├── noFight/ (empty)
│-- output/ (empty)
│-- handled_features/
│-- rnn_models/
│-- handle_frames.py
│-- handle_features.py
│-- train.py
│-- main.py
│-- Kavga-Tespiti_Rapor.pdf (Project Report)
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Download Dataset & Pretrained Models

To use the preprocessed dataset, extracted features, and trained model, download them from the following link: [Google Drive Link](https://drive.google.com/drive/folders/1Ga2j4wuFt9aGu25GQbExCNiFEJer0Csh)

After downloading, place the files in the appropriate directories:

- Copy `fight/` and `noFight/` folders to `dataset/videos/`
- Copy `handled_features/`, `output/`, and `rnn_models/` folders to the root project directory

Alternatively, if you want to use your own dataset, you can follow the steps below.

## Running the Project from Scratch

### Step 1: Extract Frames from Videos

Run the following command to extract frames from the dataset:

```bash
python handle_frames.py -d dataset/videos -o output
```

This script extracts frames from videos and saves them in the `output/` directory.

### Step 2: Extract Features Using ResNet152

```bash
python handle_features.py
```

This script extracts features from the frames and saves them in `handled_features/`.

### Step 3: Train the RNN Model

Run the following command to train the model using extracted features:

```bash
python train.py -d handled_features -s rnn_models
```

This script trains an RNN model and saves it in the `rnn_models/` directory.

### Step 4: Run Real-Time Detection

To test the trained model on live video feed, run:

```bash
python main.py -m rnn_models
```

This will start the webcam and detect fights in real-time.

## Dataset Credit

This dataset was obtained from [fight-detection-surv-dataset](https://github.com/seymanurakti/fight-detection-surv-dataset). Special thanks to the original creators.

## Project Report

For detailed explanations, methodology, and results, check the project report:\
[Kavga-Tespiti\_Rapor.pdf](Kavga-Tespiti_Rapor.pdf)

## Contribution

Feel free to contribute to this project by submitting issues or pull requests.

## License

This project is licensed under the MIT License.

