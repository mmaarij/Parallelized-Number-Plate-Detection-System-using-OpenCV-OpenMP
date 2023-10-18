# Parallelized Number Plate Detection System using OpenCV and OpenMP

This project is a Parallelized Number Plate Detection System developed for Windows using Visual Studio. It utilizes OpenCV and OpenMP to efficiently detect number plates in images and recognize characters on them. The primary goal of this system is to speed up the process of detecting and recognizing number plates in images or video frames.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [Performance](#performance)
- [Python Neural Network for Character Recognition](#neural-network-for-character-recognition)
- [License](#license)

## Introduction

The **Parallelized Number Plate Detection System** is designed to detect and recognize number plates in images. It takes advantage of parallel processing using OpenMP to enhance the detection and recognition process. This system is particularly useful for scenarios where multiple plates need to be processed quickly and efficiently.

## Dependencies

To build and run this project, you will need the following dependencies:

- [OpenCV](https://opencv.org/): Open Source Computer Vision Library.
- [OpenMP](https://www.openmp.org/): The Open Multi-Processing API for parallel programming.
- [Python](https://www.python.org/downloads/): Python used for neural network based character recognition.

## Installation

To get started with this project, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/mmaarij/Parallelized-Number-Plate-Detection-System-using-OpenCV-OpenMP.git
   ```

2. Open the project in Visual Studio.

3. Set up OpenCV and OpenMP dependencies according to your environment and project settings.
4. Install Python

5. Build and run the project.

## Usage

1. After building the project, run the executable.

2. The program will prompt you to enter the number of images in the "Resources > Plates > test" directory.

3. The system will process the images, detecting and recognizing number plates.

4. The results will be saved in the "RecognitionOutput" directory.

## How it Works

1. The system uses the Viola-Jones object detection framework to detect potential number plates in the images.

2. Detected plates are processed, and regions of interest (ROIs) are selected based on certain criteria, such as size and position.

3. These ROIs are passed to the character recognition function, which extracts characters and sends them to a neural network for recognition. The neural network is executed as a system command using Python. See [Python Neural Network for Character Recognition](#neural-network-for-character-recognition).

4. The recognized characters are labeled on the image, and rectangles are drawn around them.

5. The processed images are saved in the "RecognitionOutput" directory.

## Performance

The system's performance is significantly improved through parallelization using OpenMP. The program's execution time is significantly reduced compared to a serial implementation.

- Parallel Execution Time: 18.1869 seconds
- Serial Execution Time: 57.0827 seconds

This project is designed for optimal speed and efficiency in plate detection and character recognition.

## Neural Network for Character Recognition

Character recognition in this project is performed using a Python neural network script. The script is provided in the neuralnet.py file. It uses a Perceptron model for character recognition.

The Python script reads a saved neural network object from the "Plates_NN.pickle" file, which contains the trained model's weights and biases. It then processes the input image and returns a prediction for the recognized character.

Here is an overview of the neuralnet.py script:

```python
import numpy as np
import pandas as pd
import pickle
import sys

class Perceptron:
  all_weights = []
  all_bias = []

  def __init__(self, weights, bias):
    self.all_weights = weights
    self.all_bias = bias

# Functions for weighted sum, sigmoid, and prediction

def get_weighted_sum(feature, weights, bias):
  # Calculate the weighted sum of features
  wSum = float(0.0)
  for i in range(len(feature)):
    wSum += float(feature[i] * weights[i])

  wSum += float(bias)
  return wSum

def sigmoid(w_sum):
  # Apply sigmoid activation function
  sig = 1 / (1 + np.exp(-w_sum))
  return sig

def get_prediction(image, weights, bias):
  # Get the prediction for the input image
  w_sum = get_weighted_sum(image, weights, bias)
  prediction = sigmoid(w_sum)
  return prediction

def main(imgArray):
  # Load the trained neural network from a file
  file_to_read = open("Plates_NN.pickle", "rb")
  loaded_object = pickle.load(file_to_read)
  file_to_read.close()

  image = np.array(imgArray) / 255

  predictions_set = []
  listOfLabels = ['A', 'B', 'C', ...]  # List of possible characters

  # Get predictions for all characters
  for j in range(36):
    prediction = get_prediction(image, loaded_object.all_weights[j], loaded_object.all_bias[j])
    temp_tup = (listOfLabels[j], prediction)
    predictions_set.append(temp_tup)

  df = pd.DataFrame.from_records(predictions_set, columns=['Character', 'Prediction'])
  df['Prediction'] = df['Prediction'].astype(float).round(6)
  df.sort values(by=['Prediction'], inplace=True, ascending=False)

  # Get the character with the highest prediction
  topPrediction = str(df.iloc[0][0])
  print(topPrediction)

main(list(map(float, sys.argv[1:])))
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
