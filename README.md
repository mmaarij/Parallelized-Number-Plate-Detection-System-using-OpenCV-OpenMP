# Parallelized Number Plate Detection System using OpenCV and OpenMP

This project is a Parallelized Number Plate Detection System developed for Windows using Visual Studio. It utilizes OpenCV and OpenMP to efficiently detect number plates in images and recognize characters on them. The primary goal of this system is to speed up the process of detecting and recognizing number plates in images or video frames.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Technology Stack](#technology-stack)
- [Function Definitions](#function-definitions)
- [Parallelism Decomposition](#parallelism-decomposition)
- [Program Flow](#program-flow)
- [Usage Instructions](#usage-instructions)
- [Results](#results)
- [Python Neural Network for Character Recognition](#neural-network-for-character-recognition)
- [License](#license)

## Introduction

In today's world, image processing and task automation play a crucial role in law enforcement. A number plate detection system can digitalize this process, providing convenience to law enforcement agencies. This system has various applications, from catching over-speeding drivers to tracking criminal vehicle movements. It can also be integrated into dashboard cameras to detect, recognize, and store number plates during traffic incidents.

Conventional (serialized) systems may face efficiency challenges when dealing with a large number of vehicles in a single camera frame. Parallelized systems offer the potential for more efficient and faster results. Additionally, character recognition in traditional systems can be computationally expensive, while using a pre-trained neural network can significantly speed up the process.

## Objectives

The objectives of this project are as follows:

1. Implement a C++ system capable of detecting multiple number plates from a video stream.

2. Crop the detected number plates from the video frame and individually process them to detect characters.

3. Recognize the characters in the image by employing a recognition engine based on a Neural Net.

4. Output a corresponding string of characters representing the recognized number plate.

5. Store the image and the recognized plate number for future use.

To ensure the program runs efficiently and provides significant speedups compared to traditional systems, OpenCV is used for image processing, and OpenMP is used to parallelize the entire process.

## Technology Stack

- C++ (Visual Studio)
- OpenMP library for parallelization
- OpenCV library for image processing
- Python (Used to create and train the Neural Net, and invoked to recognize characters using the pre-trained Neural Net)

## Function Definitions
<table>
  <tr>
    <th>Function Name</th>
    <th>Use</th>
  </tr>
  <tr>
    <td>void main()</td>
    <td>
      <ul>
        <li>Load images using a for loop</li>
        <li>(Parallelized) For Each Image:
          <ul>
            <li>Load Harr-Cascade Number-Plate Classifier.</li>
            <li>Apply Median Blur to image</li>
            <li>Convert to Grayscale</li>
            <li>Use Viola-Jones Cascade Method (built into OpenCV) to detect all plates within the current frame</li>
            <li>Call processPlatesArray() function</li>
          </ul>
        </li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>void processPlatesArray(Mat& frame, Mat& grey, vector <Rect>& plates)</td>
    <td>
      <ul>
        <li>Process all plates in an image using a for loop</li>
        <li>(Parallelized) For each plate:
          <ul>
            <li>Call processSinglePlate() function</li>
          </ul>
        </li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>void processSinglePlate(Mat& croppedPlate)</td>
    <td>
      <ul>
        <li>Preprocess cropped plate:
          <ul>
            <li>Apply Binary Thresholding</li>
            <li>Apply Median Blur</li>
            <li>Erode the lines in the image</li>
            <li>Dilate the lines in the image</li>
            <li>Apply Inverse Binary Thresholding</li>
            <li>Apply Canny Edge Detection</li>
          </ul>
        </li>
        <li>Use the built-in OpenCV findContours() function to find all contours in the image</li>
        <li>Call sortContours() function</li>
        <li>Process all contours in the image using a for loop</li>
        <li>For each contour:
          <ul>
            <li>Select a bounding rectangle around the contour as the region of interest (ROI)</li>
            <li>If ROI is of acceptable parameters and does not overlap with the previous ROI, add the contour to the selected ROI list</li>
          </ul>
        </li>
        <li>Call postProcessImg() function</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>void postProcessImg(Mat& dilatedImg, vector <vector <Point>> contours, vector <int> selected_ROI)</td>
    <td>
      <ul>
        <li>Convert a copy of the image from grayscale to RGB.</li>
        <li>(Parallelized) For each character/contour:
          <ul>
            <li>Crop the character out of the grayscale image</li>
            <li>Call recognizeCharacter() function on the cropped character and get the return value</li>
            <li>Add the recognized character to the output string (recognized license plate number)</li>
            <li>Draw a bounding rectangle around the character on the colored image copy</li>
            <li>Label the character with recognized text on the colored image copy</li>
            <li>Save the processed image in the output directory with a filename set to the output string</li>
          </ul>
        </li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>string recognizeCharacter(Mat& croppedCharacter)</td>
    <td>
      <ul>
        <li>Preprocess cropped character:
          <ul>
            <li>Dilate lines</li>
            <li>Add a padded border</li>
            <li>Resize the image to 28x28 pixels</li>
          </ul>
        </li>
        <li>Parse a Python command as a string to be run as a system command in the form "python neuralnet.py p1 p2 p3 …... p784" where pN denotes pixel values to be passed as arguments to the Python script</li>
        <li>Call execSystemCommand() function with the parsed command as a parameter to run the Python script and send the cropped character image to the neural net for recognition</li>
        <li>Return the recognized character obtained as output from the neural net</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>void sortContours(vector <vector <Point>>& contours)</td>
    <td>
      <ul>
        <li>Sort contours based on x coordinates from left to right using insertion sort</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>string execSystemCommand(const char* cmd)</td>
    <td>
      <ul>
        <li>Create a pipe</li>
        <li>Execute a system command and store the output in a result string using the pipe</li>
        <li>Return the result string</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>string char_to_str(char c)</td>
    <td>
      <ul>
        <li>Convert a character to a string</li>
        <li>Return the converted string</li>
      </ul>
    </td>
  </tr>
</table>


## Parallelism Decomposition
![Parallelism Decomposition](Resources/Parallelism%20Decomposition.jpg)

## Program Flow
![Program Flow](Resources/Program%20Flow.jpg)

## Usage Instructions

### Step 1 - Install OpenCV

1. Go to [OpenCV Releases](https://github.com/opencv/opencv/releases) and download the Latest Release's EXE File.
2. Create a folder ‘OpenCV’ in the C drive.
3. Run the downloaded .exe file and select the extraction location as the folder you just created.
4. Add the bin folder (C:\OpenCV\opencv\build\x64\vc15\bin) to the Environment Variables path.
   ![Environment Variables Setup](Resources/Environment%20Variables.png)
6. Restart your computer.

### Step 2 - Install Python

1. Download the installer from [Python Downloads for Windows](https://www.python.org/downloads/windows/).
2. Run the installer and follow the setup steps.
3. Make sure the 'Add Python to Path' option is checked during the setup process.
4. Verify the Python installation by running "python -V" in the command line.

### Step 3 – Set Up Project

1. Clone the project from [GitHub](https://github.com/mmaarij/Parallelized-Number-Plate-Detection-System-using-OpenCV-OpenMP).
2. Run the Visual Studio Solution (.sln) file.
3. Verify that OpenCV is added to the project properties.
   - Go to Project > Properties.
   - In VC++ Directories > Build Directories, add “C:\OpenCV\opencv\build\include”.
   - In VC++ Directories > Library Directories, add “C:\OpenCV\opencv\build\x64\vc15\lib”.
   - In Linker > Input > Additional Dependencies, add “opencv_world455d.lib”.
4. Enable OpenMP in Visual Studio by going to Project > Properties > C/C++ > Language > OpenMP Support and enabling it to "Yes".
5. Rename a batch of images to be detected in the form "plate (1)", "plate (2)", and so on.
6. Go to [Bulk Resize Photos](https://bulkresizephotos.com/en) and bulk resize all images to 960 x 540.
7. In the project directory, navigate to Resources > Plates > test and paste all the renamed/resized images.

### Step 4 – Run the Program

1. Enter the number of images in the test directory.
2. The output will be stored inside the "RecognitionOutput" folder in the project directory.

## Results

The test was run on a batch of 10 images, containing 11 number plates. The entire batch was processed for 100 iterations, and the results were averaged out in the end.

For one single batch, the average time was as follows:

- Serial Execution Time: 57.0827 seconds
- Parallel Execution Time: 18.1869 seconds

The parallelized implementation resulted in a speedup of approximately 313.86%.

## Python Neural Network for Character Recognition

Character recognition in this project is performed using a Python neural network script. The script uses a Perceptron model for character recognition.

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
