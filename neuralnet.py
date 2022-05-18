# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kHPxQv9qpo144L8PgMDuckxnrJ8kKSZP
"""

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

def get_weighted_sum(feature, weights, bias):
  wSum = float(0.0)
  for i in range (len(feature)):
    wSum += float(feature[i] * weights[i])

  wSum += float(bias)
  return wSum

def sigmoid(w_sum):
  sig = 1/(1+np.exp(-w_sum))
  return sig


def get_prediction (image, weights, bias):
  w_sum = get_weighted_sum(image, weights, bias)
  prediction = sigmoid(w_sum)
  return prediction


def main (imgArray):
  file_to_read = open("Plates_NN.pickle", "rb") # File containing example object
  loaded_object = pickle.load(file_to_read) # Load saved object
  file_to_read.close()
  #print(loaded_object.all_weights)

  image = np.array(imgArray) / 255
  #print (imgArray)
  predictions_set = []
  listOfLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

  for j in range(36):
    prediction = get_prediction (image, loaded_object.all_weights[j], loaded_object.all_bias[j])
    temp_tup = (listOfLabels[j], prediction)
    predictions_set.append(temp_tup)

  df = pd.DataFrame.from_records(predictions_set, columns =['Character', 'Prediction'])
  df['Prediction'] = df['Prediction'].astype(float).round(6)
  df.sort_values(by=['Prediction'], inplace=True, ascending=False)
  # print(df)

  topPrediction = str(df.iloc[0][0])
  print(topPrediction)


main(list(map(float, sys.argv[1:])))