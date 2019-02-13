print("Importing libraries...")

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import os
import random
import pickle
import os, glob,shutil

get = os.getcwd()
X_pickle = get + "/X_concrete.pickle"
Y_pickle = get + "/y_concrete.pickle"

if os.path.exists(X_pickle):
	os.remove(X_pickle)
if os.path.exists(Y_pickle):
	os.remove(Y_pickle)

os.mknod("X_concrete.pickle")
os.mknod("y_concrete.pickle")

get = os.getcwd()
DATADIR = get + "/train" #insert the directory you'll be working with
IMG_SIZE = 50
CATEGORIES = ["Positive", "Negative"]
training_data = []

def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)

		for img in os.listdir(path):
			img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
			new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
			training_data.append([new_array, class_num])

def create_training_data():
	with open('positive_training.txt', 'r') as a:
		for single_img in a.readlines():
			single_img = single_img[:-1]
			img_array = cv2.imread(single_img, cv2.IMREAD_GRAYSCALE)
			new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
			training_data.append([new_array, 0])
		
	with open('negative_training.txt', 'r') as b:
		for single_nimg in b.readlines():
			single_nimg = single_nimg[:-1]
			nimg_array = cv2.imread(single_nimg, cv2.IMREAD_GRAYSCALE)
			nnew_array = cv2.resize(nimg_array, (IMG_SIZE, IMG_SIZE))
			training_data.append([nnew_array, 1])

print("Creating training data...")
create_training_data()
print("Training data successfully created!!")

print("Shuffling training data...")
random.shuffle(training_data)
print("Training data successfully shuffled!!")

X_data = []
y = []

for features, label in training_data:
	X_data.append(features)
	y.append(label)

print("X and y data successfully created!!")

print("Reshaping X data...")
X = np.array(X_data).reshape(len(X_data), IMG_SIZE, IMG_SIZE, 1)
print("X data successfully reshaped!!")

print("Saving the data...")
pickle_out = open("./X_concrete.pickle", "wb") #Replace the dots with the directory
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("./y_concrete.pickle", "wb") #Replace the dots with the directory
pickle.dump(y, pickle_out)
pickle_out.close()
print("Data successfully saved!!")
