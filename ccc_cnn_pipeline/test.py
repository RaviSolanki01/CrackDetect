import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import os
import random
import pickle
from shutil import copyfile
import os, glob,shutil

get = os.getcwd()

IMG_SIZE = 50

def prepare_image(file):
	img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	return img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Loading Model...")
model = tf.keras.models.load_model("./Concrete_Crack_Classification_model.model") 
print("Model Loaded!!")

total = 0
true_neg = 0
true_pos = 0
false_neg = 0
false_pos = 0
errlist = []
fn = []
fp = []

print("Testing...")

with open('positive_test.txt', 'r') as a:
		for single_img in a.readlines():
			single_img = single_img[:-1]
			total = total+1
			prediction = model.predict([prepare_image(single_img)])
			if prediction[0][0] == 0: 		#	This concrete HAVE a crack.
				true_pos = true_pos +1
			elif prediction[0][0] == 1: 	#	This concrete DOES NOT have a crack.
				false_neg = false_neg +1
				fn.append(single_img)
			else:							#	Something went wrong ...
				total = total-1				
				errlist.append(single_img)

with open('negative_test.txt', 'r') as b:
		for single_nimg in b.readlines():
			single_nimg = single_nimg[:-1]
			total = total+1
			prediction = model.predict([prepare_image(single_nimg)])
			if prediction[0][0] == 0:		#	This concrete HAVE a crack.
				false_pos = false_pos +1
				fp.append(single_nimg)
			elif prediction[0][0] == 1:		#	This concrete DOES NOT have a crack.
				true_neg = true_neg +1
			else:							# 	Something went wrong ...
				total = total-1
				errlist.append(single_nimg)

print("Testing Completed")
with open('Residual.txt', 'w') as f:
	f.write("\nFalse Positives - \n")
	for item1 in fp:
		f.write("		%s\n" % item1)
	f.write("\nFalse Negatives - \n")
	for item2 in fn:
		f.write("		%s\n" % item2)
	f.write("\nImages that couldnt be compiled due to some error - \n")
	for item3 in errlist:
		f.write("		%s\n" % item3)

Accuracy = ((true_pos+true_neg)/total)*100
print("Accuracy - " + str(Accuracy))