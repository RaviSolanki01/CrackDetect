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
import json
import matplotlib.pyplot as plt

get = os.getcwd()
# DATADIR = get + "/train" #insert the directory you'll be working with
IMG_SIZE = 50
CATEGORIES = ["Positive", "Negative"]
training_data = []

print("Loading the data...")
pickle_in = open("./X_concrete.pickle", "rb") #Replace the dots with the directory
X = pickle.load(pickle_in)
pickle_in = open("./y_concrete.pickle", "rb") #Replace the dots with the directory
y = pickle.load(pickle_in)
print("Data successfully loaded!")

X = X / 255.0

model = Sequential()

model.add(Conv2D(128, (3, 3), activation = "relu", input_shape = (IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Conv2D(128, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Flatten())
model.add(Dense(258, activation = "relu"))

model.add(Dense(1, activation = "sigmoid"))

print("Compiling the model...")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
print("Model successfully compiled!!")

print("Fitting the model...")
mod = model.fit(X, y, batch_size = 128, epochs = 10, validation_split = .2)
print("Model successfully fitted!!")

with open('logs.json', 'w') as f:
    json.dump(mod.history, f)
    
print("Saving the model...")
model.save("./Concrete_Crack_Classification_model.model") #Replace the dots with the directory
print("Model successfully saved!!")

def getJSON(filepathname):
    with open(filepathname,'r') as fp:
        return json.load(fp)

myobj = getJSON('./logs.json')

accuracy_list = myobj.get("acc")
validation_acc_list = myobj.get("val_acc")
legnd = 0
accuracy_inpercentage=[]
for i in accuracy_list:
    accuracy_inpercentage.append(i*100)
    legnd = i*100

valacuuracy_inpercentage=[]
for j in validation_acc_list:
    valacuuracy_inpercentage.append(j*100)
    
fig = plt.figure()
plt.plot(accuracy_inpercentage, label = "Training Accuracy")
plt.plot(valacuuracy_inpercentage, label = "Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Final Accuracy = "+ str(legnd))
plt.show()
fig.savefig('Accuracy.jpg')
