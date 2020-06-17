'''
https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/
'''


import face_recognition
import os
import cv2
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from inception_resnet_v1 import *
model = InceptionResNetV1()

model.load_weights('facenet_weights.h5')

input_shape = model.layers[0].input_shape[1:3]

print("model input shape: ", model.layers[0].input_shape[1:]) #(160, 160, 3)
print("model output shape: ", model.layers[-1].input_shape[-1]) #128


KNOWN_FACES_DIR ="known_faces" #"known_faces dir"
MODEL = "hog" #cnn, hog

def zscore(x):
	mean = np.mean(x)
	std = np.std(x)
	std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
	y = np.multiply(np.subtract(x, mean), 1 / std_adj)
	return y

print("loading known faces")
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
		image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
		face_locations = face_recognition.face_locations(image, model=MODEL)
		
		for top, right, bottom, left in face_locations:
			image = cv2.resize(np.array(image[top:bottom ,left:right]),(160, 160))
			image = np.expand_dims(image, axis=0)
			encoding = model.predict(zscore(image))
			# encoding = model.predict(image/255)		
		known_faces.append(encoding)
		known_names.append(name)


print("saving face encoding..")
pickle_out1 = open("encodings/known_faces.pickle","wb")
pickle.dump(known_faces, pickle_out1)
pickle_out1.close()
print("saved face encoding in encodings/known_faces.pickle")

print("saving names..")
pickle_out2 = open("encodings/known_names.pickle","wb")
pickle.dump(known_names, pickle_out2)
pickle_out2.close()
print("saved names in encodings/known_names.pickle")