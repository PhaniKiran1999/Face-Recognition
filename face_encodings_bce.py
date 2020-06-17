from keras.models import load_model

# model = load_model("C:/Users/phani/Downloads/Colab Archive/siamese_net_bce.h5", compile=False)
model = load_model("C:/Users/phani/Downloads/siamese_net_bce_v2.h5", compile=False)

import os
import face_recognition
import cv2
import numpy as np
import pickle

KNOWN_FACES_DIR = "known_faces"
MODEL = "hog" #cnn
IMAGE_SIZE = 96

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
			image = cv2.resize(np.array(image[top:bottom ,left:right]),(IMAGE_SIZE, IMAGE_SIZE))
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			image = np.expand_dims(image, axis=0)
			image = np.expand_dims(image, axis=-1)
			encoding = model.predict(zscore(image))		
		known_faces.append(encoding)
		known_names.append(name)

# print("saving face encoding..")
# pickle_out1 = open("encodings/known_faces.pickle","wb")
# pickle.dump(known_faces, pickle_out1)
# pickle_out1.close()
# print("saved face encoding in encodings/known_faces.pickle")

# print("saving names..")
# pickle_out2 = open("encodings/known_names.pickle","wb")
# pickle.dump(known_names, pickle_out2)
# pickle_out2.close()
# print("saved names in encodings/known_names.pickle")

print("saving face encoding..")
pickle_out1 = open("encodings/known_faces2.pickle","wb")
pickle.dump(known_faces, pickle_out1)
pickle_out1.close()
print("saved face encoding in encodings/known_faces2.pickle")

print("saving names..")
pickle_out2 = open("encodings/known_names2.pickle","wb")
pickle.dump(known_names, pickle_out2)
pickle_out2.close()
print("saved names in encodings/known_names2.pickle")

