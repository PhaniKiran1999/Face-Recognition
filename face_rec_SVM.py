import os
import sys
import numpy as np
import cv2
import pickle
from sklearn import svm
import face_recognition
from inception_resnet_v1 import *

UNKNOWN_FACES_DIR = "unknown_faces"
MODEL = 'hog'
FRAME_THICKNESS = 3
FONT_THICKNESS = 2


def zscore(x):
	mean = np.mean(x)
	std = np.std(x)
	std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
	y = np.multiply(np.subtract(x, mean), 1 / std_adj)
	return y

model = InceptionResNetV1()
model.load_weights('facenet_weights.h5')


print("fetching encodings and labels")

pickle_in1 = open("encodings/known_faces.pickle","rb")
known_faces = pickle.load(pickle_in1)

pickle_in2 = open("encodings/known_names.pickle","rb")
known_names = pickle.load(pickle_in2)

known_faces = np.concatenate(known_faces)
known_names = np.asarray([known_names])

x = known_faces
y = known_names.T


clf = svm.SVC(kernel='linear')
clf.fit(x, y)

print('opening video')
video = cv2.VideoCapture('test_videos/test_video.mp4')

while True:
	ret, image = video.read()
	image = cv2.resize(image, (640, 360))

	if not ret:
		break

	locations = face_recognition.face_locations(image, model=MODEL)
	encodings = []
	for top, right, bottom, left in locations:
		# img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		img = cv2.resize(np.array(image[top:bottom ,left:right]),(160, 160))
		img = np.expand_dims(img, axis=0)
		img = zscore(img)
		encodings.append(model.predict(img))
	
	for face_encoding, face_location in zip(encodings, locations):
		match = clf.predict(face_encoding)[0] 
		print(f"Match found: {match}")
		top_left = (face_location[3], face_location[0])
		bottom_right = (face_location[1], face_location[2])
		color = [0, 255, 0]
		cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

		top_left = (face_location[3], face_location[2])
		bottom_right = (face_location[1], face_location[2]+22)
		cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
		cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), FONT_THICKNESS)
	
	cv2.imshow('BBT_Test', image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()