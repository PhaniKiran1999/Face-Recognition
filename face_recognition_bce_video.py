"""
references : https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
"""
from keras.models import load_model

# model = load_model("C:/Users/phani/Downloads/Colab Archive/siamese_net_bce.h5", compile=False)
model = load_model("C:/Users/phani/Downloads/siamese_net_bce_v2.h5", compile=False)


import os
import face_recognition
import cv2
import numpy as np
import pickle

KNOWN_FACES_DIR = "known_faces" 
UNKNOWN_FACES_DIR = "unknown_faces"
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog" #cnn
IMAGE_SIZE = 96

def zscore(x):
	mean = np.mean(x)
	std = np.std(x)
	std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
	y = np.multiply(np.subtract(x, mean), 1 / std_adj)
	return y

# print("fetching encodings and labels")

# pickle_in1 = open("encodings/known_faces.pickle","rb")
# known_faces = pickle.load(pickle_in1)

# pickle_in2 = open("encodings/known_names.pickle","rb")
# known_names = pickle.load(pickle_in2)

print("fetching encodings and labels")

pickle_in1 = open("encodings/known_faces2.pickle","rb")
known_faces = pickle.load(pickle_in1)

pickle_in2 = open("encodings/known_names2.pickle","rb")
known_names = pickle.load(pickle_in2)


print("processing Video")
#test_video, VID_20200318_092439, excelsior, L_P, BBT2, *VID_20200322_120551, VID_20200322_115721
video = cv2.VideoCapture('test_videos/test_video.mp4')
count = 0
while True:
	ret, image = video.read()
	image = cv2.resize(image, (640, 360))

	if not ret:
		break

	locations = face_recognition.face_locations(image, model=MODEL)
	encodings = []
	for top, right, bottom, left in locations:
		img = cv2.resize(np.array(image[top:bottom ,left:right]),(IMAGE_SIZE, IMAGE_SIZE))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = np.expand_dims(img, axis=0)
		img = np.expand_dims(img, axis=-1)
		encodings.append(model.predict(zscore(img)))
	
	for face_encoding, face_location in zip(encodings, locations):
		results = [np.sqrt(np.sum(np.square(known_faces[i]-face_encoding))) for i in range(len(known_faces))]
		match = known_names[results.index(min(results))]
		print(f"Match found: {match},{min(results)}")
		# top_left = (face_location[3], face_location[0])
		# bottom_right = (face_location[1], face_location[2])
		color = [0, 255, 0]
		# cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
		top_left = (face_location[3], face_location[2])
		bottom_right = (face_location[1], face_location[2]+22)
		if min(results) < 15:				
			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
			cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), FONT_THICKNESS)
	count += 5
	video.set(1, count)
	cv2.imshow("TV", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()