import face_recognition
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

from inception_resnet_v1 import *
model = InceptionResNetV1()

model.load_weights('facenet_weights.h5')

input_shape = model.layers[0].input_shape[1:3]

print("model input shape: ", model.layers[0].input_shape[1:]) #(160, 160, 3)
print("model output shape: ", model.layers[-1].input_shape[-1]) #128


KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6 #lower the tolerance less chance of having false positive
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog" #cnn, hog

print("loading known faces")
pickle_in1 = open("encodings/known_faces.pickle","rb")
known_faces = pickle.load(pickle_in1)

pickle_in2 = open("encodings/known_names.pickle","rb")
known_names = pickle.load(pickle_in2)


print("processing unknown faces")
video = cv2.VideoCapture('test_videos/test_video.mp4')

while True:
	ret, image = video.read()
	image = cv2.resize(image, (640, 360))

	if not ret:
		break

	locations = face_recognition.face_locations(image, model=MODEL)
	encodings = []
	# count = 123
	for top, right, bottom, left in locations:
		# img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		img = cv2.resize(np.array(image[top:bottom ,left:right]),(160, 160))
		img = np.expand_dims(img, axis=0)/255
		encodings.append(model.predict(img))
	
	for face_encoding, face_location in zip(encodings, locations):
		results = [np.sqrt(np.sum(np.square(known_faces[i]-face_encoding))) for i in range(len(known_faces))]	
		match = known_names[results.index(min(results))]
		print(f"Match found: {match}, {min(results)}")
		top_left = (face_location[3], face_location[0])
		bottom_right = (face_location[1], face_location[2])
		color = [0, 255, 0]
		cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

		top_left = (face_location[3], face_location[2])
		bottom_right = (face_location[1], face_location[2]+22)
		cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
		cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), FONT_THICKNESS)
	
	cv2.imshow("BBT_test", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()
