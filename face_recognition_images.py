import matplotlib.pyplot as plt
import numpy as np

from inception_resnet_v1 import *
model = InceptionResNetV1()

model.load_weights('facenet_weights.h5')

input_shape = model.layers[0].input_shape[1:3]

print("model input shape: ", model.layers[0].input_shape[1:]) #(160, 160, 3)
print("model output shape: ", model.layers[-1].input_shape[-1]) #128


import face_recognition
import os
import cv2
import pickle

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 11 #lower the tolerance less chance of having false positive
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog" #cnn

def zscore(x):
	mean = np.mean(x)
	std = np.std(x)
	std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
	y = np.multiply(np.subtract(x, mean), 1 / std_adj)
	return y

print("loading known faces")
pickle_in1 = open("encodings/known_faces.pickle","rb")
known_faces = pickle.load(pickle_in1)

pickle_in2 = open("encodings/known_names.pickle","rb")
known_names = pickle.load(pickle_in2)


print("processing unknown faces")
for filename in os.listdir(UNKNOWN_FACES_DIR):
	image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
	image = cv2.resize(image, (640, 360))
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
		print(f"Match found: {match},{min(results)}")
		top_left = (face_location[3], face_location[0])
		bottom_right = (face_location[1], face_location[2])
		color = [0, 255, 0]
		cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

		top_left = (face_location[3], face_location[2])
		bottom_right = (face_location[1], face_location[2]+22)
		cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
		cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
	cv2.imshow(filename, image)
	cv2.waitKey(10000)
	cv2.destroyWindow(filename)


# def testing():
# 	print('testing')
# 	positive = 0
# 	negetive = 0
# 	test_dir = os.listdir('test_images')
# 	for name in test_dir:
# 		test_imgs = os.listdir(f'test_images/{name}')
# 		for test_img in test_imgs:
# 			image = cv2.imread(f'test_images/{name}/{test_img}')
# 			image = cv2.resize(image, (160, 160)) 
# 			image = np.expand_dims(image, axis=0)
# 			image = zscore(image)
# 			encoding = model.predict(image)
# 			results = [np.sqrt(np.sum(np.square(known_faces[i]-encoding))) for i in range(len(known_faces))]
# 			match = known_names[results.index(min(results))]
# 			if(match == name):
# 				positive += 1
# 			else:
# 				negetive += 1
# 			print(f'{name}/{test_img}',match, min(results))
# 	print('accuracy :',positive/(positive+negetive))

# testing()
