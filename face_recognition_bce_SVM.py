from keras.models import load_model

model = load_model("siamese_net_bce.h5", compile=False)
# model = load_model("siamese_net_bce_v2.h5", compile=False)


import os
import face_recognition
import cv2
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
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

print("fetching encodings and labels")

pickle_in1 = open("encodings/known_faces.pickle","rb")
known_faces = pickle.load(pickle_in1)

pickle_in2 = open("encodings/known_names.pickle","rb")
known_names = pickle.load(pickle_in2)

# print("fetching encodings and labels")

# pickle_in1 = open("encodings/known_faces2.pickle","rb")
# known_faces = pickle.load(pickle_in1)

# pickle_in2 = open("encodings/known_names2.pickle","rb")
# known_names = pickle.load(pickle_in2)

known_faces = np.concatenate(known_faces)
known_names = np.asarray([known_names])

x = known_faces
y = known_names.T

clf = svm.SVC(kernel='linear')
clf.fit(x, y)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x, y)

print('opening video')
#test_video, *VID_20200318_092439, excelsior, L_P, *BBT2, *VID_20200322_120551, VID_20200322_115721
video = cv2.VideoCapture(f'test_videos/test_video.mp4')
count = 0
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
		img = cv2.resize(np.array(image[top:bottom ,left:right]),(96, 96))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = np.expand_dims(img, axis=0)
		img = np.expand_dims(img, axis=-1)
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
	count += 5
	video.set(1, count)
	cv2.imshow('BBT_Test', image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()


# def testing():
# 	print('testing')
# 	positive = 0
# 	negetive = 0
# 	positive2 = 0
# 	negetive2 = 0
# 	test_dir = os.listdir("test_images")
# 	#'test_images'
# 	for name in test_dir:
# 		test_imgs = os.listdir(f'test_images/{name}')
# 		for test_img in test_imgs:
# 			image = cv2.imread(f'test_images/{name}/{test_img}')
# 			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# 			image = cv2.resize(image, (96, 96)) 
# 			image = np.expand_dims(image, axis=0)
# 			image = np.expand_dims(image, axis=-1)
# 			image = zscore(image)
# 			encoding = model.predict(image)
# 			match = clf.predict(encoding)[0]
# 			match_knn = knn.predict(encoding)[0]
# 			if(match == name):
# 				positive += 1
# 			else:
# 				negetive += 1
# 			if(match_knn == name):
# 				positive2 += 1
# 			else:
# 				negetive2 += 1
# 			print(f'{name}/{test_img}',match, match_knn)
# 	print('accuracy on svm linear kernal:',positive/(positive+negetive))
# 	print('accuracy of knn :',positive2/(positive2+negetive2))

# testing()