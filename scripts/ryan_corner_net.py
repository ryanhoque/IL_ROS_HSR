from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras import optimizers
import numpy as np
import os
import random
import cv2
import json

# build classifier
classifier = Sequential()
classifier.add(Conv2D(32, (7, 7), input_shape = (448, 448, 3), activation = 'relu'))
classifier.add(Conv2D(32, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
sgd = optimizers.SGD(lr=0.01)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# load data
PATH = "/nfs/diskstation/ryanhoque/corner_data/augmented_depth_images"
data = list()
success = os.listdir(PATH + '/success')
failure = os.listdir(PATH + '/failure')
for s in success:
	img = cv2.imread(PATH + '/success/' + s)
	data.append((img, 1))
for f in failure:
	img = cv2.imread(PATH + '/failure/' + f)
	data.append((img, 0))

random.shuffle(data)
x = [d[0]/255. for d in data]
y = [d[1] for d in data]

history = classifier.fit(x=np.array(x), y=np.array(y), batch_size=16, epochs=25, validation_split=0.2)
with open('net_output', 'w') as fh:
	fh.write(json.dumps(history.history))
