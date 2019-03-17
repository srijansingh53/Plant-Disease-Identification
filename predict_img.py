from keras.models import load_model
import cv2
import numpy as np

model = load_model('models\model_final_bin.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('2.jpg')
img = cv2.resize(img,(256,256))
img = np.reshape(img,[1,256,256,3])

classes = model.predict_classes(img)


print (classes)
