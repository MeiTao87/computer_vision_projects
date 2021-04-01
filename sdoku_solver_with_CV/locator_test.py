import numpy as np
import cv2
from model import Sudoku_locater, suduku_loc_loss, DataGenerator
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# the following codes are needed for my laptop
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

model = tf.keras.models.load_model('/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/weights/test_model.h5')
print(model.summary())
height, width = 160, 90
# read image
img1 = cv2.imread('/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/data/img/1.jpeg', 0)
img2 = cv2.imread('/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/data/img/15.jpeg', 0)
# resize image
img1 = cv2.resize(img1, (height, width))
img2 = cv2.resize(img2, (height, width))
# expand dimensions
img1 = np.expand_dims(img1, axis=2)
img2 = np.expand_dims(img2, axis=2)
img1 = np.expand_dims(img1, axis=0)
img2 = np.expand_dims(img2, axis=0)
# make sure the dimensions
print('img1.shape, img2.shape', img1.shape, img2.shape)

output1 = model.predict(img1)

out1_classifier, out1_coordinate = output1
x1, y1, x2, y2, x3, y3, x4, y4 = out1_coordinate[0]
x1, y1, x2, y2, x3, y3, x4, y4 = width*x1, height*y1, width*x2, height*y2, width*x3, height*y3, width*x4, height*y4
print(out1_classifier, out1_coordinate)
print(x1, y1, x2, y2, x3, y3, x4, y4)

# print(out2_classifier, out2_coordinate)