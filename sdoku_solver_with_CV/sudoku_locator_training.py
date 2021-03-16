import numpy as np
import cv2
from model import Sudoku_locater, suduku_loc_loss, DataGenerator
import tensorflow as tf
import os
import matplotlib.pyplot as plt



t = np.load("/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/data/training.npz")
training_data = t["img"]
# training_data = np.expand_dims(training_data, axis=3)
l = np.load("/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/data/label.npz")
training_label = l["label"]
label_classifier_branch, label_coordinate_branch = training_label[:,0,:2], training_label[:,0,2:]


height, width = 160, 90
dim = (height, width)
training_gen = DataGenerator(training_data, training_label, dim=dim, batch_size=8, shuffle=True)
validation_gen = DataGenerator(training_data[:1000], training_label[:1000], batch_size=8, shuffle=True)

# the following codes are needed for my laptop
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# training and predicting
test_model = Sudoku_locater()
test_model = test_model.build(input_size = (width, height, 1))

# test_model.compile(loss=suduku_loc_loss, optimizer='adam', metrics=["accuracy"])
# Model compile
losses = {
	"classifier_branch": "categorical_crossentropy",
	"coordinate_branch": "MSE"}
lossWeights = {"classifier_branch": 1.0, "coordinate_branch": 1.0}
test_model.compile(loss=losses, loss_weights=lossWeights, optimizer='adam', metrics=["accuracy"])

# with tf.device('/device:gpu:0'):
#     test_model.fit(x=training_data,
#                     y= {'classifier_branch': label_classifier_branch, 
#                     'coordinate_branch': label_coordinate_branch},
#                     epochs=10,
#                     verbose=2)

with tf.device('/device:gpu:0'):
    test_model.fit(training_gen,
                    steps_per_epoch=len(training_gen),
                    epochs=20,
                    verbose=1,
                    validation_data = validation_gen)

test_model.save('/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/weights/test_model.h5')
