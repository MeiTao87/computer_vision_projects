import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# the following codes are needed for my laptop
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# load the model
full_path = os.path.realpath(__file__)
save_dir = os.path.dirname(full_path) + '/weights/'
digits_classifier = tf.keras.models.load_model(save_dir + 'digits_classifier.h5')

# prepare data
img = cv2.imread('/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/test_img/sudoku_example.jpg', 0)
img = img[74:547, 26:499]
img = cv2.resize(img, (28*9, 28*9))
H, W = img.shape
h, w = int(1/9*H), int(1/9*W)
for row in range(9):
    for col in range(9):
        sub_img_origin = img[(row*w):(row*w+w), (col*h):(col*h+h)]
        sub_img = sub_img_origin / 255.0
        sub_img = np.expand_dims(sub_img, axis=2)
        sub_img = np.expand_dims(sub_img, axis=0)
        pre = digits_classifier.predict(sub_img)
        plt.title(np.argmax(pre[0]))
        plt.imshow(sub_img_origin, cmap='gray')
        plt.show()