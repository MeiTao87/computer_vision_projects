import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, concatenate, Concatenate
import os
import cv2
import json


class Face_loc:
    def __init__(self, width, height, depth, classes):
        # width and height have to be 2**n
        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes

    def build(self):
        model = tf.keras.Sequential()
        inputShape = (self.height, self.width, self.depth)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first set of FC => RELU layers
        # model.add(Flatten())
        # model.add(Dense(64))
        model.add(Conv2D(64, (1, 1), padding="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # second set of FC => RELU layers
        # model.add(Dense(64))
        model.add(Conv2D(64, (1, 1), padding="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # softmax classifier
        # model.add(Dense(self.classes))
        model.add(Conv2D(self.classes, (int(self.height/4), int(self.width/4))))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        print(model.summary())
        return model


class DataGenerator(tf.keras.utils.Sequence):
    
    @staticmethod
    def json_to_bbox(json_file_path):
        with open (json_file_path) as f:
            data = json.load(f)
            points = data['shapes'][0]['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            w, h = x2-x1, y2-y1
            centerx, centery = [x1 + 0.5*w, y1 + 0.5*h]
            centerx, centery, w, h = int(centerx), int(centery), int(w), int(h)
        return centerx, centery, w, h

    def __init__(self, training_path, batch_size=32, dim=(64*3,48*3), n_channels=1,
                 n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.training_path = training_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.img_path = []
        self.label_path = []
        for subdir, dirs, files in os.walk(self.training_path):
            for f in files:
                if(os.path.join(subdir, f)).endswith(".jpg"):
                    label = f[:-4] + '.json'
                    self.img_path.append(os.path.join(subdir, f)) 
                    self.label_path.append(os.path.join(subdir, label))
        # using assert?
        self.training = np.empty((len(self.img_path), self.dim[1], self.dim[0]))
        self.labels = np.empty((len(self.img_path), 4))
        for i in range(len(self.img_path)):
            img = cv2.imread(self.img_path[i], 0)
            # img = cv2.resize(img, self.dim)
            x, y, w, h = DataGenerator.json_to_bbox(self.label_path[i])
            print('x, y, w, h', x, y, w, h)
            x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w),  int(y+0.5*h)
            ################
            print('x1, y1, x2, y2', x1, y1, x2, y2)
            self.training[i] = img
            self.labels[i] = x1, y1, x2, y2
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.training) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of training data
        training_temp = [self.training[k,:,:] for k in indexes]
        label_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(training_temp, label_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.training))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, training_temp, label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[1], self.dim[0], self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, 4), dtype=np.int16)

        # Generate data
        for i, img_and_label in enumerate (zip(training_temp, label_temp)):
            # Store sample
            training_img, training_label = img_and_label
            training_img = training_img / 255.0
            training_img = np.expand_dims(training_img, axis=2)
            X[i,] = training_img
            # Store class
            y[i] = training_label
        return X, y

