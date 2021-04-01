import os
import cv2
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, concatenate, Concatenate, InputLayer, Dropout, Reshape, GlobalMaxPooling2D, GlobalAveragePooling2D
# from tensorflow import keras


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

        # first set of CONV => CONV => RELU => POOL layers
        model.add(Conv2D(64, (3, 3), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        # model.add(Conv2D(32, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # second set of CONV => CONV => RELU => POOL layers
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
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
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(self.classes))
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
            H, W = data['imageHeight'], data['imageWidth']
            x1, y1 = points[0]
            x2, y2 = points[1]
            w, h = x2-x1, y2-y1
            centerx, centery = x1 + 0.5*w, y1 + 0.5*h
            x, y, w, h = centerx/W, centery/H, w/W, h/H
        return x, y, w, h

    def __init__(self, training_path, S=7, B=2, batch_size=32, dim=(48*3, 64*3), n_channels=3,
                 n_classes=1, shuffle=True):
        self.dim = dim
        self.H, self.W = self.dim[0], self.dim[1]
        self.batch_size = batch_size
        self.S = S
        self.B = B
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
        ############### rebuild the labels structure
        self.training = np.empty((len(self.img_path), self.H, self.W, self.n_channels))
        self.labels = np.zeros((len(self.img_path), self.S, self.S, self.B*5+self.n_classes)) # (length, S, S, (5B+C)) -> (length, 7, 7, 11)
        for i in range(len(self.img_path)):
            img = cv2.imread(self.img_path[i], 0)
            # resize to (H, W)
            img = cv2.resize(img, (self.W, self.H))
            # convert gray scale to RGB
            img  = np.stack((img,)*3, axis=-1)
            self.training[i] = img
            # if label exists, read the label, and make the first entry 1 (means there is face in the image)
            if os.path.isfile(self.label_path[i]):
                x, y, w, h = DataGenerator.json_to_bbox(self.label_path[i])
                # from website
                loc = [self.S * x, self.S * y]
                loc_i = int(loc[1])
                loc_j = int(loc[0])
                y = loc[1] - loc_i
                x = loc[0] - loc_j
                # label is like : class_id, (x,y,w,h,conf), (x,y,w,h,conf), (x,y,w,h,conf)...
                self.labels[i, loc_i, loc_j, 0] = 1
                self.labels[i, loc_i, loc_j, self.n_classes: self.n_classes+4] = [x, y, w, h]
                self.labels[i, loc_i, loc_j, self.n_classes+4] = 1                
            
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
        # expand dimensions
        # y = np.expand_dims(y, axis=1)
        # y = np.expand_dims(y, axis=2)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.training))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, training_temp, label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.H, self.W, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.S, self.S, self.B*5+self.n_classes), dtype=np.float32)

        # Generate data
        for i, img_and_label in enumerate (zip(training_temp, label_temp)):
            # Store sample
            training_img, training_label = img_and_label
            training_img = training_img / 255.0
            # training_img = np.expand_dims(training_img, axis=2)
            X[i,] = training_img
            # Store class
            y[i] = training_label
        return X, y


class Face_classify:
    def __init__(self, width, height, depth, classes):
        # width and height have to be 2**n
        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes

    def build(self):
        model = tf.keras.Sequential()
        inputShape = (self.height, self.width, self.depth)

        # first set of CONV => CONV => RELU => POOL layers
        model.add(Conv2D(64, (3, 3), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # second set of CONV => CONV => RELU => POOL layers
        model.add(Conv2D(128, (3, 3), padding="same"))
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
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(self.classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        print(model.summary())
        return model


# https://www.maskaravivek.com/post/yolov1/
# inspried by github repo: https://github.com/JY-112553/yolov1-keras-voc
class Yolo_Reshape(tf.keras.layers.Layer):
    def __init__(self, S=7, B=2, C=1):
        super(Yolo_Reshape, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.target_shape = tuple((self.S, self.S, self.B*5+self.C))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape
        })
        return config

    def call(self, input):
        idx1 = self.S * self.S * self.C
        idx2 = idx1 + self.S * self.S * self.B
        
        # class probabilities
        class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([self.S, self.S, self.C]))
        class_probs = K.softmax(class_probs)

        #confidence
        confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([self.S, self.S, self.B]))
        confs = K.sigmoid(confs)

        # boxes
        boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([self.S, self.S, self.B * 4]))
        boxes = K.sigmoid(boxes)

        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs    


class Face_yolo:
    def __init__(self, B=2, C=1, S=7, img_w=192, img_h=144):
        self.B = B
        self.S = S
        # self.cell_w = cell_w
        # self.cell_h = cell_h
        self.C = C
        self.img_w = img_w
        self.img_h = img_h
   
    def build(self):
        lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size= (7, 7), strides=(1, 1), input_shape =(self.img_h, self.img_w, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

        model.add(Conv2D(filters=64, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

        model.add(Conv2D(filters=64, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        model.add(Conv2D(filters=64, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        model.add(Conv2D(filters=64, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

        model.add(Conv2D(filters=64, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        model.add(Conv2D(filters=64, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

        model.add(Conv2D(filters=64, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        # model.add(Conv2D(filters=1024, kernel_size= (3, 3), strides=(2, 2), padding = 'same'))

        # model.add(Conv2D(filters=1024, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))
        model.add(Conv2D(filters=128, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(128))
        model.add(Dropout(0.5))
        model.add(Dense(self.S * self.S * (self.B * 5 + self.C), activation='sigmoid'))
        model.add(Yolo_Reshape(S=self.S, B=self.B, C=self.C))
        model.summary()
        return model


# Using pretrained ResNet50 as feather extraction
class Face_yolo_Resnet:
    def __init__(self, B=2, C=1, S=7, img_w=192, img_h=144):
        self.B = B
        self.S = S
        self.C = C
        self.img_w = img_w
        self.img_h = img_h
   
    def build(self):
        lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
        
        resnet = ResNet50(include_top=False, input_shape=(self.img_h, self.img_w, 3))
        for layer in resnet.layers:
            layer.trainable = False
        
        flat = Flatten()(resnet.layers[-1].output)
        dense_layer = Dense(256, activation='relu')(flat)
        output = Dense(self.S * self.S * (self.B * 5 + self.C), activation='sigmoid')(dense_layer) 
        output = Yolo_Reshape(S=self.S, B=self.B, C=self.C)(output)
        model = Model(inputs=resnet.inputs, outputs=output)
        
        model.summary()
        return model


# image classifier with pretrained MobileNet (mobilenet is quick and small)
class Face_classifier_mobilenet:
    def __init__(self, n_classes, input_shape):
        self.n_classes = n_classes
        self.input_shape = input_shape

    def build(self):
        base_model = MobileNetV2(input_shape=self.input_shape, weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(self.n_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        model.summary()
        return model


class DataGenerator_classsifier(tf.keras.utils.Sequence):

    def __init__(self, training_path, batch_size=32, H=144, W=192, n_classes=3, n_channels=3, shuffle=True):
        self.H = H
        self.W = W
        self.n_channels = n_channels
        self.img_path = []
        self.label_path = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training_path = training_path
        self.sub_dirs = os.listdir(training_path)
        le = LabelEncoder()
        le.fit(self.sub_dirs)
        self.n_classes = len(self.sub_dirs)
        for self.sub_dir in self.sub_dirs:
            for f in (os.listdir(os.path.join(self.training_path, self.sub_dir))):
                if f.endswith('jpg'):
                    self.img_path.append(os.path.join(self.training_path, self.sub_dir, f))
                    # self.label_path.append(le.transform(self.sub_dir.split(" ")))
                    self.label_encoder = le.transform(self.sub_dir.split(" "))
                    self.OneHotLabel = tf.keras.utils.to_categorical(self.label_encoder, num_classes=self.n_classes)
                    self.label_path.append(self.OneHotLabel)
                    # print(self.OneHotLabel)
        self.training = np.empty((len(self.img_path), self.H, self.W, self.n_channels))
        self.labels = np.zeros((len(self.img_path), self.n_classes)) 
        for i in range(len(self.img_path)):
            img = cv2.imread(self.img_path[i], 0)
            # resize to (H, W)
            img = cv2.resize(img, (self.W, self.H))
            # convert gray scale to RGB
            img  = np.stack((img,)*3, axis=-1)
            self.training[i] = img
            self.labels[i] = self.label_path[i]
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
        # expand dimensions
        # y = np.expand_dims(y, axis=1)
        # y = np.expand_dims(y, axis=2)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.training))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, training_temp, label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.H, self.W, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.n_classes), dtype=np.int16)

        # Generate data
        for i, img_and_label in enumerate (zip(training_temp, label_temp)):
            # Store sample
            training_img, training_label = img_and_label
            training_img = training_img / 255.0
            # training_img = np.expand_dims(training_img, axis=2)
            X[i,] = training_img
            # Store class
            y[i] = training_label
        return X, y