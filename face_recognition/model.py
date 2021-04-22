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
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, concatenate, Concatenate, InputLayer, Dropout, Reshape, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape


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

    def __init__(self, training_path, S=7, B=1, batch_size=32, dim=(48*3, 64*3), n_channels=3,
                 shuffle=True, augmentation=False):
        self.dim = dim
        self.H, self.W = self.dim[0], self.dim[1]
        self.batch_size = batch_size
        self.S = S
        self.B = B
        self.training_path = training_path
        self.sub_dirs = os.listdir(training_path)
        le = LabelEncoder()
        le.fit(self.sub_dirs)
        # self.label_name = le.classes_
        self.n_classes = len(self.sub_dirs)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.img_path = []
        self.label_path = []
        for self.sub_dir in self.sub_dirs:
            for f in (os.listdir(os.path.join(self.training_path, self.sub_dir))):
                if f.endswith('jpg'): # iterate over all images
                    # img_path contains all images
                    self.img_path.append(os.path.join(self.training_path, self.sub_dir, f))
                    # generate OneHotLabel for this sub folder
                    self.label_encoder = le.transform(self.sub_dir.split(" "))
                    self.OneHotLabel = tf.keras.utils.to_categorical(self.label_encoder[0], num_classes=self.n_classes)
                    # check if the json file exists, yes: clss_id, xywhpc, xywhpc; (no: class_id, 00000) makes no sense...
                    self.json_path = os.path.join(self.training_path,self.sub_dir, f[:-4] + '.json')
                    self.label_path.append([self.json_path, self.OneHotLabel])
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
            #  read the label, and make the first entry 1 (means there is face in the image)
            x, y, w, h = DataGenerator.json_to_bbox(self.label_path[i][0])
            # from website
            loc = [self.S * x, self.S * y]
            loc_i = int(loc[1])
            loc_j = int(loc[0])
            y = loc[1] - loc_i
            x = loc[0] - loc_j
            # label is like : class_id, (x,y,w,h,conf), (x,y,w,h,conf), (x,y,w,h,conf)...
            self.labels[i, loc_i, loc_j, :self.n_classes] = self.label_path[i][1]
            self.labels[i, loc_i, loc_j, self.n_classes: self.n_classes+4] = [x, y, w, h]
            self.labels[i, loc_i, loc_j, self.n_classes+4] = 1                
        self.augmentation = augmentation    
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
        if self.augmentation == True:
            pass
        

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


# Using pretrained ResNet50 as feather extraction
class Face_yolo_Resnet:
    def __init__(self, B=1, C=1, S=7, img_w=192, img_h=144):
        self.B = B # bbox number
        self.S = S # grid number
        self.C = C # class number
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
        output = Reshape((self.S, self.S,(self.C+self.B*5)))(output)
        #output = Yolo_Reshape(S=self.S, B=self.B, C=self.C)(output)
        model = Model(inputs=resnet.inputs, outputs=output)
        
        model.summary()
        return model


# image classifier with pretrained MobileNet (mobilenet is quick and small)
'''class Face_classifier_mobilenet:
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
        return model'''


'''class DataGenerator_classsifier(tf.keras.utils.Sequence):

    def __init__(self, training_path, batch_size=32, H=144, W=192, n_channels=3, shuffle=True):
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
        return X, y'''