from model import DigitsClassifier
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import albumentations as A
import os

# the following codes are needed for my laptop
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class DataGenerator(tf.keras.utils.Sequence):
    # data: (60000, 28, 28)     label: (60000,)
    def __init__(self, training, labels, batch_size=32, dim=(28,28), n_channels=1,
                 n_classes=10, shuffle=True, augmentation=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.training = training
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augmentation = augmentation

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

    # add line to one of the sides
    def add_line(self, image, p=0.5):
        if np.random.random() < p:
        # roll dice
            dice = np.random.random()
            if dice < 0.25:
                image[0:np.random.randint(1,3), :] = 255.0
            elif dice < 0.5:
                image[-np.random.randint(1,3):, :] = 255.0
            elif dice < 0.75:
                image[:, 0:np.random.randint(1,3)] = 255.0
            else:
                image[:, -np.random.randint(1,3):] = 255.0

    def __data_generation(self, training_temp, label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, img_label in enumerate (zip(training_temp, label_temp)):
            # Store sample
            training_img, training_label = img_label
            # If for training, add AUGMENTATION
            if self.augmentation:
                transform = A.Compose([
                    A.GaussianBlur(p=0.2), # gaussian blur
                    A.RandomBrightnessContrast(p=0.5), # brightness contrast change
                    A.GaussNoise(p=0.5), # inject gaussian noise
                    A.GridDistortion(p=0.2), # grid distortion
                    A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=5, p=0.5),
                    ])
                training_img = transform(image=training_img)['image']
                self.add_line(training_img, p=0.5)
            training_img = training_img.astype("float32") / 255.0
            training_img = np.expand_dims(training_img, axis=2)
            X[i,] = training_img
            # Store class
            y[i] = training_label
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

# Build the model
digits_classifier = DigitsClassifier(width=28, height=28, depth=1, classes=10)
digits_classifier = digits_classifier.build()
digits_classifier.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
#### load data
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
full_path = os.path.realpath(__file__)
save_dir = os.path.dirname(full_path) + '/training_img/'
x_train, y_train, x_test, y_test = np.load(save_dir + 'X_train.npy'), np.load(save_dir + 'y_train.npy'), np.load(save_dir + 'X_test.npy'), np.load(save_dir + 'y_test.npy')
print('SHAPE ARE: ', x_train.shape, x_test.shape)
'''
# for training
for i in range(y_train.shape[0]):
    print('For training, finished', i+1, 'in', y_train.shape[0])
    if y_train[i] == 0:
        x_train[i] = np.zeros_like(x_train[i])
        for j in range(np.random.randint(0,10)):
            x_train[i, np.random.randint(0,28), np.random.randint(0,28)] = 255.0 * np.random.random()
        transform = A.Compose([
                    A.GaussNoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    ])
        x_train[i] = transform(image= x_train[i])['image']
# for testing
for i in range(y_test.shape[0]):
    print('For testing, finished', i+1, 'in', y_test.shape[0])
    if y_test[i] == 0:
        x_test[i] = np.zeros_like(x_test[i])
        for j in range(np.random.randint(0,10)):
            x_test[i, np.random.randint(0,28), np.random.randint(0,28)] = 255.0 * np.random.random()
        transform = A.Compose([
                    A.GaussNoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    ])
        x_test[i] = transform(image= x_test[i])['image']'''

# data generator for training and validation
training_gen = DataGenerator(x_train, y_train, batch_size=128, dim=(28,28), n_channels=1, n_classes=10, shuffle=True, augmentation=False)
validation_gen = DataGenerator(x_test, y_test, batch_size=128, dim=(28,28), n_channels=1, n_classes=10, shuffle=True, augmentation=False)

def training(Epochs=15):
    with tf.device('/device:gpu:0'):
        digits_classifier.fit(training_gen,
                steps_per_epoch=len(training_gen),
                epochs=Epochs,
                verbose=2,
                validation_data = validation_gen)
    full_path = os.path.realpath(__file__)
    save_dir = os.path.dirname(full_path) + '/weights/'
    digits_classifier.save(save_dir + 'digits_classifier.h5')

if __name__ == '__main__':
    training()