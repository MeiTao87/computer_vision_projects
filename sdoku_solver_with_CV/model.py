import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, concatenate, Concatenate
import numpy as np

class DigitsClassifier:
	def __init__(self, width, height, depth, classes):
		self.width = width
		self.height = height
		self.depth = depth
		self.classes = classes


	def build(self):
		# initialize the model
		model = tf.keras.Sequential()
		inputShape = (self.height, self.width, self.depth)
        
		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(32, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
		# first set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		
		# second set of FC => RELU layers
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		
		# softmax classifier
		model.add(Dense(self.classes))
		model.add(Activation("softmax"))
		# return the constructed network architecture
		return model

# https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
class Sudoku_locater:
	# def __init__(self, input_size = (160,90,1)):
	# 	self.input_size = input_size
	
	@staticmethod
	def classifier_branch_make(input):
		conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		conv2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		flatten1 = Flatten()(pool2)
		dense1 = Dense(64, activation='relu')(flatten1)
		dense2 = Dense(64, activation='relu')(dense1)
		classifier_branch = Dense(2, activation='softmax', name='classifier_branch')(dense2)
		return classifier_branch

	@staticmethod
	def coordinate_branch_make(input):
		conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		conv2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		flatten1 = Flatten()(pool2)
		dense1 = Dense(64, activation='relu')(flatten1)
		dense2 = Dense(64, activation='relu')(dense1)
		coordinate_branch = Dense(8, activation='softmax', name='coordinate_branch')(dense2)
		return coordinate_branch

	@staticmethod
	def build(input_size):
		inputs = tf.keras.Input(input_size, name='input_build')
		classifier_branch = Sudoku_locater.classifier_branch_make(inputs)
		coordinate_branch = Sudoku_locater.coordinate_branch_make(inputs)
		
		model = tf.keras.Model(
			inputs=inputs,
			outputs=[classifier_branch, coordinate_branch],
			name="sudoku_model")
		# return the constructed network architecture
		print(model.summary())
		return model
		
		
		# inputs = tf.keras.Input(self.input_size)
		# conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		# pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		# conv2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
		# pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		# flatten1 = Flatten()(pool2)
		# dense1 = Dense(64, activation='relu')(flatten1)
		# dense2 = Dense(64, activation='relu')(dense1)
		# dense_classifier = Dense(2, activation='sigmoid')(dense2)
		# dense_coordinates = Dense(8, activation='relu')(dense2)
		# merge = tf.keras.layers.concatenate([dense_classifier, dense_coordinates], axis=-1)
		# model = tf.keras.Model(inputs, merge)
		# model.summary()
		# return model

def suduku_loc_loss(y_true, y_pred, factor=1):
	y_true_class, y_true_coor = y_true[:,:2], y_true[:,2:]
	print('type(y_true_class)', type(y_true_class))
	print('y_true_class', y_true_class)
	y_true_class = tf.cast(y_true_class, tf.int32)
	
	proto_tensor = tf.make_tensor_proto(y_true_class)  # convert `tensor a` to a proto tensor
	y_true_class = tf.make_ndarray(proto_tensor)
	
	y_pred_class, y_pred_coor = y_pred[:,:2], y_pred[:,2:]
	y_true_class = tf.keras.utils.to_categorical(y_true_class)
	loss_class = tf.keras.losses.categorical_crossentropy(y_true_class, y_pred_class)
	# x1, y1 , x2, y2, x3, y3, x4, y4
	loss_coor = tf.keras.losses.MSE(y_true_coor, y_pred_coor)
	
	return loss_class + factor * loss_coor

class DataGenerator(tf.keras.utils.Sequence):
    # data: (60000, 28, 28)     label: (60000,)
    def __init__(self, training, labels, batch_size=32, dim=(160,90), n_channels=1,
                 n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.training = training
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
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
        X, y_classifier_branch, y_coordinate_branch = self.__data_generation(training_temp, label_temp)
        return X, {"classifier_branch": y_classifier_branch, "coordinate_branch": y_coordinate_branch}

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.training))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, training_temp, label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y_classifier_branch = np.empty((self.batch_size, 2), dtype=np.int8)
        y_coordinate_branch = np.empty((self.batch_size, 8), dtype=np.float32)

        # Generate data
        for i, img_and_label in enumerate (zip(training_temp, label_temp)):
            # Store sample
            training_img, training_label = img_and_label
            training_img = training_img / 255.0
            training_img = np.expand_dims(training_img, axis=2)
            X[i,] = training_img
            # Store class
            y_classifier_branch[i] = training_label[:,:2]
            y_coordinate_branch[i] = training_label[:, 2:]
        return X, y_classifier_branch, y_coordinate_branch
