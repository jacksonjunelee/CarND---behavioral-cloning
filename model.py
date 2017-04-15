from keras.models import Sequential
from keras.layers import Convolution2D, AveragePooling2D, Dense, Flatten, \
						 Lambda, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from preprocess import preprocess, get_preprocess_dict
from sklearn.model_selection import train_test_split
from load import load_data
import pickle
from keras.utgit sils.visualize_util import plot

### load data

X_train, y_train = load_data("./data/driving_log.csv")

### preprocess data

# get preprocess specifications from preprocess.py
preprocess_dict = get_preprocess_dict()

# preprocess training data
X_train = preprocess(X_train, preprocess_dict)

# save preprocess specs to pickle file. This will be read by drive.py later
pickle.dump(preprocess_dict, open("preprocess_dict.p", "wb"))

# split data into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(X_train,
																y_train,
																test_size=0.2)

### build model

model = Sequential()

nb_train_samples, height, width, channels = X_train.shape

# normalize data
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(height,
														width,
														channels)))

# 2D convolution
model.add(Convolution2D(16, 6, 6, border_mode="same"))
model.add(Activation('relu'))

# average pooling
model.add(AveragePooling2D(pool_size=(3, 3)))

# 2D convolution
model.add(Convolution2D(32, 6, 6, border_mode="same"))
model.add(Activation('relu'))

# average pooling
model.add(AveragePooling2D(pool_size=(2, 2)))

# 2D convolution
model.add(Convolution2D(64, 6, 6, border_mode="same"))
model.add(Activation('relu'))

# 2D average pooling
model.add(AveragePooling2D(pool_size=(2, 2)))

# flatten
model.add(Flatten())

# 50% dropout
model.add(Dropout(.5))

# fully connected layer, 1024 features
model.add(Dense(1024))
model.add(Activation('relu'))

# 50% dropout
model.add(Dropout(.5))

# final output, output is between -1 and 1
model.add(Dense(1))
model.add(Activation('tanh'))

# adam optimizer, learning rate of 1e-4
adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss="mse")

# ### train model

nb_validation_samples = len(X_validation)

train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=32)
validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=32)

# train for 5 epochs
model.fit_generator(train_generator, samples_per_epoch=nb_train_samples,
					nb_epoch=5, validation_data=validation_generator,
					nb_val_samples=nb_validation_samples)

# model.h5
model.save_weights('model.h5')
