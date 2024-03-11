import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
import tensorflow

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tensorflow.compat.v1.InteractiveSession(config=config)

DATADIR = "D:/Moje dokumenty/Studia/Semestr 6/WMA/LAB04/Images"
CATEGORIES = ["Banana", "Orange", "Lemon"]
IMG_SIZE = 100
imgset = ImageDataGenerator(validation_split=0.1)
dataset = imgset.flow_from_directory(directory=DATADIR, shuffle=True, color_mode="grayscale",
                                     target_size=(IMG_SIZE, IMG_SIZE), batch_size=8, seed=111, subset="training",
                                     class_mode='categorical')
valset = imgset.flow_from_directory(directory=DATADIR, shuffle=True, color_mode="grayscale",
                                    target_size=(IMG_SIZE, IMG_SIZE), batch_size=8, seed=111, subset="validation",
                                    class_mode='categorical')

dense_layers = [2]
layer_sizes = [128]
conv_layers = [1]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            print(NAME)
            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), padding='same', input_shape=(100, 100, 1)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(3))
            model.add(Activation('softmax'))

            model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
            model.fit(dataset, batch_size=16, epochs=20, validation_data=valset, callbacks=[tensorboard])

            model.save("models/" + NAME + '.model')
