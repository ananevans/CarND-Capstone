# Mohamed Elbanhawi (mohamedbanhawi@gmail.com)
import cv2
import numpy as np
import json
import data_augmentation
import time
import glob

start_time = time.time()

DA = data_augmentation.DataAugmentation()

# classes
UNKNOWN=4
GREEN=2
YELLOW=1
RED=0

with open('parameters.json') as f:
    parameters = json.load(f)

lines = []
images = []
traffic_state = []

datasets = parameters['datasets']
data_augmentation = parameters['data_augmentation']
training_parameters = parameters['training']
labels = parameters['labels']

X_train = np.array([])
y_train = np.array([])
images = []
traffic_state = []
# load files
for dataset, state in zip(datasets, labels):
    print ('Loading '+ dataset)
    print ('Class ' + str(state))
    for image_path in glob.glob(dataset+'/*.png'):
        image = cv2.imread(image_path)
        images.append(image)
        traffic_state.append(state)

        if data_augmentation['blur']:
            kernel = data_augmentation['blur_kernel']
            image_blurred = DA.blur_dataset(image, kernel)
            images.append(image)
            traffic_state.append(state)

        if data_augmentation['brighten']:
            gamme = 5
            image_bright = DA.adjust_gamma(image, gamme)
            images.append(image_bright)
            traffic_state.append(state)

        if data_augmentation['darken']:
            gamme = 0.35
            image_dark = DA.adjust_gamma(image, gamme)
            images.append(image_dark)
            traffic_state.append(state)

        if data_augmentation['translate']:
            distance = data_augmentation['distance_px']
            image_pos = DA.shift_dataset(image, distance)
            images.append(image_pos)
            traffic_state.append(state)

            distance = -distance
            image_neg = DA.shift_dataset(image, distance)
            images.append(image_neg)
            traffic_state.append(state)

        if data_augmentation['rotate']:
            angle = data_augmentation['rotation_angle_deg']
            image_rotate_pos = DA.rotate_dataset(image, angle)
            images.append(image_rotate_pos)
            traffic_state.append(state)
            angle = -angle
            image_rotate_neg= DA.rotate_dataset(image, angle)
            images.append(image_rotate_neg)
            traffic_state.append(state)

if data_augmentation['flip']: 
    print ('Flipping all..'+str(len(images)))
    for i in range(len(images)): 
        image_flip = cv2.flip(images[i], 1)
        images.append(image_flip)
        traffic_state.append(traffic_state[i])


if data_augmentation['show_images']: 
    cv2.imshow('Original Image',image)
    if data_augmentation['flip']:
        cv2.imshow('Flipped Image',image_flip)
    if data_augmentation['blur']:
        cv2.imshow('Blurred',image_blurred)
    if data_augmentation['brighten']:
        cv2.imshow('Bright',image_bright)
    if data_augmentation['darken']:
        cv2.imshow('Dark',image_dark)
    if data_augmentation['translate']:
        cv2.imshow('Translate Positive',image_pos)
        cv2.imshow('Translate Negative',image_neg)
    if data_augmentation['rotate']:
        cv2.imshow('Rotate Pos',image_rotate_pos)
        cv2.imshow('Rotate - Neg ',image_rotate_neg)

    key = cv2.waitKey(0)
    if key == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

if data_augmentation['write_images']: 
    cv2.imwrite('Original Image.png',image)
    if data_augmentation['flip']:
        cv2.imwrite('Flipped Image.png',image_flip)
    if data_augmentation['blur']:
        cv2.imwrite('Blurred.png',image_blurred)
    if data_augmentation['brighten']:
        cv2.imwrite('Bright.png',image_bright)
    if data_augmentation['darken']:
        cv2.imwrite('Dark.png',image_dark)
    if data_augmentation['translate']:
        cv2.imwrite('Translate Positive.png',image_pos)
        cv2.imwrite('Translate Negative.png',image_neg)
    if data_augmentation['rotate']:
        cv2.imwrite('Rotate Pos.png',image_rotate_pos)
        cv2.imwrite('Rotate - Neg.png',image_rotate_neg)

X_train = np.array(images)
y_train = np.array(traffic_state)

shape = X_train[-1].shape

print shape

from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.losses import sparse_categorical_crossentropy
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

# y_train = to_categorical(y_train, len(labels)) # one hot encoding

model = Sequential()

if training_parameters['network'] == 'alexnet':
    # implement alexnet
    # 1st Convolutional Layer
    # Initialize model
    # Layer 1
    model.add(Conv2D(96, (11, 11), input_shape=(shape[0], shape[1], shape[2]),
        padding='same', kernel_regularizer=l2(0)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 2
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 3
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 4
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 5
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 6
    model.add(Flatten())
    model.add(Dense(3072))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Layer 7
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Layer 8
    model.add(Dense(len(labels)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

elif training_parameters['network'] == 'lenet':
    model.add(Conv2D(32,  (3, 3), input_shape=shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))

model.summary()

# Compile the model
model.compile(loss=sparse_categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

if training_parameters['load_model']:
    model = load_model(training_parameters['model_name'])

if training_parameters['train']:
    history_object = model.fit(X_train, y_train, validation_split = 0.2, epochs=training_parameters['epochs'], shuffle = True)

    model.save(training_parameters['network']+'_classifier_model.h5')

    if training_parameters['visualise_performance']:
        # summarize history for accuracy
        plt.plot(history_object.history['acc'])
        plt.plot(history_object.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig(training_parameters['network']+'_accuracy.png')
        # summarize history for loss
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(training_parameters['network']+'_loss.png')