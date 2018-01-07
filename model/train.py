# coding: utf8

"""
Description :
Generate the train\val data and training the CNN model.
"""

import os
import sys
import argparse
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot

from cnn import cnn_net


def main(argv):
    parser = argparse.ArgumentParser()

    # Optional arguments.
    parser.add_argument(
        "--epochs",
        default=200,
        help="The number of train iterations",
    )
    parser.add_argument(
        "--batch_size_train",
        default=100,
        help="number of train samples per batch",
    )
    parser.add_argument(
        "--batch_size_val",
        default=40,
        help="number of validation samples per batch",
    )
    parser.add_argument(
        "--images_dim",
        default=64,
        help="'height, width' dimensions of input images.",
    )
    args = parser.parse_args()

    model = cnn_net(args.images_dim)
    print("Plotting the model")
    plot(model, to_file='model.png')
    train(model, args.epochs, args.batch_size_train, args.batch_size_val, args.images_dim)


def data_process(size, batch_size_train, batch_size_val):
    path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        path + '//data//train',
        target_size=(size, size),
        batch_size=batch_size_train,
        class_mode='binary')

    validation_generator = datagen2.flow_from_directory(
        path + '//data//validation',
        target_size=(size, size),
        batch_size=batch_size_val,
        class_mode='binary')

    return train_generator, validation_generator


def train(model, epochs, batch_size_train, batch_size_val, size):
    train_generator, validation_generator = data_process(size, batch_size_train, batch_size_val)
    # Using the early stopping technique to prevent overfitting.
    earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')

    hist = model.fit_generator(
        train_generator,
        nb_epoch=epochs,
        samples_per_epoch=batch_size_train,
        validation_data=validation_generator,
        nb_val_samples=batch_size_val,
        callbacks=[earlyStopping])

    # Saving the loss and acc during the traning time.
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('hist.csv', encoding='utf-8', index=False)
    model.save('weights.h5')


if __name__ == '__main__':
    main(sys.argv)
