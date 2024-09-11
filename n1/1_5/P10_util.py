import os
import shutil
import sys

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



def mkdir(d, rm=False):
    if rm:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    else:
        try: os.makedirs(d)
        except FileExistsError: pass

def make_generator(src_dir, valid_rate, input_size, batch_size):

    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        validation_split = valid_rate
    )


    train_generator = train_datagen.flow_from_directory(
        src_dir,
        target_size = input_size,
        batch_size = batch_size,
        shuffle = True,
        class_mode = 'categorical',
        subset = 'training'

    )


    valid_generator = train_datagen.flow_from_directory(
        src_dir,
        target_size = input_size,
        batch_size = batch_size,
        shuffle = True,
        class_mode = 'categorical',
        subset = 'training'


    )


    train_ds = Dataset.from_generator(
        lambda: train_generator,
        output_types = (tf.float32, tf.float32),
        output_shapes = (
            [None, *valid_generator.image_shape]
            [None, *valid_generator.num_classes]
        )
    )

    train_ds = train_ds.repeat()
    valid_ds = valid_ds.repeat()

    return train_ds, train_generator, valid_ds, valid_generator.n
    

def plot(hisotry, filename):

    def add_subplot(nrows, ncols, index, xdata, train_ydata, valid_ydata, ylim, ylabel):
        plt.subplot(nrows, ncols, index)
        plt.plot(xdata, train_ydata, label='training', linestyle='--')
        plt.plot(xdata, valid_ydata, label='validataion')
        plt.xlim(1, len(xdata))
        plt.ylim(*ylim)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')


    plt.figure(figsize=(10,10))
    xdata = range(1, 1 + len(hisotry['loss']))
    add_subplot(2, 1, 1, xdata, hisotry['loss'], hisotry['val_loss'], (0, 5), 'loss')

    add_subplot(2, 1, 2, xdata, hisotry['accuracy'], hisotry['val_accuracy'], (0, 1), 'accuracy')
    plt.savefig(filename)
    plt.close('all')
    


    