import os

import keras.backend as K
from config import (IMAGE_COLORMODE, LOSS, MASK_COLORMODE, MASK_USECOLORS,
                    SAMPLE_SIZE, CLASS_WEIGHTS)
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


def tversky_loss(y_true, y_pred):
    """コピペコード：
    https://github.com/advaitsave/
    Multiclass-Semantic-Segmentation-CamVid/
    blob/master/
    Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
    """
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0*g0, (0, 1, 2, 3))
    den = (num
           + alpha*K.sum(p0*g1, (0, 1, 2, 3))
           + beta*K.sum(p1*g0, (0, 1, 2, 3)))

    T = K.sum(num/den)

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


def weighted_binary_crossentropy(y_true, y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred),
                             axis=[0, 1, 2])
    return K.sum(class_loglosses * K.constant(CLASS_WEIGHTS))


def load_unet(weights=None):
    if IMAGE_COLORMODE == 'L':
        input_size = SAMPLE_SIZE + (1,)
    elif IMAGE_COLORMODE == 'RGB':
        input_size = SAMPLE_SIZE + (3,)

    if MASK_COLORMODE == 'L':
        model = unet(weights=weights, input_size=input_size, output_size=1)
        model.compile(optimizer=Adam(lr=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    elif MASK_COLORMODE == 'RGB':
        model = unet(weights=weights, input_size=input_size,
                     output_size=len(MASK_USECOLORS))
        if LOSS == 'categorical_crossentropy':
            model.compile(optimizer=Adam(lr=1e-4),
                          loss="categorical_crossentropy",
                          metrics=['accuracy'])
        elif LOSS == 'tversky':
            model.compile(optimizer=Adam(lr=1e-4),
                          loss=tversky_loss,
                          metrics=[tversky_loss, 'accuracy'])
        elif LOSS == 'binary_crossentropy':
            model.compile(optimizer=Adam(lr=1e-4),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        elif LOSS == 'weighted_binary_crossentropy':
            model.compile(optimizer=Adam(lr=1e-4),
                          loss=weighted_binary_crossentropy,
                          metrics=['accuracy'])
        else:
            raise Exception("Unexpected loss function")

    else:
        raise Exception("Invalid MASK_COLORMODE")

    return model


def unet(weights, input_size, output_size):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal'
                 )(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal'
                 )(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal'
                 )(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal'
                 )(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(output_size, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    if weights:
        model.load_weights(weights)

    return model
