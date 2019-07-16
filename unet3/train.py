import os
import shutil
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE, DATA_GEN_ARGS, IMAGE_COLORMODE, MASK_COLORMODE, SAMPLE_SIZE,
    TARGET_SIZE, EA_EPOCHS, TRAIN_STEPS, VALID_STEPS, EPOCHS)
from generator import ImageMaskGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from models import load_unet
from util import get_weights


def run_train(dataset_dir, outdir):
    print('Start training')
    print('-----'*5)

    weights_dir = os.path.join(outdir, '__checkpoints__')
    hdfname = os.path.join(weights_dir, 'model_1.hdf5')
    if os.path.exists(weights_dir):
        shutil.rmtree(weights_dir)
    os.makedirs(weights_dir)

    trainGen = ImageMaskGenerator(batch_size=BATCH_SIZE,
                                  dataset_dir=dataset_dir,
                                  folder='train',
                                  aug_dict=DATA_GEN_ARGS,
                                  image_colormode=IMAGE_COLORMODE,
                                  mask_colormode=MASK_COLORMODE,
                                  target_size=TARGET_SIZE,
                                  sample_size=SAMPLE_SIZE,
                                  shuffle=True)

    validGen = ImageMaskGenerator(batch_size=BATCH_SIZE,
                                  dataset_dir=dataset_dir,
                                  folder='valid',
                                  aug_dict=None,
                                  image_colormode=IMAGE_COLORMODE,
                                  mask_colormode=MASK_COLORMODE,
                                  target_size=TARGET_SIZE,
                                  sample_size=SAMPLE_SIZE,
                                  shuffle=True)

    model_checkpoint = ModelCheckpoint(hdfname,
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=EA_EPOCHS, verbose=0,
                                   mode='auto')

    callbacks = [
        early_stopping,
        model_checkpoint,
        ReduceLROnPlateau(factor=0.1, patience=EA_EPOCHS//2,
                          min_lr=0.00001, verbose=1),
    ]
    acc_train = []
    acc_val = []
    model = load_unet(weights=None)
    print(model.summary())

    history = model.fit_generator(
        trainGen,
        steps_per_epoch=TRAIN_STEPS,
        epochs=EPOCHS,
        validation_data=validGen,
        validation_steps=VALID_STEPS,
        callbacks=callbacks)

    acc_train = acc_train + list(history.history['acc'])
    acc_val = acc_val + list(history.history['val_acc'])
    epochs = range(1, len(acc_train) + 1)

    plt.plot(epochs, acc_train, label='train')
    plt.plot(epochs, acc_val, label='valid')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'training_history.png'))

    print("train acc:", acc_train)
    print("valid acc:", acc_val)


