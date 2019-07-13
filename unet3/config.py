""" Training config
    COLORMODE = Enum(['L', 'RGB'])
"""

IMAGE_COLORMODE = 'L'
MASK_COLORMODE = 'RGB'
MASK_USECOLORS = 'RB'

MODEL = 'unet'

TARGET_SIZE = (512, 512)
SAMPLE_SIZE = (256, 256)
FRAME_SIZE = 32

BATCH_SIZE = 2

TRAIN_STEPS = 500
VALID_STEPS = 100
EPOCHS = 5

EA_EPOCHS = 5

#: 基本的にはこの設定値なら影響がない
PCA_COLOR_RANGE = (-0.2, 0.2)

DATA_GEN_ARGS = dict(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.0,
    zoom_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    cval=0,
    fill_mode='constant')

"""
基本はグレスケ入力-L/RGB出力を想定
未検証：RGB入力

UpSampling2D使用は出力の格子模様を防ぐ
"""
