#: RGB or L
IMAGE_COLORMODE = 'L'
MASK_COLORMODE = 'RGB'
#MASK_COLORMODE = 'L'

MODEL = 'unet'
TARGET_SIZE = (1024, 1024)
SAMPLE_SIZE = (256, 256)

BATCH_SIZE = 2

INITIAL_EPOCHS = 30
SECOND_EPOCHS = 150
EA_EPOCHS = 10

#: 基本的にはこの設定値なら影響がない
PCA_COLOR_RANGE = (-0.3, 0.3)

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
未検証：RGB入力 グレスケ出力
"""
