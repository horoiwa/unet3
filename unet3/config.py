#: RGB or GREY
COLORMODE = 'RGB'

TARGET_SIZE = (512, 512)

BATCH_SIZE = 6

INITIAL_EPOCHS = 30
SECOND_EPOCHS = 150
EA_EPOCHS = 10

#: 基本的にはこの設定値なら影響がない
PCA_COLOR_RANGE = (-0.3, 0.3)

DATA_GEN_DEFAULT = dict(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.0,
    zoom_range=0.1,
    vertical_flip=False,
    horizontal_flip=True,
    cval=0,
    fill_mode='constant')
