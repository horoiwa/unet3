import os
import shutil

import numpy as np
from PIL import Image

from config import (
    BACKGROUND_COLOR, BATCH_SIZE, BG_COLOR, DATA_GEN_ARGS, IMAGE_COLORMODE,
    MASK_COLORMODE, MASK_USECOLORS, PCA_COLOR, PCA_COLOR_RANGE, SAMPLE_SIZE,
    TARGET_SIZE)
from keras.preprocessing.image import ImageDataGenerator
from util import get_rgbmask


def test_generator(dataset_dir, outdir):
    gen_test_dir = os.path.join(outdir, 'GenConfigTest')
    image_dir = os.path.join(gen_test_dir, 'image')
    mask_dir = os.path.join(gen_test_dir, 'mask')

    if os.path.exists(gen_test_dir):
        shutil.rmtree(gen_test_dir)

    os.makedirs(gen_test_dir)
    os.makedirs(image_dir)
    os.makedirs(mask_dir)

    customGen = ImageMaskGenerator(batch_size=40,
                                   dataset_dir=dataset_dir,
                                   folder='train',
                                   aug_dict=DATA_GEN_ARGS,
                                   image_colormode=IMAGE_COLORMODE,
                                   mask_colormode=MASK_COLORMODE,
                                   target_size=TARGET_SIZE,
                                   sample_size=SAMPLE_SIZE,
                                   shuffle=True)

    images, masks = customGen.__next__()
    for n, (image, mask) in enumerate(zip(images, masks)):
        if IMAGE_COLORMODE == 'RGB':
            image = Image.fromarray(np.uint8(image*255))
        elif IMAGE_COLORMODE == 'L':
            image = image.reshape(SAMPLE_SIZE)*255
            image = Image.fromarray(np.uint8(image))
            image = image.convert('RGB')
        image.save(os.path.join(image_dir, f'{n}.jpg'))

        if MASK_COLORMODE == 'RGB':
            mask_new = get_rgbmask(mask)
            mask = Image.fromarray(np.uint8(mask_new*255))

        elif MASK_COLORMODE == 'L':
            mask = mask.reshape(SAMPLE_SIZE)*255
            mask = Image.fromarray(np.uint8(mask))
            mask = mask.convert('RGB')

        mask.save(os.path.join(mask_dir, f'{n}.jpg'))


def ImageMaskGenerator(batch_size, dataset_dir, folder, aug_dict,
                       image_colormode, mask_colormode,
                       target_size, sample_size, shuffle):
    """ image: 入力がLでも返すのはRGB
        mask: Nチャネルndarray
    """
    seed = np.random.randint(999)

    if image_colormode == 'L':
        image_colormode = 'grayscale'
    elif image_colormode == 'RGB':
        image_colormode = 'rgb'
    else:
        raise Exception('Invalid image colormode')

    if mask_colormode == 'L':
        mask_colormode = 'grayscale'
    elif mask_colormode == 'RGB':
        mask_colormode = 'rgb'
    else:
        raise Exception('Invalid mask colormode')

    if not aug_dict:
        aug_dict = dict(rescale=1./255)

    train_path = os.path.join(dataset_dir, folder)

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    imageGen = image_datagen.flow_from_directory(
        train_path,
        classes=['image'],
        class_mode=None,
        color_mode=image_colormode,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed)

    maskGen = mask_datagen.flow_from_directory(
        train_path,
        classes=['mask'],
        class_mode=None,
        color_mode=mask_colormode,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed)

    imagemaskGen = zip(imageGen, maskGen)
    for images, masks in imagemaskGen:
        #: 任意の処理を挟むことが可能
        #: 1./255にリスケされてることに注意
        n_images = images.shape[0]
        images_new = np.zeros((n_images,) + SAMPLE_SIZE + (images.shape[3],))

        if MASK_COLORMODE == 'L':
            masks_new = np.zeros((n_images,) + SAMPLE_SIZE + (1,))
        elif MASK_COLORMODE == 'RGB':
            masks_new = np.zeros(
                (n_images,) + SAMPLE_SIZE + (len(MASK_USECOLORS),))

        for i in range(n_images):
            image = images[i, :, :, :]
            mask = masks[i, :, :, :]

            image, mask = resampling(image, mask)
            if IMAGE_COLORMODE == 'RGB':
                image = pca_color_augmentation_rgb(image*255)
            elif IMAGE_COLORMODE == 'L':
                #: 入力がLならいったんRGB変換する
                image = image.reshape(SAMPLE_SIZE)
                image = Image.fromarray(np.uint8(image*255))
                if PCA_COLOR:
                    image = image.convert('RGB')
                    image = pca_color_augmentation_rgb(np.array(image))
                    image = Image.fromarray(image).convert('L')
                image = np.array(image).reshape(SAMPLE_SIZE + (1,))

            image = image / 255
            images_new[i, :, :, :] = image

            mask = adjustmask(mask)
            masks_new[i, :, :, :] = mask

        yield (images_new, masks_new)


def adjustmask(mask):
    if MASK_COLORMODE == 'L':
        assert mask.shape[2] == 1
        mask[mask > 0.5] = 1
        mask[mask < 0.5] = 0
        return mask

    elif MASK_COLORMODE == 'RGB':
        assert mask.shape[2] == 3
        r = mask[:, :, 0]
        r[r > 0.5] = 1
        r[r < 0.5] = 0
        g = mask[:, :, 1]
        g[g > 0.5] = 1
        g[g < 0.5] = 0
        b = mask[:, :, 2]
        b[b > 0.5] = 1
        b[b < 0.5] = 0

        mask_new = np.zeros(SAMPLE_SIZE + (len(MASK_USECOLORS),))

        i = 0
        if 'R' in MASK_USECOLORS:
            mask_new[:, :, i] = r
            i += 1
        if 'G' in MASK_USECOLORS:
            mask_new[:, :, i] = g
            i += 1
        if 'B' in MASK_USECOLORS:
            mask_new[:, :, i] = b
            i += 1

        """背景を1クラスとして扱う
        """
        if BG_COLOR:
            mask_new = np.apply_along_axis(
                lambda x: x if np.any(x) else BACKGROUND_COLOR, 2, mask_new)
        return mask_new

    else:
        raise Exception("Unexpected Error #234")


def resampling(image, mask):
    h = image.shape[0]
    w = image.shape[1]

    upperleft = (np.random.randint(0, h-SAMPLE_SIZE[0]+1), np.random.randint(0, w-SAMPLE_SIZE[1]+1))

    image = image[upperleft[0]:upperleft[0]+SAMPLE_SIZE[0],
                  upperleft[1]:upperleft[1]+SAMPLE_SIZE[1], :]
    mask = mask[upperleft[0]:upperleft[0]+SAMPLE_SIZE[0],
                upperleft[1]:upperleft[1]+SAMPLE_SIZE[1], :]

    return image, mask


def pca_color_augmentation_rgb(image_array_input):
    """
        RGBカラー画像限定
        コピぺ：https://qiita.com/koshian2/items/78de8ccd09dd2998ddfc
    """

    img = image_array_input.reshape(-1, 3).astype(np.float32)
    # 分散を計算
    ch_var = np.var(img, axis=0)
    # 分散の合計が3になるようにスケーリング
    scaling_factor = np.sqrt(3.0 / sum(ch_var))
    # 平均で引いてスケーリング
    img = (img - np.mean(img, axis=0)) * scaling_factor

    cov = np.cov(img, rowvar=False)
    lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)

    while True:
        rand = np.random.randn(3) * 0.1
        if np.all(rand > PCA_COLOR_RANGE[0]):
            if np.all(rand < PCA_COLOR_RANGE[1]):
                break

    delta = np.dot(p_eigen_vector, rand*lambd_eigen_value)
    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
    return img_out
