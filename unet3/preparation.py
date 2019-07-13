import glob
import os

import numpy as np
from PIL import Image, ImageChops

from config import (FRAME_SIZE, IMAGE_COLORMODE, MASK_COLORMODE, SAMPLE_SIZE,
                    TARGET_SIZE)
from generator import test_generator


def prep_dataset(dataset_path, outdir):
    print()
    print("Start dataset checking:")
    print("------"*5)
    print("Directory check")
    check_datasetdirs(dataset_path)
    print("Names check")
    check_imagenames(dataset_path)
    print("Size check")
    check_imagesize(dataset_path)
    print("Frame size check")
    check_framesize()
    print("Format check")
    check_format(dataset_path)
    print("Mask check")
    check_mask(dataset_path)
    print("------"*5)
    print("All checks finished gracefully")
    print()


def check_datasetdirs(dataset_path):
    dirs = os.listdir(dataset_path)
    if 'train' not in dirs:
        raise Exception(f"train dir not found in {dataset_path}")
    elif 'valid' not in dirs:
        raise Exception(f"valid dir not found in {dataset_path}")

    for dir_ in ['train', 'valid']:
        if 'image' not in os.listdir(os.path.join(dataset_path, dir_)):
            raise Exception(f"train dir not found in {dir_}")
        elif 'mask' not in os.listdir(os.path.join(dataset_path, dir_)):
            raise Exception(f"valid dir not found in {dataset_path}")


def check_framesize():
    assert TARGET_SIZE[0] % SAMPLE_SIZE[0] == 0,  "Size error 1"
    assert TARGET_SIZE[1] % SAMPLE_SIZE[1] == 0,  "Size error 2"

    assert TARGET_SIZE[0] % FRAME_SIZE == 0,  "Size error 3"
    assert TARGET_SIZE[1] % FRAME_SIZE == 0,  "Size error 4"


def check_imagenames(dataset_path):
    for dir_ in ['train', 'valid']:
        path_images = glob.glob(os.path.join(dataset_path, dir_,
                                             'image', '*.jpg'))
        names_image = [os.path.basename(path) for path in path_images]

        path_masks = glob.glob(os.path.join(dataset_path, dir_,
                                            'mask', '*.jpg'))
        names_mask = [os.path.basename(path) for path in path_masks]

        if names_image != names_mask:
            raise Exception("Image name consitence error")
        elif not names_image:
            raise Exception("No images found")
        elif not names_mask:
            raise Exception("No masks found")


def check_imagesize(dataset_path):
    for dir_ in ['train', 'valid']:
        path_images = glob.glob(os.path.join(dataset_path, dir_,
                                             'image', '*.jpg'))

        path_masks = glob.glob(os.path.join(dataset_path, dir_,
                                            'mask', '*.jpg'))

        for image_path, mask_path in zip(path_images, path_masks):
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            if image.size != mask.size:
                print(image_path, mask_path)
                raise Exception("Image size inconsitent")


def check_format(dataset_path):

    for dir_ in ['train', 'valid']:
        images = glob.glob(os.path.join(dataset_path, dir_, 'image', '*.jpg'))
        for image_path in images:
            image = Image.open(image_path)
            if image.mode != IMAGE_COLORMODE:
                print(image.mode)
                raise Exception("Image mode inconsitence")
    for dir_ in ['train', 'valid']:
        images = glob.glob(os.path.join(dataset_path, dir_, 'mask', '*.jpg'))
        for image_path in images:
            image = Image.open(image_path)
            if image.mode != MASK_COLORMODE:
                print(image.mode)
                raise Exception("Image mode inconsitence")


def check_mask(dataset_path):
    for dir_ in ['train', 'valid']:
        masks = glob.glob(os.path.join(dataset_path, dir_, 'mask', '*.jpg'))
        for mask_path in masks:
            mask = Image.open(mask_path)
            if mask.mode == 'L':
                greyscale_check(mask, mask_path)
            elif mask.mode == 'RGB':
                rgb_check(mask, mask_path)


def greyscale_check(mask, mask_path):
    mask = np.array(mask).flatten()
    mask[mask[:] > 240] = 0
    if mask.sum() > 0:
        print(f"Warning: Not binary image {mask_path}")


def rgb_check(mask, mask_path):
    """
        RGBの3クラス分類なので変な色が混じってるとダメ
        e.g. (234, 230, 0)はエラーになるべき
    """
    r, g, b = mask.split()
    _r = r.point(lambda x: 1 if x > 230 else 0, mode="1")
    _g = g.point(lambda x: 1 if x > 230 else 0, mode="1")
    _b = b.point(lambda x: 1 if x > 230 else 0, mode="1")

    rg = np.array(ImageChops.logical_and(_r, _g))
    rb = np.array(ImageChops.logical_and(_r, _b))
    gb = np.array(ImageChops.logical_and(_g, _b))

    if np.any(rg) or np.any(rb) or np.any(gb):
        print(mask_path)
        raise Exception("Invalid image")
