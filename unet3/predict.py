import glob
import os
import shutil
import copy
import pathlib
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import numpy as np
import click

from config import (IMAGE_COLORMODE, MASK_COLORMODE, MASK_USECOLORS,
                    SAMPLE_SIZE, TARGET_SIZE, FRAME_SIZE)
from models import load_unet
from util import get_rgbmask


@click.command()
@click.option('--folder', '-f', required=True,
              help='Data folder path',
              type=click.Path(exists=True))
@click.option('--outdir', '-o', default='__unet__',
              help='output folder path',
              type=click.Path(exists=True))
@click.option('--padding', '-p', is_flag=True,
              help='padding predict results')
@click.option('--separate', '-s', is_flag=True,
              help='Output image separated by RGB chanel')
@click.option('--debug', is_flag=True,
              help='Debug mode')
def main(folder, outdir, padding, separate, debug):
    folder_path = os.path.abspath(folder)
    outdir = os.path.abspath(outdir)
    predict(folder_path, outdir, padding, separate, debug)


def predict(folder_path, outdir, padding, separate, debug):
    weights_dir = os.path.join(outdir, '__checkpoints__')
    hdfname = os.path.join(weights_dir, 'model_1.hdf5')

    results_dir = os.path.join(outdir, 'results')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    images = glob.glob(os.path.join(folder_path, '*.jpg'))
    model = load_unet(weights=hdfname)

    for image_path in images:
        image = load_image(image_path)
        imagename = pathlib.Path(image_path)
        print(f"Process: {imagename.name}")
        mask_ndarray = inference(image, model, padding)
        mask_RGB = postprocess(mask_ndarray)

        mask_pil = Image.fromarray(np.uint8(mask_RGB))
        mask_pil.save(os.path.join(results_dir, f'{imagename.stem}.jpg'))

        if separate:
            r = Image.fromarray(np.uint8(mask_RGB[:, :, 0])).convert('L')
            r.save(os.path.join(results_dir, f'{imagename.stem}_r.jpg'))
            g = Image.fromarray(np.uint8(mask_RGB[:, :, 1])).convert('L')
            g.save(os.path.join(results_dir, f'{imagename.stem}_g.jpg'))
            b = Image.fromarray(np.uint8(mask_RGB[:, :, 2])).convert('L')
            b.save(os.path.join(results_dir, f'{imagename.stem}_b.jpg'))

        if debug:
            break


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    image = np.array(image)
    return image


def postprocess(mask_ndarray):
    if MASK_COLORMODE == 'L':
        assert mask_ndarray.shape[-1] == 1, 'Invalid format'
        mask_ndarray = mask_ndarray * 255
        mask_ndarray = mask_ndarray.reshape(TARGET_SIZE)
    elif MASK_COLORMODE == 'RGB':
        mask_ndarray = get_rgbmask(mask_ndarray)
        mask_ndarray = mask_ndarray * 255

    return mask_ndarray


def inference(image, model, padding=True):
    """
        Return:
            ndarray: size=TARGET_SIZE, 0~1
    """
    frame = FRAME_SIZE
    if MASK_COLORMODE == 'L':
        image_results = np.zeros((image.shape[0], image.shape[1], 1))
    elif MASK_COLORMODE == 'RGB':
        image_results = np.zeros(
            (image.shape[0], image.shape[1], len(MASK_USECOLORS)))
    else:
        raise Exception('Unexpected maskusecolors')

    height = SAMPLE_SIZE[0]
    width = SAMPLE_SIZE[1]

    x_lim = TARGET_SIZE[0]
    y_lim = TARGET_SIZE[1]

    x_list = list(range(0, x_lim, frame))
    y_list = list(range(0, y_lim, frame))

    for i in range(len(x_list)):
        for j in range(len(y_list)):
            x = x_list[i]
            y = y_list[j]
            x_s = x if x == 0 else x-frame
            y_s = y if y == 0 else y-frame
            x_e = x_s + width
            y_e = y_s + height

            image_temp = copy.deepcopy(image[y_s:y_e, x_s:x_e])

            if (x_e > x_lim) or (y_e > y_lim):
                continue

            if IMAGE_COLORMODE == 'L':
                image_temp = image_temp.reshape(SAMPLE_SIZE+(1,))
                image_temp = image_temp.reshape((1,)+SAMPLE_SIZE+(1,))
            elif IMAGE_COLORMODE == 'RGB':
                image_temp = image_temp.reshape(SAMPLE_SIZE+(3,))
                image_temp = image_temp.reshape((1,)+SAMPLE_SIZE+(3,))

            pred = model.predict(image_temp)
            if MASK_COLORMODE == 'L':
                pred = pred.reshape(SAMPLE_SIZE+(1,))
            elif MASK_COLORMODE == 'RGB':
                pred = pred.reshape(SAMPLE_SIZE+(pred.shape[-1],))
            else:
                raise Exception('Unexpected Error')

            if padding:
                if x == 0 and y == 0:
                    image_results[0:frame, 0:frame, :] = pred[0:frame,
                                                              0:frame, :]
                elif x == 0 and y_e == y_lim:
                    image_results[-frame:, 0:frame, :] = pred[-frame:,
                                                              0:frame, :]
                elif y == 0 and x_e == x_lim:
                    image_results[0:frame, -frame:, :] = pred[0:frame,
                                                              -frame:, :]
                elif x_e == x_lim and y_e == y_lim:
                    image_results[-frame:, -frame:, :] = pred[-frame:,
                                                              -frame:, :]

                if x == 0:
                    image_results[y_s+frame:y_e-frame, x_s:x_s+frame, :] = pred[frame:-frame, 0:0+frame, :]
                elif x_e == x_lim:
                    image_results[y_s+frame:y_e-frame, x_e-frame:x_e, :] = pred[frame:-frame, -frame:, :]

                if y == 0:
                    image_results[y_s:y_s+frame, x_s+frame:x_e-frame, :] = pred[0:frame, frame:-frame, :]
                elif y_e == y_lim:
                    image_results[y_e-frame:y_e, x_s+frame:x_e-frame, :] = pred[-frame:, frame:-frame, :]

            image_results[y_s+frame:y_e-frame, x_s+frame:x_e-frame, :] = pred[frame:-frame, frame:-frame, :]

    return image_results


if __name__ == '__main__':
    main()
