import os
import shutil

import numpy as np
from config import MASK_USECOLORS


def initilize(outdir, remove=True):
    if remove:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)


def get_weights(weights_dir, n=1):
    name = 'model_'
    currentname = name + str(n) + '.hdf5'
    nextname = name + str(n+1) + '.hdf5'

    if os.path.exists(os.path.join(weights_dir, nextname)):
        get_weights(name, n+1)
    elif os.path.exists(os.path.join(weights_dir, currentname)):
        return os.path.join(weights_dir, currentname)
    else:
        return None


def get_rgbmask(mask):
    """
        Input
        ----------
        mask : np.ndarray shape=(N, M, len(MAKS_USECOLORS))
               各要素の値は0-1
        Return
        ----------
        mask_new : np.ndarray shape=(N, M, 3)
                   各要素の値は0-1
    """
    mask_new = np.zeros((mask.shape[:2] + (3,)))

    i = 0
    if 'R' in MASK_USECOLORS:
        mask_new[:, :, 0] = mask[:, :, i]
        i += 1
    else:
        mask_new[:, :, 0] = 0

    if 'G' in MASK_USECOLORS:
        mask_new[:, :, 1] = mask[:, :, i]
        i += 1
    else:
        mask_new[:, :, 1] = 0

    if 'B' in MASK_USECOLORS:
        mask_new[:, :, 2] = mask[:, :, i]
        i += 1
    else:
        mask_new[:, :, 2] = 0

    return mask_new
