import os
import shutil


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
