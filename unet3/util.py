import os
import shutil


def initilize(outdir, remove=True):
    if remove:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
