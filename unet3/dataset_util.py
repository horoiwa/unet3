import click
from PIL import Image
import cv2

import os
import glob


@click.command()
@click.option('--folder', '-f', required=True,
              help='image folder path',
              type=click.Path(exists=True))
@click.option('--mode', '-m', required=True,
              type=click.Choice(['contour', 'molph']))
def main(folder, mode):
    folder = os.path.abspath(folder)
    images = glob.glob(os.path.join(folder, '*.jpg'))
    print(images)




if __name__ == '__main__':
    main()
