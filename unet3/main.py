import os
import click

from preparation import prep_dataset
from generator import test_generator
from train import run_train
from inference import run_inference
from util import initilize


@click.command()
@click.option('--mode', '-m', required=True,
              help="Choose from generator, prep, train, test",
              type=click.Choice(['generator', 'check', 'train', 'test']))
@click.option('--dataset', '-d', required=True,
              type=click.Path(exists=True))
@click.option('--outdir', '-o', default='__unet__', type=str)
def main(mode, dataset, outdir):
    dataset = os.path.abspath(dataset)

    if mode == 'check':
        initilize(outdir)
        prep_dataset(dataset, outdir)
        print("Generator check")
        test_generator(dataset, outdir)
    elif mode == 'train':
        initilize(outdir, remove=False)
        run_train(outdir)
    elif mode == 'test':
        run_inference(dataset, outdir)
    elif mode == 'generator':
        initilize(outdir)
        test_generator(dataset, outdir)
    else:
        raise Exception("Unexpected error")


if __name__ == '__main__':
    main()
