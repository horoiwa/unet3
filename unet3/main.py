import os
import warnings

warnings.filterwarnings('ignore')

import click
from generator import test_generator
from preparation import prep_dataset
from train import run_train
from util import initilize


@click.group()
def main():
    pass


@main.command()
@click.option('--dataset', '-d', required=True,
              help='Dataset folder path',
              type=click.Path(exists=True))
@click.option('--outdir', '-o', default='__unet__', type=str)
def check(dataset, outdir):
    dataset = os.path.abspath(dataset)
    initilize(outdir)
    prep_dataset(dataset, outdir)
    print("Generator check")
    test_generator(dataset, outdir)


@main.command()
@click.option('--dataset', '-d', required=True,
              help='Dataset folder path',
              type=click.Path(exists=True))
@click.option('--outdir', '-o', default='__unet__', type=str)
def train(dataset, outdir):
    initilize(outdir, remove=False)
    run_train(dataset, outdir)


@main.command()
@click.option('--dataset', '-d', required=True,
              help='Dataset folder path',
              type=click.Path(exists=True))
@click.option('--outdir', '-o', default='__unet__', type=str)
def generator(dataset, outdir):
    initilize(outdir)
    test_generator(dataset, outdir)


if __name__ == '__main__':
    main()
