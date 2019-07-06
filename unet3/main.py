import os
import click

from preparation import prep_dataset
from generator import test_generator
from train import run_train
from inference import run_inference


@click.command()
@click.option('--mode', '-m', required=True,
              help="Choose from generator, prep, train, test",
              type=click.Choice(['generater', 'prep', 'train', 'test']))
@click.option('--dataset', '-d', required=True,
              type=click.Path(exists=True))
@click.option('--outdir', '-o', default='__unet__', type=str)
def main(mode, dataset, outdir):
    dataset = os.path.abspath(dataset)

    if mode == 'prep':
        prep_dataset(dataset, outdir)
    elif mode == 'generator':
        test_generator(dataset, outdir)
    elif mode == 'train':
        run_train(outdir)
    elif mode == 'test':
        run_inference(dataset, outdir)


if __name__ == '__main__':
    main()
