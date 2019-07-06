import os
import sys

import click


@click.command()
@click.option('--mode', '-m', required=True,
              help="Choose from generator, prep, train, test",
              type=click.Choice(['generater', 'prep', 'train', 'test']))
@click.option('--dataset', '-d', required=True,
              type=click.Path(exists=True))
def main(mode, dataset):
    dataset = os.path.abspath(dataset)
    if mode == 'prep':
        prep_data(dataset)
    elif mode == 'generator':
        test_generator(dataset)
    elif mode == 'train':
        train(dataset)
    elif mode == 'test':
        test(dataset)


if __name__ == '__main__':
    main()
