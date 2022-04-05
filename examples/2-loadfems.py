import os
import glob
import argparse
from visionart.opening import load_fem
from visionart.functional.read import read_json


# Usage:
# >>    python3 2-loadfems.py  --fem_dir ./data/fem  --verbose


def parse_args():
    parser = argparse.ArgumentParser(description='Loading all FEM sequences in the given directory one-by-one.')

    parser.add_argument('--fem_dir', type=str, default='./data/fem',
                        help='Directory where the .npy FEM sequences are stored. Note that the files where FEM steps\n'
                             'are stored as Nx2 numpy arrays must have names with the structure: [#seed]_steps.npy')
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='Whether to print out some information on each FEM sequence.')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    saw_args = read_json(os.path.join(args.fem_dir, 'fem_info.json'))
    print(f'\nThe parameters of the SAW model used for all the FEMs stored in the given directory are:')
    for key in saw_args.keys():
        print(f'{key}:  {saw_args[key]}')

    all_files = sorted(glob.glob(os.path.join(args.fem_dir, '*_steps.npy')))
    for fem_file in all_files:
        seed = int(os.path.basename(fem_file).split('_')[0])
        print(f'\n--------- {seed} ---------')
        load_fem(fem_file, verbose=args.verbose)
