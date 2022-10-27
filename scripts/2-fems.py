import os
import glob
import argparse
from movneye.motion import fem
from movneye.functional.read import read_json
from movneye.functional.store import store_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate and save FEM sequences according to the adapted Self-Avoiding random Walk (SAW) model.')

    parser.add_argument('--fem_dir', type=str, default='./data/fem',
                        help='Directory where the output .npy data files of the generated FEM should be saved.\n'
                             'Note: 3 files are stored for each FEM, with names:\n'
                             '      - [#seed]_steps.npy\n'
                             '      - [#seed]_activation-field.npy\n'
                             '      - [#seed]_is-ms.npy\n'
                             'Also, a fem_info.json file, where all SAW parameters are stored, will be saved once\n'
                             'for all the FEM seeds in the given directory (since these parameters should be the same\n'
                             'for all of them).')
    parser.add_argument('--n_burnin', type=int, default=16000,
                        help='Number of iterations (steps) considered for the burn-in.')
    parser.add_argument('--n_traj', type=int, default=80,
                        help='Number of steps of the FEM sequence.')
    parser.add_argument('--foveola', type=float, default=120,
                        help='Biological diameter (in minutes of arc) of the foveola.')
    parser.add_argument('--max_step', type=float, default=6,
                        help='Maximum size (in minutes of arc) for a single step of the FEM sequence.')
    parser.add_argument('--ptu_resolution', type=float, default=46.2857/60,
                        help='Pan/Tilt resolution (in minutes of arc) of the PTU device.')
    parser.add_argument('--grid', type=int, default=301,
                        help='Size of the squared grid where the FEM is defined.')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                        help='Relaxation factor of the SAW model.')
    parser.add_argument('--lamda', type=float, default=0.005,
                        help='Weight of the quadratic potential U of the SAW model.')
    parser.add_argument('--chi', type=float, default=None,
                        help='Weight of the Micro-Saccadic potential U1 of the SAW model.')
    parser.add_argument('--hc', type=float, default=0.5,  # set to 0.38 for adding 1 or 2 MS in the FEM
                        help='Activation threshold for Micro-Saccade generation.')
    parser.add_argument('--max_seed', type=int, default=99,
                        help='The maximum seed that can be run. No more than max_seed FEM sequences can be stored in\n'
                             'the given directory. Default is 99.')
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='Whether to print out some information on each FEM sequence.')
    parser.add_argument('--ask_feedback', action="store_true", default=False,
                        help='Whether to ask the user a feedback after each FEM sequence has been generated on whether'
                             '\nto store the current FEM seed.')

    return parser.parse_args()


def main(fem_dir, max_seed, ask_feedback, verbose,
         kwargs):

    fem_generator = fem.SAW(**kwargs, init_burnin=False)

    os.makedirs(fem_dir, exist_ok=True)
    all_existing = sorted(glob.glob(os.path.join(fem_dir, '*_steps.npy')))
    seed = int(os.path.basename(all_existing[-1]).split('_')[0]) + 1 if all_existing else 0

    fem_info = os.path.join(fem_dir, 'fem_info.json')
    if not os.path.isfile(fem_info) or seed == 0:
        store_json(fem_info, kwargs)
    else:
        old_kwargs = read_json(fem_info)
        if kwargs != old_kwargs:
            print(f'\nThe parameters of the SAW model changed since last time you saved FEM sequences in {fem_dir}.\n'
                  f'The following parameters were set [parameter name:  OLD value  -->  NEW value]:')
            for key in old_kwargs.keys():
                print(f'{key}:  {old_kwargs[key]}  -->  {kwargs[key]}')
            enter = input(
                'Type "y"/"yes" if you want to compute and store the new FEM sequences with the given parameters.\n'
                'Note that in this case the json file where all FEM information are stored will be overwritten.\n'
                'If you type "n"/"no" the new parameters will be ignored and the old ones will be used. ')
            if enter.lower() in ['y', 'yes']:
                store_json(fem_info, kwargs)
            else:
                fem_generator = fem.SAW(**old_kwargs)

    try:
        while True:
            # Generate FEM sequence and print some info
            fem_generator.burn_in()
            fem_generator.generate_fem(verbose=verbose)

            if ask_feedback:
                # Visualise some plots
                fem_generator.show_fem(unit='arcmin', view_foveola=True, show=False)
                fem_generator.angles_distribution(plot=True, show=False)
                fem_generator.steps_sizes_distribution(unit='arcmin', plot=True, show=False)
                fem_generator.show_fem_positions(unit='arcmin', show=True)

                # Store the FEM if you wish
                file2save = os.path.join(fem_dir, str(seed) + '_steps.npy')
                warning_message = '' if not os.path.isfile(file2save) else \
                    ' [CAUTION: the file to save already exists and will be over-written!]'
                enter = input(f'\n{seed}) Enter y if you want to save this FEM seed{warning_message}: ')
                if enter.lower() in ['y', 'yes']:
                    fem_generator.save_fem(dir=fem_dir, seed=str(seed).zfill(len(str(max_seed))))
                    seed += 1
                    if seed > max_seed:
                        break
            else:
                fem_generator.save_fem(dir=fem_dir, seed=str(seed).zfill(len(str(max_seed))))
                seed += 1
                if seed > max_seed:
                    break

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":

    args = parse_args()

    saw_args = dict(n_burnin=args.n_burnin,                 # number of iterations (steps) considered for the burn-in
                    n_traj=args.n_traj,                     # number of steps actually considered for the FEM
                    grid=args.grid,                         # grid size
                    epsilon=args.epsilon,                   # relaxation factor
                    lamda=args.lamda,                       # weight of the quadratic potential U
                    chi=args.chi,                           # weight of the MS potential U1
                    hc=args.hc,                             # activation threshold for MS generation
                    foveola=args.foveola,                   # (arcmin) biological diameter of the foveola
                    max_step=args.max_step,                 # (arcmin) maximum step size
                    ptu_resolution=args.ptu_resolution)     # (arcmin) resolution of PTU-E46-17

    main(args.fem_dir, args.max_seed, args.ask_feedback, args.verbose,
         kwargs=saw_args)
