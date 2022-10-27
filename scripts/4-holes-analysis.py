import argparse
import numpy as np
import matplotlib.pyplot as plt
from movneye.functional.diagnose import detect_holes_distribution
from movneye.utils.add2os import keep_list_extension, listdir_flatten


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize the histogram of holes in the recordings.')

    # Error-specific parameters
    parser.add_argument('-i', '--rec_dir', type=str, required=True,
                        help='Directory where the recordings to check are stored.')
    parser.add_argument('--max_hole_duration', type=float, default=4,
                        help='The maximum duration (in ms) of a hole in the recordings for defining a bad file.\n'
                             'Default is 4 ms.')
    parser.add_argument('--max_num_holes', type=int, default=2,
                        help='The maximum number of holes (with duration greater than "max_hole_duration") in the\n'
                             'recordings for defining a bad file. Default is 2.')
    parser.add_argument('--rec_burnin', type=float, default=150,
                        help='Time period (in ms) to cut off at the beginning of the recording for detecting holes.\n'
                             'Default is 100ms.')
    parser.add_argument('--rec_burnout', type=float, default=150,
                        help='Time period (in ms) to cut off at the end of the recording for detecting holes.\n'
                             'Default is 100ms.')
    parser.add_argument('--store_results', action="store_true", default=False,
                        help='Whether to store both file paths and hole duration of all the bad recordings detected.\n'
                             'Default is False.')

    return parser.parse_args()


def main(rec_dir,
         max_num_holes, max_hole_duration,
         rec_burnin, rec_burnout,
         store_results):

    files = sorted(keep_list_extension(listdir_flatten(rec_dir), '.json', empty_error=False))
    holes_hist, holes_bins = detect_holes_distribution(files,
                                                       num_bins=300, max_hole_duration=300,
                                                       rec_burnin=rec_burnin, rec_burnout=rec_burnout,
                                                       show_pbar=True)

    if store_results:
        np.save(f'./holes_hist.npy', holes_hist)
        np.save(f'./holes_bins.npy', holes_bins)
    # holes_hist = np.load('./holes_hist.npy')
    # holes_bins = np.load('./holes_bins.npy')

    plt.figure()
    plt.errorbar(holes_bins, np.mean(holes_hist, axis=0), yerr=np.std(holes_hist, axis=0), ecolor='r')
    plt.show()

    # if not max_hole_duration:
    #     max_hole_duration = holes_bins[np.argmax(np.median(holes_hist, axis=0))]+0.5

    problematic_files, holes_in_files = [], []
    recs_ids = np.where(holes_hist > 0)[0]
    unique_recs_ids, num_holes = np.unique(recs_ids, return_counts=True)
    for i in unique_recs_ids[num_holes > 1]:
        bins2check = holes_bins[np.where(holes_hist[i] >= max_num_holes)[0]]
        condition = (bins2check >= max_hole_duration)
        if any(condition):
            problematic_files.append(files[i])
            holes_in_files.append(bins2check[condition])
    print(f'\nThere are {len(problematic_files)} bad recordings (i.e. having at least {max_num_holes} hole(s) lasting '
          f'for at least {max_hole_duration} ms).')
    x = input('\nDo you want to save them? [y/n]\n')
    if x.lower() in ['y', 'yes']:
        np.save('./bad_recordings.npy', np.array(problematic_files, dtype=str))


if __name__ == "__main__":

    args = parse_args()

    main(rec_dir=args.rec_dir,
         max_hole_duration=args.max_hole_duration, max_num_holes=args.max_num_holes,
         rec_burnin=args.rec_burnin, rec_burnout=args.rec_burnout,
         store_results=args.store_results)
