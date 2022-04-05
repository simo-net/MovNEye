import os
import argparse
import matplotlib.pyplot as plt
from visionart.sensor import dvs
from visionart.functional.wrap import read_bin
from visionart.opening import load_img_dataset


# Usage:
# >>    python3 6-load-bin.py  -f ./data/mnist/test/2/00000.bin  --bin_size 10


def parse_args():
    parser = argparse.ArgumentParser(description='Visualizing a file of recorded and pre-processed DAVIS events.')

    # Recording file
    parser.add_argument('-f', '--file2load', type=str, required=True,
                        help='Full path of the input .json file of recording info to load.\nNote: the names '
                             'for all the other files saved (APS, DVS and IMU events, as well as FEM and IMG info) '
                             'are taken from here.')
    parser.add_argument('--bin_size', type=float, default=20,
                        help='The time length (in ms) of each frame for visualizing the DVS events.')

    return parser.parse_args()


def load_bin_recording(file2load: str,
                       bin_size: float = 1,
                       verbose: bool = True, plot: bool = True):
    if verbose:
        print('\nLoading DVS recording...')
    events = read_bin(file2load)

    # Pre-process DVS events
    with dvs.handler(data=events, reset_timestamps=False, shape=(34, 34)) as dvs_handler:
        rec_duration = dvs_handler.duration
        if verbose:
            print(f'   - Duration is: ~{int(round(rec_duration * 1e-3))} ms\n'
                  f'   - Number of DVS events: {dvs_handler.num_events}\n'
                  f'   - Mean firing rate is: {round(dvs_handler.mean_firing_rate(), 2)} Hz\n'
                  f'   - Fraction of ON/OFF events is: {tuple([round(f, 2) for f in dvs_handler.fraction_onoff()])}')

        if plot:
            dvs_handler.show_rasterplot(show=True)

            # dvs_handler.show_video(bin_size=bin_size*1e3)
            dvs_handler.show_video_onoff_bluewhite(bin_size=bin_size*1e3)
            dvs_handler.show_surface_active_events(bin_size=bin_size*1e3)
            dvs_handler.show_time_difference(bin_size=bin_size*1e3)

            # dvs_handler.store_video(file='./video.avi', bin_size=bin_size*1e3)
            # dvs_handler.store_surface_active_events(file='./video_sae.avi', bin_size=bin_size*1e3)
            # dvs_handler.store_time_difference(file='./video_td.avi', bin_size=bin_size*1e3)

            dvs_handler.show_view3D_onoff(duration=500 * 1e3)  # JUST SHOW THE FIRST 200ms (or plot will be too heavy)

            ifr_on, ifr_off = dvs_handler.video_onoff(bin_size=200 * 1e3)
            img_dvs = (ifr_on[0] - ifr_off[0])

        if verbose:
            print('\n')

    # View the corresponding stimulus (dataset sample)
    _, _, dataset_type, img_split, img_label, img_id = os.path.splitext(file2load)[0].split('/')
    img_label, img_id = int(img_label), int(img_id)
    img = load_img_dataset(dataset_type, split=img_split, index=img_id, verbose=verbose, plot=False)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    plt.suptitle(f'{dataset_type.upper()} sample {img_id} from the {img_split.upper()} set - label {img_label}')
    axs[0].set_title(f'DVS recording {img_dvs.shape}')
    axs[0].imshow(img_dvs, cmap='gray')
    axs[0].axis('off')
    axs[1].set_title(f'Original image {img.shape[:2]}\nstatic display')
    axs[1].imshow(img[..., 0], cmap='gray')
    axs[1].axis('off')
    plt.show()


if __name__ == "__main__":

    args = parse_args()

    load_bin_recording(file2load=args.file2load, bin_size=args.bin_size,
                       verbose=True, plot=True)
