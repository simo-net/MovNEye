import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from movneye.calib.paths import CALIB2D_CAM_FILE
from movneye.utils.add2os import listdir_flatten, keep_list_extension
from movneye.functional.read import recording_info, stimulus_info
from movneye.opening import load_roi, load_roi_with_margin
from movneye.packaging import return_preprocessed_events, find_fem_period
from movneye.filtering import return_filtered_events
from movneye.functional.wrap import write_bin


NUM_DATASET_SAMPLES = 70000
FEM_DURATION = 1.4 * 1e6


def parse_args():
    parser = argparse.ArgumentParser(description='Offline filtering and packing of DAVIS events.')

    parser.add_argument('-i', '--in_dir', type=str, required=True,
                        help='Directory where the input recordings to filter are stored (there must be .json files\n'
                             'in such dir or in some of its sub-dirs).')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='Directory where the output (noise-filtered) recordings should be stored (both "[].json"\n'
                             'and "[]_dvs.csv" files will be saved and organized with the same names and directory '
                             'structure as the input).\n'
                             'Note that if the output directory is the same as the input one, all DVS recording files\n'
                             'in "in_dir" will be overwritten with their filtered version.')
    parser.add_argument('--filter_file', type=str, default='./sensor/configs/dvsnoisefilter_biases.json',
                        help='Full path of the .json file where DVS background-noise filter parameters are stored.')
    parser.add_argument('--undistort', action="store_true", default=False,
                        help='Whether to undistort the output DVS events and APS frames according to the '
                             'calibration parameters\nin ./calib/res/calib2D/lens_distortion.xml.')
    parser.add_argument('--cut_fem', action="store_true", default=False,
                        help='Whether to cut the recording in the FEM interval gathered from the IMU events.\n'
                             'Default is False.')
    parser.add_argument('--crop_roi', action="store_true", default=False,
                        help='Whether to crop the output DVS events and APS frames according to the '
                             'ROI coordinates\nstored in ./calib/res/calib3D/img_roi.npy.')
    parser.add_argument('--fem_margin', type=float, default=None,
                        help='Fraction (in degrees) of the camera FOV to leave as a margin for the ROI, e.g. set'
                             '\nthe margin to the radius in which the FEMs are confined. Default is None.')

    return parser.parse_args()


def filter_and_preprocess_events(dvs_file: str, filter_file: str,
                                 imu_file: str = None, cut_fem: bool = True,
                                 undistort: bool = False, crop_roi: bool = False,
                                 fem_margin: float or None = None) -> np.ndarray:

    # Define the calibration file if you want to undistort the data
    calib_file = None
    if undistort:
        calib_file = CALIB2D_CAM_FILE

    # Load the top-left and bottom-right corners of the ROI (in the pixel array) as (x, y) coordinates
    roi = None
    if crop_roi:
        if fem_margin is not None:
            roi = load_roi_with_margin(margin=fem_margin)
        else:
            roi = load_roi(calib_file=calib_file)

    # Compute from IMU the timestamp when FEM begins (only if you want to cut DVS events in FEM interval)
    fem_start, fem_stop = None, None
    if cut_fem and imu_file:
        fem_start, _ = find_fem_period(imu_file, burnin=0, burnout=0)
        fem_stop = fem_start + FEM_DURATION

    # Filter DVS data
    events = return_filtered_events(dvs_file=dvs_file, filter_bias=filter_file,
                                    return_header=False, show_progress=False)

    # Pre-process DVS data
    events = return_preprocessed_events(dvs_events=events, fem_start=fem_start, fem_stop=fem_stop,
                                        calib_file=calib_file, roi=roi,
                                        burnin=0, burnout=0, refractory=None, hotpix_space_window=None)

    return events


def main(in_dir: str, out_dir: str, filter_file: str,
         cut_fem: bool, undistort: bool, crop_roi: bool, fem_margin: float or None = None):
    # Load all file names in the given directory
    all_files = sorted(keep_list_extension(listdir_flatten(in_dir), extension='.json', empty_error=True))

    # Loop through all recordings
    print('\nFiltering out all the data in the given directory...')
    for jsonfile2load in tqdm(all_files, total=len(all_files), desc='Progress', file=sys.stdout):
        _, dvs_file, imu_file, err_file = recording_info(jsonfile2load)
        _, img_index, img_label, img_split, _, _ = stimulus_info(jsonfile2load)
        assert err_file is None
        assert dvs_file is not None and os.path.isfile(dvs_file)
        assert imu_file is not None and os.path.isfile(imu_file)
        assert img_split is not None and isinstance(img_split, str)
        assert img_label is not None and isinstance(img_label, str)
        assert img_index is not None and isinstance(img_index, int)

        # Define output full path
        out_path = os.path.join(out_dir, img_split, img_label)
        os.makedirs(out_path, exist_ok=True)
        binfile2save = os.path.join(out_path, str(img_index).zfill(len(str(NUM_DATASET_SAMPLES)))) + '.bin'
        # binfile2save = os.path.join(out_path, file_name(dvs_file, keep_extension=False)) + '.bin'

        # Preprocess DVS events and write them to a binary file
        events = filter_and_preprocess_events(dvs_file=dvs_file, filter_file=filter_file, imu_file=imu_file,
                                              cut_fem=cut_fem, fem_margin=fem_margin,
                                              undistort=undistort, crop_roi=crop_roi)
        write_bin(bin_file=binfile2save, events=events)


if __name__ == "__main__":

    args = parse_args()

    main(in_dir=args.in_dir, out_dir=args.out_dir, filter_file=args.filter_file,
         undistort=args.undistort, cut_fem=args.cut_fem, crop_roi=args.crop_roi, fem_margin=args.fem_margin)
