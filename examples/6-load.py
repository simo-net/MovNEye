import argparse
from visionart.opening import load_dataset_recording

# Usage:
# >>    python3 6-load.py  -f ./data/rec/batch_4/test/8348.json  --undistort  --crop_roi  --cut_fem  --fem_margin 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualizing a file of recorded DAVIS events.')

    # Recording file
    parser.add_argument('-f', '--file2load', type=str, required=True,
                        help='Full path of the input .json file of recording info to load.\nNote: the names '
                             'for all the other files saved (APS, DVS and IMU events, as well as FEM and IMG info) '
                             'are taken from here.')

    # Visualization parameters
    parser.add_argument('--cut_fem', action="store_true", default=False,
                        help='Whether to cut the recording in the FEM interval gathered from the IMU events.\n'
                             'Default is False.')
    parser.add_argument('--undistort', action="store_true", default=False,
                        help='Whether to undistort the output DVS events and APS frames according to the '
                             'calibration parameters\nin ./calib/res/calib2D/lens_distortion.xml.')
    parser.add_argument('--crop_roi', action="store_true", default=False,
                        help='Whether to crop the output DVS events and APS frames according to the '
                             'ROI coordinates\nstored in ./calib/res/calib3D/img_roi.npy.')
    parser.add_argument('--fem_margin', type=float, default=None,
                        help='Fraction (in degrees) of the camera FOV to leave as a margin for the ROI, e.g. set'
                             '\nthe margin to the radius in which the FEMs are confined. Default is None.')
    parser.add_argument('--rec_burnin', type=float, default=0,
                        help='Time period (in ms) to cut off at the beginning of the recording. Default is 0.')
    parser.add_argument('--rec_burnout', type=float, default=0,
                        help='Time period (in ms) to cut off at the end of the recording. Default is 0.')
    parser.add_argument('--bin_size', type=float, default=20,
                        help='The time length (in ms) of each frame for visualizing the DVS events.')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    load_dataset_recording(file2load=args.file2load,  # Select the recording
                           cut_fem=args.cut_fem, undistort=args.undistort,
                           crop_roi=args.crop_roi,  # Cutting/undistorting parameters
                           refractory=None, hotpix_space_window=None,  # Filter parameter
                           bin_size=args.bin_size,  # Visualization parameters
                           rec_burnin=args.rec_burnin, rec_burnout=args.rec_burnout,
                           fem_margin=args.fem_margin,
                           verbose=True, plot=True)
