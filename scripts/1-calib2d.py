import argparse
from movneye.calib.paths import *
from movneye.calib import calib2d


def parse_args():
    parser = argparse.ArgumentParser(
        description='Record APS frames (multiple one-shot/single-frame acquisitions) with a DAVIS sensor while a '
                    'physical chessboard stimulus is shown in front of it.\n'
                    f'Recordings can either be stored as .jpg files in "{CALIB2D_RECORD_DIR}" or only be used for '
                    'finding the camera calibration parameters.\n'
                    'At the end, the resulting camera matrix and distortion coefficients will be stored in a .xml file '
                    f'in "{CALIB2D_RESULT_DIR}".')

    # Recording parameters
    parser.add_argument('--config_file', type=str, default='./sensor/configs/davis346_biases.json',
                        help='Full path of the .json file where biases for DAVIS sensor configuration are stored.')
    parser.add_argument('--num_recs', type=int, default=200,
                        help='Number of frames to acquire for computing camera calibration matrix.')
    parser.add_argument('--save_recs', action="store_true", default=False,
                        help='Whether to store all the APS recordings (snapshots) as .jpg images in '
                             f'"{CALIB2D_RECORD_DIR}" (starting from "001.jpg").\nDefault is False.')
    parser.add_argument('--chess_shape', type=int, nargs=2, default=(8, 5),
                        help='Number of internal corners (internal b/w squares in both dimensions) of the physical '
                             'chessboard used.')
    parser.add_argument('--square_length', type=float, default=41.,
                        help='The length (in mm) of each square in the chessboard.')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    calib2d.calibrate2d(n_recs=args.num_recs, save_recs=args.save_recs,
                        chess_shape=args.chess_shape, square_length=args.square_length,
                        config_file=args.config_file)
