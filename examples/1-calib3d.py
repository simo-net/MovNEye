import argparse
from visionart.calib.paths import *
from visionart.calib import calib3d
from visionart.calib.chess import chess4screen
from visionart.motion.ptu import PTU, find_ptuport


# Usage:
# >>    python3 1-calib3d.py  --screen_resolution 1920 1080  --screen_diagonal 24
#                             --chess_shape 10 6  --chess_border 160 60
#                             --img_resolution 28 28  --img_border 0.86  --img_margin 1
#                             --config_file ./sensor/configs/davis346_biases.json


def parse_args():
    parser = argparse.ArgumentParser(
        description='In this script we do 4 main things:\n'
                    '  A) Create a suitable chessboard for the monitor used.\n'
                    f'     4 different files will be stored in "{INFO_DIR}":\n'
                    '        - screen.json: all info on the monitor (resolution, diagonal and PPI)\n'
                    '        - chess.json: all info on the chessboard (shape, border and dimension of each square);\n'
                    '        - chessboard.npy: the chessboard stimulus;\n'
                    '        - chesscorners.npy: the positions of chessboard internal corners (in pixels on monitor).\n'
                    '  B) Record APS frames with a DAVIS sensor while the previous chessboard stimulus is displayed\n'
                    '     on the monitor and use the results to compute the camera-monitor 3D transformation.\n'
                    f'     At least 2 files are stored in "{CALIB3D_RECORD_DIR}" (maximum 3, if issues occurred):\n'
                    '        - chessboard.json\n'
                    '        - chessboard_aps.avi\n'
                    '     Note that we record a full video of such chessboard for 1s and consider the median frame\n'
                    '     as a reference to compute the 3D calibration.\n'
                    '     Rotation and translation vectors between sensor and monitor are stored as .npy files\n'
                    f'     in "{CALIB3D_RESULT_DIR}".\n'
                    '  C) Find the 4 pixel coordinates, on the camera plane, corresponding to the 4 corners of\n'
                    '     the monitor. The resulting monitor-ROI and relative FOV are saved as .npy files\n'
                    f'     in "{CALIB3D_RESULT_DIR}".\n'
                    '  D) Find the 4 pixel coordinates, on the camera plane, corresponding to the 4 corners of\n'
                    '     an image with the following given characteristics: its native resolution and the amount of\n'
                    '     border that should surround it (as a fraction of the whole monitor plane). The resulting\n'
                    f'     image-ROI and relative FOV are saved as .npy files in "{CALIB3D_RESULT_DIR}".\n'
                    '     In this case APS frames are also recorded while a sample image having the given properties\n'
                    f'     is displayed on the monitor. Thus, at least 2 files are stored in "{CALIB3D_RECORD_DIR}":\n'
                    '        - image.json\n'
                    '        - image_aps.avi\n'
                    '     Note that step D) is only performed if "img_resolution" is given. If it is None,\n'
                    '     as by default, this whole process will be skipped.\n')

    # Monitor info
    parser.add_argument('--screen_resolution', type=int, nargs=2, default=(1920, 1080),
                        help='The resolution of the monitor where the chessboard image should be displayed (in units '
                             'of monitor pixels). Default is (1920, 1080).')
    parser.add_argument('--screen_diagonal', type=float, default=24,
                        help='The diagonal of the monitor (in inches). Default is 24".')

    # Chessboard info
    parser.add_argument('--chess_shape', type=int, nargs=2, default=(10, 6),
                        help='The number of squares in the chessboard on both directions (in units of monitor pixels). '
                             'Default is (10, 6).')  # also (9, 5)
    parser.add_argument('--chess_border', type=int, nargs=2, default=(160, 60),
                        help='The white border around the chessboard in both directions (in units of monitor pixels). '
                             'Default is (160, 60).')  # also (60, 40)

    # Image info
    parser.add_argument('--img_resolution', type=int, nargs=2, default=None,
                        help='Shape (width, height) of the images to use as stimuli.')
    parser.add_argument('--img_border', type=float, default=None,
                        help='Percentage of the monitor that should be occupied by a white border surrounding the image'
                             '\n(e.g. 0.3 means that 30% of the monitor is occupied by the border and 70% by image).\n'
                             'Default is None.')
    parser.add_argument('--img_info', type=str, default=None,
                        help='Some information (in the form of text) relative to the image used as stimulus.')
    parser.add_argument('--img_margin', type=float, default=None,
                        help='Portion (in degrees) of the camera FOV to leave as a margin for the ROI filter, e.g. set'
                             '\nthe margin equal to the radius in which the FEMs are confined. Default is None.')

    # Recording parameters
    parser.add_argument('--config_file', type=str, default=None,
                        help='Full path of the .json file where biases for DAVIS sensor configuration are stored.')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Initialize the PTU
    ptu = PTU(port=find_ptuport(verbose=False), steps=('E', 'E'), verbose=True)
    if any(ptu.positions):
        ptu.positions = (0, 0)
    # ptu.recalibrate()
    ptu.close()

    # Define the chessboard stimulus
    print('\n\n=================================== '
          'Define the chessboard stimulus '
          '====================================')
    chess4screen(resolution=tuple(args.screen_resolution), diagonal=args.screen_diagonal,     # monitor info
                 shape=tuple(args.chess_shape), border=tuple(args.chess_border))              # chessboard info
    # Find the camera-monitor 3D transformation
    print('\n\n============================== '
          'Find the camera-monitor 3D transformation '
          '==============================')
    calib3d.calibrate3d(config_file=args.config_file)
    # Find the monitor ROI
    print('\n\n===================================== '
          'Find the ROI of the MONITOR '
          '=====================================')
    calib3d.find_monitor_roi(show_results=True)
    # Find the image ROI
    if args.img_resolution:
        print('\n\n====================================== '
              'Find the ROI of the IMAGE '
              '======================================')
        img_bounds = calib3d.find_image_roi(config_file=args.config_file,
                                            border_img=args.img_border,
                                            img_resolution=args.img_resolution, img_strinfo=args.img_info,
                                            show_results=True, store_results=True)
        roi = calib3d.compute_roi_from_bounds(img_bounds)  # [[top_left[x,y]], [bottom_right[x,y]]]
        print('\nROI top-left and bottom-right (x,y) coordinates are: ', roi)
        print(f'\nThe output shape of the recording (without the FEM margin) will be:\n'
              f'  - width:   {roi[1][0] - roi[0][0]}\n'
              f'  - height:  {roi[1][1] - roi[0][1]}\n')

        if args.img_margin:
            img_bounds_with_margin = calib3d.find_image_roi_with_margin(margin=args.img_margin,
                                                                        show_results=True)
            roi = calib3d.compute_roi_from_bounds(img_bounds_with_margin)  # [[top_left[x,y]], [bottom_right[x,y]]]
            print(f'While including the FEM margin it will be:\n'
                  f'  - width:   {roi[1][0] - roi[0][0]}\n'
                  f'  - height:  {roi[1][1] - roi[0][1]}\n')
