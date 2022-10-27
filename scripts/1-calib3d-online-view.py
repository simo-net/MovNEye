import argparse
from movneye.motion.ptu import PTU, find_ptuport
from movneye.calib.calib3d import continuous_image_roi_search


def parse_args():
    parser = argparse.ArgumentParser(
        description='In this script we show APS frames online in an infinite loop to allow the user adjusting '
                    'the position of the monitor according to the desired ROI. For each frame, we superimpose '
                    'the 4 pixel coordinates of the detected ROI. Once the displayed points and the corresponding '
                    'ROI shape matches the desired one, you must force the loop interruption with CTRL+C. '
                    'No data will be stored during the process, it is only an utility tool for the user. '
                    'Note that you must run the script "1-calib3d.py" before and after running the present script.')

    # Image info
    parser.add_argument('--img_resolution', type=int, nargs=2, default=None,
                        help='Shape (width, height) of the images to use as stimuli.')
    parser.add_argument('--img_border', type=float, default=None,
                        help='Percentage of the monitor that should be occupied by a white border surrounding the image'
                             '\n(e.g. 0.3 means that 30% of the monitor is occupied by the border and 70% by image).\n'
                             'Default is None.')

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
    ptu.close()

    # Search for the image ROI and adapt the monitor/camera position accordingly
    continuous_image_roi_search(config_file=args.config_file,
                                border_img=args.img_border,
                                img_resolution=args.img_resolution)
