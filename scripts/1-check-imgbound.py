import argparse
from movneye.calib import calib3d
from movneye.motion.ptu import find_ptuport, PTU


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find the 4 pixel coordinates (the ROI), on the camera plane, corresponding to the 4 corners '
                    'of the image on the monitor.')

    # Image info
    parser.add_argument('--img_resolution', type=int, nargs=2, required=True,
                        help='Shape (width, height) of the images to use as stimuli.')
    parser.add_argument('--img_border', type=float, default=None,
                        help='Percentage of the monitor that should be occupied by a white border surrounding the image'
                             '\n(e.g. 0.3 means that 30% of the monitor is occupied by the border and 70% by image).\n'
                             'Default is None.')
    parser.add_argument('--img_info', type=str, default=None,
                        help='Some information (in the form of text) relative to the images to use as stimuli.')
    parser.add_argument('--img_margin', type=float, default=None,
                        help='Portion (in degrees) of the camera FOV to leave as a margin for the ROI filter, e.g. set'
                             '\nthe margin equal to the radius in which the FEMs are confined. Default is 1deg.')
    parser.add_argument('--store_results', action="store_true", default=False,
                        help='Whether to store all the resulting files to disk. Default is False.')

    # Recording parameters
    parser.add_argument('--config_file', type=str, default=None,
                        help='Full path of the .json file where biases for DAVIS sensor configuration are stored.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # Initialize the PTU
    ptu = PTU(port=find_ptuport(verbose=False), steps=('E', 'E'), verbose=True)
    if any(ptu.positions):
        ptu.positions = (0, 0)
    # ptu.recalibrate()
    ptu.close()

    img_bounds = calib3d.find_image_roi(config_file=args.config_file,
                                        border_img=args.img_border,
                                        img_resolution=args.img_resolution, img_strinfo=args.img_info,
                                        show_results=True, store_results=args.store_results)
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
