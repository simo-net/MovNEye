import os
import sys
import time
import argparse
from movneye.stimulus import datasets
from movneye.motion.fem import load_femsteps
from movneye.recording import RecordImageFEM
from movneye.functional.diagnose import detect_all_problems
from movneye.opening import load_roi, load_roi_with_margin, print_imu
from movneye.functional.read import monitor_resolution, recording_info, movement_info, stimulus_info, stimulus_dataset
from movneye.calib.paths import SCREEN_FILE, CALIB3D_ROIimg_FILE
from movneye.usb.powercontrol import inactivate_usb_timeout
from movneye.usb.idfinder import find_usb_id


PAUSE_PERIOD = 8  # minutes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recording a set of images with a DAVIS sensor through FEMs.')

    # Record-again parameters
    parser.add_argument('--rec_again', action="store_true", default=False,
                        help='Whether to record again all files that encountered an issue during recording.\n'
                             'Default is False.')
    parser.add_argument('--max_iters', type=int, default=10,
                        help='The maximum number of times for repeating the recordings to remove their issues.\n'
                             'Default is 10.')
    parser.add_argument('--num_consecutive_recs', type=int, default=100,
                        help=f'Number of consecutive recordings to run before pausing for {PAUSE_PERIOD} minutes (if\n'
                             f'temperature is too high). Default is 100.')
    parser.add_argument('--max_temperature', type=float, default=55,
                        help=f'Maximum temperature (in Celsius) of DAVIS device for pausing recordings (waiting for\n'
                             f'device to cool down). Default is 55 C')

    # Error-specific parameters
    parser.add_argument('--max_hole_duration', type=int, default=6,
                        help='The maximum total duration (in ms) of all holes in a recording for defining it bad.\n'
                             'Default is 6ms.')
    parser.add_argument('--rec_burnin', type=float, default=100,
                        help='Time period (in ms) to cut off at the beginning of the recording for detecting holes.\n'
                             'Default is 100ms.')
    parser.add_argument('--rec_burnout', type=float, default=100,
                        help='Time period (in ms) to cut off at the end of the recording for detecting holes.\n'
                             'Default is 100ms.')
    parser.add_argument('--min_fem_delay', type=int, default=200,
                        help='Minimum time delay (in ms) between the starting moment of all recordings and the actual\n'
                             'starting moment of the PTU movement.')

    # Recording parameters
    parser.add_argument('-i', '--rec_dir', type=str, required=True,
                        help='Directory where the recordings to check are stored.')
    parser.add_argument('--rec_duration', type=int, default=3e3,
                        help='Time duration (in ms) of the whole recording. Default is 3s.')
    parser.add_argument('--rec_timeout', type=float, default=5e3,
                        help='Time period to wait (in ms) for the recording thread to stream and log data\n'
                             'before killing it. Default is 5s.')
    parser.add_argument('--rec_delay', type=int, default=600,
                        help='Time period to wait (in ms) before starting to record data after the stimulation begins.'
                             '\nDefault is 600ms.')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Full path of the .json file where biases for DAVIS sensor configuration are stored.')
    parser.add_argument('--what2rec', type=str, default='all',
                        help='Whether to record APS only ("aps"), APS+IMU ("aps&imu"), '
                             'DVS only ("dvs"), DVS+IMU ("dvs&imu") or APS+DVS+IMU ("all").\n'
                             'Default is "all".')
    parser.add_argument('--noise_filter', action="store_true", default=False,
                        help='Whether to filter out the background noise on DVS events with the specific libcaer '
                             'functionalities.\nDefault is False.')
    parser.add_argument('--roi_filter', action="store_true", default=False,
                        help='Whether to crop the DVS events according to the ROI coordinates stored in '
                             f'{CALIB3D_ROIimg_FILE}.')
    # Stimulus parameters
    parser.add_argument('--img_duration', type=int, default=None,
                        help='Time duration (in ms) of the stimulation with the static image on the monitor.\n'
                             'Default is None (i.e. 5 times rec_duration).')
    parser.add_argument('--second_monitor', action="store_true", default=False,
                        help='Whether to display the image on a second monitor connected to the host computer.\n'
                             'Default is False.')
    parser.add_argument('--img_border', type=float, default=0.3,
                        help='Percentage of the monitor that should be occupied by a white border surrounding '
                             'the image.\n'
                             'Default is 0.3 (i.e. 30% of the monitor is occupied by the border and 70% by the image).')
    parser.add_argument('--img_border_color', type=int, nargs=3, default=None,
                        help='The color (int, int, int) of the border surrounding the image. If it is None it will be\n'
                             'inferred, i.e. adapted to the median color in the stimulus. Default is None (="infer").')
    parser.add_argument('--img_margin', type=float, default=None,
                        help='Fraction (in degrees) of the camera FOV to leave as a margin for the ROI filter, e.g. set'
                             '\nthe margin to the radius in which the FEMs are confined. Default is None.')
    # Movement parameters
    parser.add_argument('--fem_delay', type=int, default=600,
                        help='Time period to wait (in ms) before starting to move the PTU after the recording begins.\n'
                             'Default is 600ms.')
    parser.add_argument('--speed_bounds', type=int, nargs=3, default=(0, 150, 180),
                        help='Minimum, base (immediately reachable) and maximum allowable PTU speeds (in pos/sec).\n'
                             'Default is (0, 150, 180).')
    parser.add_argument('--speed', type=int, default=130,
                        help='Same target speed (in pos/sec) for both pan and tilt rotations. Default is 130.')
    parser.add_argument('--acceleration', type=int, default=180,
                        help='Same acceleration (in pos/sec^2) for both pan and tilt rotations. Default is 180.')
    parser.add_argument('--step_resolution', type=str, default='E',
                        help='The step mode of the PTU, which determines its resolution. It is expressed in fractions\n'
                             'of the max resolution: {"F"=full-step, "H"=half, "Q"=quarter, "E"=eighth, "A"=auto}.'
                             '\nDefault is "E".')

    return parser.parse_args()


def record_again(issued_recs,
                 num_consecutive_recs, max_temperature,                                         # Session parameters
                 rec_duration, rec_timeout, rec_delay,                                          # Recording parameters
                 what2rec, config_file, noise_filter, roi_filter,
                 fem_delay, speed_bounds, speed, acceleration, step,                            # Movement parameters
                 img_duration, second_monitor, border_img, border_img_color, img_margin):       # Stimulus parameters

    # Check if there are actually bad files to record again
    num_files = len(issued_recs)
    if not num_files:
        return None

    # Load the top-left and bottom-right corners of the ROI (in the pixel array) as (x, y) coordinates
    roi = None
    if roi_filter:
        if img_margin is not None:
            roi = load_roi_with_margin(margin=img_margin)  # [[int,int],[int,int]]=[top-left[x,y],bottom-right[x,y]]
        else:
            roi = load_roi()                               # idem

    # Load all static-image stimuli
    dataset_type = stimulus_dataset(issued_recs[0])
    builder = datasets.build_dataset(dataset_type, dataset_dir=None)
    train_ds, test_ds = datasets.load_dataset(builder, split=['train', 'test'])
    dataset_info = datasets.info_dataset(builder, verbose=False)
    classes = dataset_info['features']['label']['names']

    # Wait 30s before starting recording
    time.sleep(30)
    
    # Create the object for recording data (communicate with both PTU and DAVIS)
    myexp = RecordImageFEM(
        # 1) Recording info
        rec_file=None,
        duration_rec=rec_duration, timeout_rec=rec_timeout, delay_rec=rec_delay,
        noise_filter=noise_filter, roi=roi, config_file=config_file, what2rec=what2rec,
        # 2) Movement info
        fem_file=None, fem_seed=None,
        delay_fem=fem_delay,
        baud=9600, steps=(step, step), speed_bounds=speed_bounds, speed=speed, acc=acceleration,
        # 3) Stimulus info
        img=None, img_id=None, img_file=None, img_label=None,
        img_dataset=None, img_split=None, img_transforms=[],
        duration_img=img_duration,
        second_monitor=second_monitor, monitor_resolution=monitor_resolution(SCREEN_FILE),
        border_img=border_img, border_img_color=border_img_color if border_img_color is not None else "infer"
    )
    usb_id = find_usb_id(bus_number=myexp.davis_usb_bus_number,
                         device_address=myexp.davis_usb_device_address,
                         serial_number=myexp.davis_serial_number)

    # Loop through all issued recordings to record them again
    k_rec, n_completed_recs = 0, 0
    for rec_file in issued_recs:
        n_completed_recs += 1
        k_rec += 1

        print(
            f"\n---------------------------------------------------------------------------------------------\n"
            f"Progress: {round(n_completed_recs / num_files * 100, 3)} %  -  File: {rec_file}"
            f"\n---------------------------------------------------------------------------------------------\n"
        )

        # Grub motion sequence used
        fem_file, fem_seed = movement_info(rec_file)
        # Grub the stimulus index in the relative dataset and all stimulus' information
        imgfile, index, label, split, _, transforms = stimulus_info(rec_file)
        if split == 'train':
            sample = datasets.retrieve_sample_from_id(train_ds, index)
        elif split == 'test':
            sample = datasets.retrieve_sample_from_id(test_ds, index)
        else:
            raise Exception(
                f'The recording must be either in a "train" or "test" folder but this is not the case for {rec_file}.')
        img = datasets.retrieve_image(sample)
        assert label == classes[datasets.retrieve_label(sample)],\
            f'Image label does not correspond to that stored in the json file of the recording {rec_file}.'
        # Grub all recordings
        _, _, imu_file, err_file = recording_info(rec_file)
        if err_file is not None and os.path.isfile(err_file):
            print('The original issue was the following:')
            with open(err_file) as txt_err:
                msg = ''.join(txt_err.readlines())
                print(msg)
        else:
            print('Probably there were holes in the recording (check IMU sampling rate):')
            print_imu(imu_file)

        # Update movement sequence in the experiment object
        myexp.fem = load_femsteps(fem_file)
        myexp.fem_file = os.path.abspath(fem_file)
        myexp.fem_seed = fem_seed

        # Update image info in the experiment object
        myexp.img_dataset = dataset_type
        myexp.img_split = split
        myexp.img_transforms = transforms
        myexp.img_id = index
        myexp.img_label = label
        myexp.img_file = os.path.abspath(imgfile)
        myexp.img = img

        # Update the full path of the file to save in the experiment object
        myexp.rec_file = os.path.abspath(rec_file)

        # Run the experiment (recording)
        myexp.run()  # note all old recordings are automatically deleted at this point

        # After a set of consecutive recordings, check the temperature of the device and, if it is too high,
        # interrupt communication with device and disable its USB port for a given amount of time
        if k_rec > num_consecutive_recs:
            temperature = myexp.check_temperature()
            if temperature is None:
                pass
            elif temperature < max_temperature:
                k_rec = 0
                pass
            else:
                k_rec = 0
                # Close the DAVIS object
                myexp.close()
                # Disable the USB of the DAVIS device for a given amount of time
                deact = inactivate_usb_timeout(usb_id=usb_id, inactive_period=int(PAUSE_PERIOD * 60))
                if not deact:
                    raise Exception(f'No USB found with ID {usb_id}.')
                # Re-open the DAVIS object
                myexp = RecordImageFEM(
                    # 1) Recording info
                    rec_file=None,
                    duration_rec=rec_duration, timeout_rec=rec_timeout, delay_rec=rec_delay,
                    noise_filter=noise_filter, roi=roi, config_file=config_file, what2rec=what2rec,
                    # 2) Movement info
                    fem_file=None, fem_seed=None,
                    delay_fem=fem_delay,
                    baud=9600, steps=(step, step), speed_bounds=speed_bounds, speed=speed, acc=acceleration,
                    # 3) Stimulus info
                    img=None, img_id=None, img_file=None, img_label=None,
                    img_dataset=None, img_split=None, img_transforms=[],
                    duration_img=img_duration,
                    second_monitor=second_monitor, monitor_resolution=monitor_resolution(SCREEN_FILE),
                    border_img=border_img,
                    border_img_color=border_img_color if border_img_color is not None else "infer"
                )

    # Close communication with devices and exit
    myexp.close()


def main(rec_dir,
         rec_again, max_iters,
         min_fem_delay, max_hole_duration, rec_burnin, rec_burnout,
         args4rec):

    all_problematic_files = detect_all_problems(rec_dir=rec_dir,
                                                rec_burnin=rec_burnin, rec_burnout=rec_burnout,
                                                max_hole_duration=max_hole_duration, min_fem_delay=min_fem_delay,
                                                verbose=True, show_pbars=True)

    if rec_again:
        k = 1
        while True:

            # Interrupt if nothing to record
            if not all_problematic_files:
                print('\nThere are no more corrupted files to record again. Ending.\n', file=sys.stderr)
                break

            print('==================================================================================================')
            print(f'=========================================  slot {k}  =============================================')
            print('==================================================================================================')

            # Record again all the detected files with problems
            record_again(all_problematic_files,
                         **args4rec)

            # Check if there are major issues in the new recordings and record them again
            issues = detect_all_problems(rec_list=all_problematic_files,
                                         rec_burnin=rec_burnin, rec_burnout=rec_burnout,
                                         max_hole_duration=max_hole_duration, min_fem_delay=min_fem_delay,
                                         verbose=True, show_pbars=True)
            while issues:
                record_again(issues,
                             **args4rec)
                issues = detect_all_problems(rec_list=all_problematic_files,
                                             rec_burnin=rec_burnin, rec_burnout=rec_burnout,
                                             max_hole_duration=max_hole_duration, min_fem_delay=min_fem_delay,
                                             verbose=True, show_pbars=True)

            # Interrupt if you reached the maximum number of consecutive iterations
            if k >= max_iters:
                print('\nThe maximum number of repeated recordings was reached. Ending.\n', file=sys.stderr)
                break
            else:
                time.sleep(15*60)

            # Check the recordings with issues for the next iteration:
            # note that now we only check the previously-detected bad recordings, not all the files in rec_dir
            all_problematic_files = detect_all_problems(rec_list=all_problematic_files,
                                                        rec_burnin=rec_burnin, rec_burnout=rec_burnout,
                                                        max_hole_duration=max_hole_duration, min_fem_delay=min_fem_delay,
                                                        verbose=True, show_pbars=True)
            k += 1


if __name__ == "__main__":

    args = parse_args()

    rec_args = dict(rec_duration=args.rec_duration,
                    rec_timeout=args.rec_timeout,
                    rec_delay=args.rec_delay,
                    what2rec=args.what2rec,
                    config_file=args.config_file,
                    noise_filter=args.noise_filter,
                    roi_filter=args.roi_filter,

                    fem_delay=args.fem_delay,
                    speed_bounds=args.speed_bounds,
                    speed=args.speed,
                    acceleration=args.acceleration,
                    step=args.step_resolution,

                    img_duration=args.img_duration,
                    second_monitor=args.second_monitor,
                    border_img=args.img_border,
                    border_img_color=args.img_border_color,
                    img_margin=args.img_margin,

                    num_consecutive_recs=args.num_consecutive_recs,
                    max_temperature=args.max_temperature)

    main(rec_dir=args.rec_dir,
         rec_again=args.rec_again, max_iters=args.max_iters,
         min_fem_delay=args.min_fem_delay, max_hole_duration=args.max_hole_duration,
         rec_burnin=args.rec_burnin, rec_burnout=args.rec_burnout,
         args4rec=rec_args)
