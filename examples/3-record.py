import os
import sys
import time
import glob
import argparse
from visionart.stimulus import datasets
from visionart.motion.fem import load_femsteps
from visionart.recording import RecordImageFEM
from visionart.opening import load_roi, load_roi_with_margin
from visionart.functional.read import monitor_resolution
from visionart.functional.alert import sound_alert, email_alert
from visionart.calib.paths import SCREEN_FILE, CALIB3D_ROIimg_FILE
from visionart.usb.powercontrol import inactivate_usb_timeout
from visionart.usb.idfinder import find_usb_id

PAUSE_PERIOD = 8  # minutes

# Usage:
# >>    python3 3-record.py  --dataset 'mnist'  --batch_portion 0.05  --batch_start 5  --num_batches 2
#                            --num_consecutive_recs  --max_temperature 55
#                            --rec_dir ./data/rec/batch_2
#                            --fem_seed 1  --wait_time 30
#                            --fem_dir ./data/fem
#                            --config_file ./sensor/configs/davis346_biases.json
#                            --roi_filter  --what2rec 'dvs&imu'
#                            --img_duration 4000  --rec_duration 2500  --rec_timeout 5000
#                            --fem_delay 600  --rec_delay 300
#                            --second_monitor  --img_border 0.86  --img_border_color 255 255 255  --img_margin 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recording a set of images with a DAVIS sensor through FEMs.')

    # Experiment parameters
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='The type of dataset to load and display on the monitor. Can either be '
                             '"mnist" or "cifar10".\nDefault is mnist.')
    parser.add_argument('--batch_portion', type=float, default=1,
                        help='The portion (percentage) of the original dataset making up each batch that will be\n'
                             'converted to spiking with the neuromorphic camera in the current session.\n'
                             'Default is 1 (i.e. 100%, the whole dataset).')
    parser.add_argument('--batch_start', type=int, default=0,
                        help='The ID of the FIRST batch that should be converted to spiking during the current session:'
                             '\ni.e. the dataset is divided in N batches according to the given batch_portion parameter'
                             '\nand during the current session of recordings a given number of batches (num_batches)'
                             '\nwill be converted to spiking; which one to start from is determined by batch_start.'
                             '\nDefault is 0 (the first batch).')
    parser.add_argument('--num_batches', type=int, default=1,
                        help='The total number of dataset batches to record during the current session of recordings.'
                             '\nDefault is 1.')
    parser.add_argument('--num_consecutive_recs', type=int, default=100,
                        help=f'Number of consecutive recordings to run before pausing for {PAUSE_PERIOD} minutes (if\n'
                             f'temperature is too high). Default is 100.')
    parser.add_argument('--max_temperature', type=float, default=55,
                        help=f'Maximum temperature (in Celsius) of DAVIS device for pausing recordings (waiting for\n'
                             f'device to cool down). Default is 55 C')
    parser.add_argument('--wait_time', type=int, default=30,
                        help='Time period to wait (in minutes) after each session of recordings. Default is 30min.')
    parser.add_argument('--fem_seed', type=int, default=1,
                        help='The seed of the FEM sequence to use for all recordings. Default is 1.')
    parser.add_argument('--alert_me', action="store_true", default=False,
                        help='Whether to alert the user when the whole script has finished running.\n'
                             'Default is False.')

    # Recording parameters
    parser.add_argument('--rec_dir', type=str, default='./data/rec/nmnist',
                        help='Directory where the output .csv files of sensor events should be stored.\n'
                             'Note: a maximum of 6 files are stored for each recording (at least 2, depending on how\n'
                             'the parameter what2rec is set) with names derived from the image stimulus in img_dir:\n'
                             '      - [img_name].json\n'
                             '      - [img_name]_aps.avi\n'
                             '      - [img_name]_apsts.csv\n'
                             '      - [img_name]_dvs.csv\n'
                             '      - [img_name]_imu.csv\n'
                             '      - [img_name]_err.csv (only if there were issues, should never be present!)')
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
    parser.add_argument('--fem_dir', type=str, default='./data/fem',
                        help='Directory where the .npy FEM sequences are stored. Note that the files where FEM steps\n'
                             'are stored as Nx2 numpy arrays must have names with the structure: [#seed]_steps.npy')
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


def record_data(dataset_type, batch_portion, batch_start, num_batches,                         # Dataset parameters
                num_consecutive_recs, max_temperature, wait_time,                              # Session parameters
                rec_dir, rec_duration, rec_timeout, rec_delay,                                 # Recording parameters
                what2rec, config_file, noise_filter, roi_filter,
                fem_dir, fem_delay, fem_seed, speed_bounds, speed, acceleration, step,         # Movement parameters                                  # Fem
                img_duration, second_monitor, border_img, border_img_color, img_margin):       # Stimulus parameters

    # Create directory of the files to save (if it does not already exist)
    os.makedirs(rec_dir, exist_ok=True)

    # Load the top-left and bottom-right corners of the ROI (in the pixel array) as (x, y) coordinates
    roi = None
    if roi_filter:
        if img_margin is not None:
            roi = load_roi_with_margin(margin=img_margin)  # [[int,int],[int,int]]=[top-left[x,y],bottom-right[x,y]]
        else:
            roi = load_roi()                               # idem

    # Load information on the dataset batches
    builder = datasets.build_dataset(dataset_type, dataset_dir=None)
    dataset_info = datasets.info_dataset(builder, verbose=True)
    n_samples = {'train': dataset_info['splits']['train']['samples'],
                 'test': dataset_info['splits']['test']['samples']}
    batch_size = {'train': datasets.portion2batch(n_samples['train'], batch_portion)[0],
                  'test': datasets.portion2batch(n_samples['test'], batch_portion)[0]}
    tot_num_batches = datasets.portion2batch(n_samples['test'], batch_portion)[1]
    assert 0 <= batch_start <= tot_num_batches,\
        'The FIRST batch cannot be greater than the total number of batches.'
    assert 0 < num_batches <= tot_num_batches - batch_start, \
        f'The number of batches to record is < 1 or exceeds the total number of possible batches in the dataset ' \
        f'({tot_num_batches}),\n given that we start from batch {batch_start} and with each batch of size ' \
        f'{round(batch_portion*100,2)}% of the whole dataset.'

    # Sort all FEM sequences by their name (i.e. their seed)
    fem_file = sorted(glob.glob(os.path.join(fem_dir, '*_steps.npy')), key=os.path.basename)[fem_seed]

    # Estimate the duration of the whole session of recordings (i.e. the approximate time needed for recording all
    # the images in both the train and test batches)
    images4batch = batch_size['train'] + batch_size['test']
    duration4batch = datasets.evaluate_recording_duration(img_duration * 1e-3, images4batch, verbose=False)
    print(f'\nThe whole session of recordings will last approximately {round(duration4batch*num_batches, 1)} hours: '
          f'i.e. ~{int(img_duration * 1e-3)} seconds for each one of the {images4batch*num_batches} images to record.\n'
          f'The first session (batch {batch_start}) will start in 30 seconds and all the next {num_batches-1} batches '
          f'will be recorded at intervals of {wait_time} minutes.\n')
    time.sleep(30)

    # START THE MAIN LOOP
    k_rec, n_completed_recs = 0, 0
    for batch_id in range(batch_start, batch_start+num_batches):

        # Load the batch to record during the current session
        ds_batch = {'train': datasets.take_batch(datasets.load_dataset(builder, split='train'),
                                                 batch_size=batch_size['train'], batches2skip=batch_id),
                    'test': datasets.take_batch(datasets.load_dataset(builder, split='test'),
                                                batch_size=batch_size['test'], batches2skip=batch_id)}

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
            img=None, img_id=None, img_file=None, img_label=None, img_dataset=None, img_split=None,
            duration_img=img_duration,
            second_monitor=second_monitor, monitor_resolution=monitor_resolution(SCREEN_FILE),
            border_img=border_img, border_img_color=border_img_color if border_img_color is not None else "infer"
        )
        usb_id = find_usb_id(bus_number=myexp.davis_usb_bus_number,
                             device_address=myexp.davis_usb_device_address,
                             serial_number=myexp.davis_serial_number)

        # Loop through images in both train and test sets of the given batch and record them all
        for split in ['train', 'test']:
            rec_dir_split = os.path.join(rec_dir, split)
            os.makedirs(rec_dir_split, exist_ok=True)
            n_digits_split = len(str(n_samples[split] - 1))

            for n_img in range(batch_size[split]):
                n_completed_recs += 1
                k_rec += 1

                img = datasets.retrieve_image_from_batch(ds_batch[split], n_img)
                label = datasets.retrieve_label_from_batch(ds_batch[split], n_img)
                id_str = datasets.retrieve_id_from_batch(ds_batch[split], n_img)
                index = datasets.id2int(id_str, dataset_info)
                imgfile = os.path.join(dataset_info['general']['path'], id_str.split('__')[0])

                print(f'\n********************************************************************************************'
                      f'********************************************************************************************'
                      f'\nProgress: {round(n_completed_recs / (images4batch*num_batches) * 100, 3)} %'
                      f'\n{split.upper()} - {round(n_img / batch_size[split] * 100, 3)}% slot {batch_id-batch_start}'
                      f' - label: {label} - id: {index}'
                      f'\n********************************************************************************************'
                      f'********************************************************************************************')

                # Update movement sequence in the experiment object
                myexp.fem = load_femsteps(fem_file)
                myexp.fem_file = os.path.abspath(fem_file)
                myexp.fem_seed = fem_seed

                # Update image info in the experiment object
                myexp.img = datasets.preprocess_image(img, dataset_type=dataset_type)
                myexp.img_id = index
                myexp.img_label = label
                myexp.img_file = os.path.abspath(imgfile)
                myexp.img_split = split
                myexp.img_dataset = dataset_type

                # Update the full path of the file to save in the experiment object
                file2save = os.path.join(rec_dir_split, str(index).zfill(n_digits_split) + '.json')
                if os.path.isfile(file2save):
                    print('!!! CAUTION !!! The file to save already exists and will be over-written!',
                          file=sys.stderr)
                myexp.rec_file = os.path.abspath(file2save)

                # Run the experiment (recording)
                myexp.run()

                # If
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
                            img=None, img_id=None, img_file=None, img_label=None, img_dataset=None, img_split=None,
                            duration_img=img_duration,
                            second_monitor=second_monitor, monitor_resolution=monitor_resolution(SCREEN_FILE),
                            border_img=border_img,
                            border_img_color=border_img_color if border_img_color is not None else "infer"
                        )

        # Close communication with device and disable its USB for a given amount of time before starting the next session of recordings
        myexp.close()
        if batch_id < batch_start+num_batches-1:
            print(f'\nEnd of current session: {time.ctime()}')
            print(f'The next session of recordings will start in {wait_time} minutes and will last for '
                  f'{round(duration4batch, 1)} hours.\nDuring this session we will record the batch {batch_id+1} '
                  f'having {images4batch} images (i.e. {int(batch_portion*100)}% of the whole dataset).\n')
            deact = inactivate_usb_timeout(usb_id=usb_id, inactive_period=int(wait_time * 60))
            if not deact:
                raise Exception(f'No USB found with ID {usb_id}.')


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

    t_start = time.time()
    record_data(rec_dir=args.rec_dir,
                fem_dir=args.fem_dir, fem_seed=args.fem_seed,
                dataset_type=args.dataset, wait_time=args.wait_time,
                batch_portion=args.batch_portion, batch_start=args.batch_start, num_batches=args.num_batches,
                **rec_args)
    t_end = time.time()
    duration_script = round((t_end - t_start) / 3600, 2)  # in hours
    print(f'\n\nIt took {duration_script} hours to record all data:\n'
          f'    - start:   {time.ctime(t_start)}\n'
          f'    - finish:  {time.ctime(t_end)}\n')

    if args.alert_me:
        sound_alert()
        email_alert(message=f'Subject: Recordings finished!\n\n'
                            f'It took {duration_script} hours to record all data! You can take back your PC now.')
