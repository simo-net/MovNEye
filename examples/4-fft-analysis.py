import argparse
import numpy as np
import matplotlib.pyplot as plt
from visionart.sensor import dvs, imu
from visionart.calib.paths import CALIB2D_CAM_FILE, CALIB3D_ROIimg_FILE
from visionart.utils.add2os import keep_list_extension, listdir_flatten
from visionart.functional.read import recording_info, sensor_info
from visionart.opening import load_roi


FEM_DURATION = 1.5e6  # us

# Usage:
#       python3 4-fft-analysis.py -i ./data/rec/fem  --undistort  --crop_roi  --bin_size 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualizing the average FFT profile across all recordings '
                    '(i.e. the # of events at all temporal frequencies).')

    # Recording directory
    parser.add_argument('-i', '--in_dir', type=str, required=True,
                        help='Directory where the input recordings to analyse are stored (there must be .json files\n'
                             'in such dir or in some of its sub-dirs).')

    # Visualization parameters
    parser.add_argument('--undistort', action="store_true", default=False,
                        help='Whether to undistort the output DVS events and APS frames according to the '
                             'calibration parameters\nin ./calib/res/calib2D/lens_distortion.xml.')
    parser.add_argument('--crop_roi', action="store_true", default=False,
                        help='Whether to crop the output DVS events and APS frames according to the '
                             'ROI coordinates\nstored in ./calib/res/calib3D/img_roi.npy.')
    parser.add_argument('--bin_size', type=float, default=1,
                        help='The time length (in ms) of each frame for visualizing the DVS events. Default is 1ms.')

    return parser.parse_args()


def find_fft(file2load,
             calib_file, roi,
             bin_size):

    # Define the files of recordings to load
    _, dvs_file, imu_file, _ = recording_info(file2load)
    if dvs_file is None or imu_file is None:
        raise Exception('DVS and IMU files not present since probably not recorded! Cannot run this script...')

    # Detect FEM interval from IMU data
    with imu.handler(reset_timestamps=False) as imu_handler:
        imu_handler.load_file(imu_file)
        imu_handler.find_fem()
        fem_start, _ = imu_handler.fem_start, imu_handler.fem_stop
        fem_stop = fem_start + FEM_DURATION

    # Pre-process DVS events
    with dvs.handler(reset_timestamps=False) as event_handler:
        event_handler.load_file(dvs_file)
        if calib_file is not None:
            _, model, serial = sensor_info()
            event_handler.undistort(calib_file, model=model, serial=serial)
        if roi is not None:
            event_handler.crop_region(start=roi[0], end=roi[1])
        event_handler.cut_timewindow(start=fem_start, stop=fem_stop)
        event_handler.timereset(reference_timestamp=fem_start)
        ifr_fem = event_handler.ifr_histogram(bin_size=bin_size * 1e3, duration=FEM_DURATION, smooth=False,
                                              num_neurons=1) * bin_size * 1e-3

    # Compute the FFT
    ifr_fft = np.abs(np.fft.fftshift(np.fft.fft(ifr_fem - ifr_fem.mean())))

    return ifr_fft[len(ifr_fem) // 2:]


def main(in_dir,
         undistort, crop_roi,
         bin_size):

    # Define the calibration file if you want to undistort the data
    if undistort:
        calib_file = CALIB2D_CAM_FILE
    else:
        calib_file = None

    # Load the top-left and bottom-right corners of the ROI (in the pixel array) as (x, y) coordinates
    if crop_roi:
        roi = load_roi(CALIB3D_ROIimg_FILE, calib_file=None)  # [[int,int],[int,int]]=[top-left[x,y],bottom-right[x,y]]
    else:
        roi = None

    # Find all files in the given directory
    all_files = keep_list_extension(listdir_flatten(in_dir), extension='.json', empty_error=True)
    n_files = len(all_files)

    # Find number of bins in the IFR histogram of all files (same since FEM_DURATION and the bin size are set to be
    # equal for all recs)
    n_bins = int(round(FEM_DURATION / bin_size * 1e-3))

    # Initialize the final average FFT profile (and relative frequencies)
    freq = np.fft.fftshift(np.fft.fftfreq(n_bins, d=bin_size * 1e-3))[n_bins//2:]  # Hz
    fft = np.zeros(len(freq))

    # Loop through all recordings in the given directory to find the average FFT (of the IFR)
    k_file = 1
    for file2load in all_files:
        print(f'Progress: {round(k_file / n_files * 100)} %', end="\r", flush=True)
        fft += find_fft(file2load, calib_file=calib_file, roi=roi, bin_size=bin_size) / n_files
        k_file += 1
    print('Progress: 100 %    (done!)')

    plt.figure()
    plt.plot(freq, fft)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('|events(f)|')
    plt.show()


if __name__ == '__main__':

    args = parse_args()

    main(args.in_dir,
         args.undistort, args.crop_roi,
         args.bin_size)
