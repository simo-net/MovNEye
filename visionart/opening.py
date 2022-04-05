import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from visionart.motion import fem
from visionart.stimulus import datasets
from visionart.sensor import aps, dvs, imu
from visionart.utils.add2np import gaussian_smoothing
from visionart.utils.add2cv import undistort_img_points, load_calibration
from visionart.calib.paths import SENSOR_FILE, CALIB2D_CAM_FILE, CALIB3D_ROIimg_FILE
from visionart.functional.read import read_json, sensor_info, movement_info, stimulus_info, recording_info
from visionart.calib.calib3d import find_image_roi_with_margin, compute_roi_from_bounds


# ---------------------------------------------------- Calibration ---------------------------------------------------

def load_roi(roi_file: str = CALIB3D_ROIimg_FILE, calib_file: str = None):
    assert os.path.isfile(roi_file), f'Yuo must run the 3D camera calibration in order to define a ROI in recordings.' \
                                     f'\n{roi_file} is not a valid file!'
    img_bounds = np.load(roi_file)
    if calib_file is None or not os.path.isfile(calib_file):
        roi = compute_roi_from_bounds(img_bounds)  # [[top_left[x,y]], [bottom_right[x,y]]]
    else:
        cam_shape, cam_model, cam_serial = sensor_info(SENSOR_FILE)
        cam_mtx, dist_coeffs = load_calibration(calib_file, model=cam_model, serial=cam_serial)
        img_bounds_undistorted = undistort_img_points(img_bounds, cam_shape, cam_mtx, dist_coeffs)
        roi = compute_roi_from_bounds(img_bounds_undistorted)  # [[top_left[x,y]], [bottom_right[x,y]]]
    return roi


def load_roi_with_margin(margin: float, calib_file: str = None):
    img_bounds_with_margin = find_image_roi_with_margin(margin=margin, show_results=False)
    if calib_file is None or not os.path.isfile(calib_file):
        roi = compute_roi_from_bounds(img_bounds_with_margin)  # [[top_left[x,y]], [bottom_right[x,y]]]
    else:
        cam_shape, cam_model, cam_serial = sensor_info(SENSOR_FILE)
        cam_mtx, dist_coeffs = load_calibration(calib_file, model=cam_model, serial=cam_serial)
        img_bounds_undistorted = undistort_img_points(img_bounds_with_margin, cam_shape, cam_mtx, dist_coeffs)
        roi = compute_roi_from_bounds(img_bounds_undistorted)  # [[top_left[x,y]], [bottom_right[x,y]]]
    return roi


# ---------------------------------------------------- Recordings ----------------------------------------------------

def load_aps(apsfile: str,  # Select the APS recording
             cut_in_fem: bool, calib_file: str, roi: [[int, int], [int, int]],  # Pre-processing parameters
             burnin: float = 0, burnout: float = 0,
             imufile: str = None,  # Select IMU file for cutting data in FEM
             verbose: bool = True, plot: bool = True):
    if verbose:
        print('\nLoading APS recording...')

    # Pre-process APS frames
    with aps.handler(reset_timestamps=False) as aps_handler:
        aps_handler.load_file(apsfile)
        aps_handler.cut_timewindow(start=aps_handler.start + burnin*1e3, stop=aps_handler.stop - burnout*1e3)
        rec_duration = aps_handler.duration
        if calib_file is not None:
            _, model, serial = sensor_info(SENSOR_FILE)
            aps_handler.undistort(calib_file, model=model, serial=serial)
        if roi is not None:
            aps_handler.crop_region(start=roi[0], end=roi[1])
        if verbose:
            print(f'Recording duration is ~{int(round(rec_duration * 1e-3))} ms and '
                  f'frame rate is {int(aps_handler.sampling)} Hz.')

        if cut_in_fem and imufile is not None and os.path.isfile(imufile):
            # Detect FEM interval from IMU data
            imu_handler = imu.handler(reset_timestamps=False)
            imu_handler.load_file(imufile)
            imu_handler.cut_timewindow(start=imu_handler.start + burnin * 1e3, stop=imu_handler.stop - burnout * 1e3)
            imu_handler.find_fem(cut_init=(0 if burnin != 0 else 100*1e3))
            fem_start, fem_stop = imu_handler.fem_finder.fem_start, imu_handler.fem_finder.fem_stop
            fem_duration = fem_stop - fem_start
            del imu_handler

            # Take only APS frames in FEM interval
            aps_handler.cut_timewindow(start=fem_start, stop=fem_stop)
            aps_handler.timereset(reference_timestamp=fem_start)
            if verbose:
                print(f'While FEM duration is ~{int(round(fem_duration * 1e-3))} ms.')
        else:
            aps_handler.timereset()

        if plot:
            aps_handler.show_video()
            aps_handler.show_frame()

        if verbose:
            print('\n')

        return aps_handler


def load_dvs(dvsfile: str,  # Select the DVS recording
             cut_in_fem: bool, calib_file: str, roi: [[int, int], [int, int]],  # Pre-processing parameters
             burnin: float = 0, burnout: float = 0,
             refractory: float = None, hotpix_space_window: int = None,  # Filtering parameters
             imufile: str = None,  # Select IMU file for cutting data in FEM
             bin_size: float = 1,  # Visualization parameters
             verbose: bool = True, plot: bool = True):
    if verbose:
        print('\nLoading DVS recording...')

    # Pre-process DVS events
    with dvs.handler(reset_timestamps=False) as dvs_handler:
        dvs_handler.load_file(dvsfile)
        dvs_handler.cut_timewindow(start=dvs_handler.start + burnin*1e3, stop=dvs_handler.stop - burnout*1e3)
        rec_duration = dvs_handler.duration
        if calib_file is not None:
            _, model, serial = sensor_info(SENSOR_FILE)
            dvs_handler.undistort(calib_file, model=model, serial=serial)
        if roi is not None:
            dvs_handler.crop_region(start=roi[0], end=roi[1])
        if refractory is not None:
            dvs_handler.refractory_filter(refractory=int(refractory*1e3))
        if hotpix_space_window is not None:
            dvs_handler.hot_pixels_filter(space_window=hotpix_space_window)
        if verbose:
            print(f'   - Duration is: ~{int(round(rec_duration * 1e-3))} ms\n'
                  f'   - Number of DVS events: {dvs_handler.num_events}\n'
                  f'   - Mean firing rate is: {round(dvs_handler.mean_firing_rate(), 2)} Hz\n'
                  f'   - Fraction of ON/OFF events is: {tuple([round(f, 2) for f in dvs_handler.fraction_onoff()])}')

        if imufile is not None and os.path.isfile(imufile):
            imu_handler = imu.handler(reset_timestamps=False)
            imu_handler.load_file(imufile)
            imu_handler.cut_timewindow(start=imu_handler.start + burnin*1e3, stop=imu_handler.stop - burnout*1e3)
            dvs_ts, dvs_id, dvs_pol = dvs_handler.ts, dvs_handler.id, dvs_handler.pol
            imu_ts, imu_speed = imu_handler.ts, imu_handler.angular_speed
            fem_ts, fem_steps_ts = None, None
            if cut_in_fem:
                # Detect FEM interval from IMU data
                imu_handler.find_fem((0 if burnin != 0 else 100*1e3))
                fem_start, fem_stop = imu_handler.fem_finder.fem_start, imu_handler.fem_finder.fem_stop
                fem_duration = fem_stop - fem_start
                fem_ts, fem_steps_ts = imu_handler.fem_finder.fem_timestamps, imu_handler.fem_finder.steps_start

                # Take only DVS events in FEM interval
                dvs_handler.cut_timewindow(start=fem_start, stop=fem_stop)
                dvs_handler.timereset(reference_timestamp=fem_start)
                if verbose:
                    print(f'After cutting the recording in the FEM period:\n'
                          f'   - Duration is: ~{int(round(fem_duration * 1e-3))} ms\n'
                          f'   - Number of DVS events: {dvs_handler.num_events}\n'
                          f'   - Mean firing rate is: {round(dvs_handler.mean_firing_rate(), 2)} Hz\n'
                          f'   - Fraction of ON/OFF events is: '
                          f'{tuple([round(f, 2) for f in dvs_handler.fraction_onoff()])}')
            del imu_handler

            if plot:
                plot_speedONrasterplot(dvs_ts=dvs_ts, dvs_id=dvs_id, dvs_pol=dvs_pol,
                                       imu_ts=imu_ts, imu_speed=imu_speed,
                                       fem_ts=fem_ts, fem_steps_ts=fem_steps_ts)

        else:
            dvs_handler.timereset()

            if plot:
                dvs_handler.show_rasterplot(show=True)

        if plot:
            dvs_handler.show_video_onoff_bluewhite(bin_size=bin_size*1e3)
            dvs_handler.show_surface_active_events(bin_size=bin_size*1e3)
            dvs_handler.show_time_difference(bin_size=bin_size*1e3)
            dvs_handler.show_view3D_onoff(duration=200*1e3)  # JUST SHOW THE FIRST 200ms (or plot will be too heavy)

            # dvsvid_on, dvsvid_off = dvs_handler.video_onoff(duration=200*1e3, bin_size=bin_size*1e3, smooth=False)
            # dvsvid = np.zeros((*dvsvid_on.shape, 3))  # BGR video
            # minval, maxval = min(dvsvid_on.min(), dvsvid_off.min()), max(dvsvid_on.max(), dvsvid_off.max())
            # dvsvid[..., 0] = np.uint8(np.interp(dvsvid_off, (minval, maxval), (0, 255)))  # red = OFF
            # dvsvid[..., 1] = np.uint8(np.interp(dvsvid_on, (minval, maxval), (0, 255)))  # green = ON
            # del dvsvid_on, dvsvid_off
            #
            # for frame in dvsvid:
            #     plt.figure()
            #     plt.imshow(frame)
            #     plt.show()

        if verbose:
            print('\n')

        return dvs_handler


# TODO: new function!!!!
def load_dvs_final(dvsfile: str, shape: (int, int) = (34, 34),
                   bin_size: float = 1, plot: bool = True, verbose: bool = True):

    if verbose:
        print('\nLoading DVS recording...')

    # Pre-process DVS events
    with dvs.handler(reset_timestamps=False) as dvs_handler:
        dvs_handler.load_file(dvsfile, shape=shape, rows4header=0, shape_from_header=False)
        if verbose:
            print(f'   - Duration is: ~{int(round(dvs_handler.duration * 1e-3))} ms\n'
                  f'   - Number of DVS events: {dvs_handler.num_events}\n'
                  f'   - Mean firing rate is: {round(dvs_handler.mean_firing_rate(), 2)} Hz\n'
                  f'   - Fraction of ON/OFF events is: {tuple([round(f, 2) for f in dvs_handler.fraction_onoff()])}\n')
        if plot:
            dvs_handler.show_rasterplot(show=True)
            dvs_handler.show_video_onoff_bluewhite(bin_size=bin_size*1e3)
            dvs_handler.show_surface_active_events(bin_size=bin_size*1e3)
            dvs_handler.show_time_difference(bin_size=bin_size*1e3)
            dvs_handler.show_view3D_onoff(duration=200*1e3)  # JUST SHOW THE FIRST 200ms (or plot will be too heavy)

        return dvs_handler


def plot_speedONrasterplot(dvs_ts: np.ndarray, dvs_id: np.ndarray, dvs_pol: np.ndarray,
                           imu_ts: np.ndarray, imu_speed: np.ndarray,
                           fem_ts: np.ndarray = None, fem_steps_ts: np.ndarray = None):
    start_ts, stop_ts = dvs_ts[0], dvs_ts[-1]
    fig, axs = plt.subplots(figsize=(14, 8), nrows=2, ncols=1, sharex='col')
    plt.suptitle("DVS raster-plot v.s. IMU angular speed")

    pol_on = (dvs_pol == 1)
    pol_off = np.logical_not(pol_on)
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Pixel ID')
    axs[0].set_xlim(0, (stop_ts - start_ts) * 1e-3)
    axs[0].plot((dvs_ts - start_ts)[pol_on] * 1e-3, dvs_id[pol_on], '|g', markersize=2, label='ON')
    axs[0].plot((dvs_ts - start_ts)[pol_off] * 1e-3, dvs_id[pol_off], '|r', markersize=2, label='OFF')
    axs[0].legend(loc='upper right', fontsize=14, markerscale=2.5)

    upper_speed = np.max(imu_speed) * 1.2
    imu_speed_smooth = gaussian_smoothing(imu_speed, window=6)
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Angular Speed (deg/s)')
    axs[1].set_ylim(0, upper_speed)
    axs[1].set_xlim(0, (stop_ts - start_ts) * 1e-3)
    axs[1].plot((imu_ts - start_ts) * 1e-3, imu_speed)
    axs[1].plot((imu_ts - start_ts) * 1e-3, imu_speed_smooth)

    if fem_ts is not None:
        axs[0].axvline(x=(fem_ts[0] - start_ts) * 1e-3, color='k', lw=2, ls='--')
        axs[0].axvline(x=(fem_ts[-1] - start_ts) * 1e-3, color='k', lw=2, ls='--')
        axs[1].axvline(x=(fem_ts[0] - start_ts) * 1e-3, color='k', lw=2, ls='--')
        axs[1].axvline(x=(fem_ts[-1] - start_ts) * 1e-3, color='k', lw=2, ls='--')
        if fem_steps_ts is not None:
            for start_step in fem_steps_ts:
                axs[1].axvline(x=(start_step - start_ts) * 1e-3, color='g', lw=1.2, ls='--')
            # axs[1].axvline(x=(fem_ts[-1] - start_ts) * 1e-3, color='r', lw=1.2, ls='--')

    plt.tight_layout()
    plt.show()


def print_dvs(dvsfile: str,
              burnin: float = 0, burnout: float = 0):
    with dvs.handler(reset_timestamps=False) as dvs_handler:
        dvs_handler.load_file(dvsfile)
        dvs_handler.cut_timewindow(start=dvs_handler.start + burnin*1e3, stop=dvs_handler.stop - burnout*1e3)
        dvs_dur = int(round(dvs_handler.duration * 1e-3))
        num_events = dvs_handler.num_events
        print('DVS:\n'
              f'   - Duration of DVS recording: {dvs_dur} ms\n'
              f'   - Number of DVS events: {num_events}\n'
              f'   - Mean firing rate of DVS events: {round(dvs_handler.mean_firing_rate(), 2)} Hz\n')


def print_aps(apsfile: str,
              burnin: float = 0, burnout: float = 0):
    with aps.handler(reset_timestamps=False) as aps_handler:
        aps_handler.load_file(apsfile)
        aps_handler.cut_timewindow(start=aps_handler.start + burnin*1e3, stop=aps_handler.stop - burnout*1e3)
        num_frames = aps_handler.num_frames
        aps_dur = int(round(aps_handler.duration * 1e-3))
        print('APS:\n'
              f'   - Duration of APS recording: {aps_dur} ms\n'
              f'   - Number of APS frames: {num_frames}\n'
              f'   - Frame rate of APS device: {int(num_frames / aps_dur * 1e3) if aps_dur > 0 else "..."} Hz\n')


def print_imu(imufile: str,
              burnin: float = 0, burnout: float = 0):
    with imu.handler(reset_timestamps=False) as imu_handler:
        imu_handler.load_file(imufile)
        imu_handler.cut_timewindow(start=imu_handler.start + burnin*1e3, stop=imu_handler.stop - burnout*1e3)
        imu_dur = int(round(imu_handler.duration * 1e-3))
        num_events = imu_handler.num_events
        print('IMU:\n'
              f'   - Duration of IMU recording: {imu_dur} ms\n'
              f'   - Number of IMU events: {imu_handler.num_events}\n'
              f'   - Sampling rate of IMU device: {int(num_events / imu_dur * 1e3) if imu_dur > 0 else "..."} Hz\n'
              f'   - Maximum IMU time-step is: {int(round(imu_handler.compute_dt_max() * 1e-3))} ms\n'
              f'   - Average device temperature is: {round(float(np.median(imu_handler.temperature)), 2)} °C\n')


def print_recording_info(file2load: str,
                         burnin: float = 0, burnout: float = 0):

    # Define all files of recordings to load
    aps_file, dvs_file, imu_file, _ = recording_info(file2load)

    # View the DVS recording
    if dvs_file is not None:
        print_dvs(dvs_file, burnin=burnin, burnout=burnout)

    # View the APS recording
    if aps_file is not None:
        print_aps(aps_file, burnin=burnin, burnout=burnout)

    # View the IMU recording
    if imu_file is not None:
        print_imu(imu_file, burnin=burnin, burnout=burnout)


# ----------------------------------------------------- Movement -----------------------------------------------------

def load_fem(femfile: str,
             verbose: bool = True, plot: bool = True):
    infofile = os.path.join(os.path.split(femfile)[0], 'fem_info.json')
    fieldfile = femfile[:-10] + '_activation-field.npy'
    ismsfile = femfile[:-10] + '_is-ms.npy'

    saw_args = read_json(infofile)
    fem_handler = fem.SAW(**saw_args, init_burnin=False)

    fem_handler.steps = np.load(femfile)
    fem_handler.activation_field = np.load(fieldfile)
    fem_handler.is_microsaccade = np.load(ismsfile)

    if verbose:
        print('\nLoading FEM information...', end='\r')
        fem_handler.print_info()
        print('\n')
    if plot:
        fem_handler.show_fem(unit=None, view_foveola=True, show=False)
        fem_handler.angles_distribution(plot=True, show=False)
        fem_handler.steps_sizes_distribution(unit='arcmin', plot=True, show=False)
        fem_handler.show_fem_positions(unit='arcmin', show=True)

    return fem_handler


# ----------------------------------------------------- Stimulus -----------------------------------------------------

def load_img_dataset(dataset_type: str, split: str = 'train', index: int = 0,
                     verbose: bool = True, plot: bool = True):
    ds = datasets.load_dataset(datasets.build_dataset(dataset_type, dataset_dir=None), split=split)
    sample = datasets.retrieve_sample_from_id(ds, index)

    img = datasets.retrieve_image(sample)
    label = datasets.retrieve_label(sample)

    if verbose:
        print('\nLoading STIMULUS information...')
        print(f'   - Dataset: {dataset_type.upper()}\n'
              f'   - Set split: {split}\n'
              f'   - Class label: {label}\n'
              f'   - Sample number: {index}/{len(ds)}\n')
    if plot:
        plt.figure(figsize=(6, 6))
        plt.title(f'Image {index}/{len(ds)} with label {label} from the {split.upper()} set')
        plt.imshow(img[..., 0], cmap='gray')
        plt.axis('off')
        plt.show()

    return img


# ======================================================================================================================
# ======================================================================================================================

def load_dataset_recording(file2load: str,  # Select the recording
                           cut_fem: bool, undistort: bool, crop_roi: bool,  # Cutting/cropping/undistort parameters
                           refractory: float or None, hotpix_space_window: int or None,  # Filtering parameters
                           bin_size: float,  # Visualization parameters
                           rec_burnin: float = 0, rec_burnout: float = 0,
                           fem_margin: float or None = None,
                           verbose: bool = True, plot: bool = True):

    # Define all files of recordings to load:
    # Recordings
    aps_file, dvs_file, imu_file, err_file = recording_info(file2load)
    if err_file is not None:
        manage_recording_issue(err_file)
    # Movement
    fem_file, fem_seed = movement_info(file2load)
    # Stimulus
    _, img_id, img_label, img_split, dataset_type = stimulus_info(file2load)

    # Define the calibration file if you want to undistort the data
    calib_file = None
    if undistort:
        calib_file = CALIB2D_CAM_FILE

    # Load the top-left and bottom-right corners of the ROI (in the pixel array) as (x, y) coordinates
    roi = None
    if crop_roi:
        if fem_margin is not None:
            roi = load_roi_with_margin(margin=fem_margin)  # [[int,int],[int,int]]=[top-left[x,y],bottom-right[x,y]]
        else:
            roi = load_roi(calib_file=calib_file)          # idem

    # View the APS recording
    if aps_file is not None:
        _ = load_aps(apsfile=aps_file,
                     cut_in_fem=cut_fem, calib_file=calib_file, roi=roi,
                     burnin=rec_burnin, burnout=rec_burnout,
                     imufile=imu_file,
                     verbose=verbose, plot=plot)

    # View the DVS recording
    if dvs_file is not None:
        dvs_handler = load_dvs(dvsfile=dvs_file,
                               cut_in_fem=cut_fem, calib_file=calib_file, roi=roi,
                               refractory=refractory, hotpix_space_window=hotpix_space_window,
                               burnin=rec_burnin, burnout=rec_burnout,
                               imufile=imu_file,
                               bin_size=bin_size,
                               verbose=verbose, plot=plot)

    # View the corresponding stimulus (dataset sample)
    if dataset_type is not None and img_split is not None and img_id is not None:
        img = load_img_dataset(dataset_type, split=img_split, index=img_id,
                               verbose=verbose, plot=False)

        if plot and dvs_file is not None:
            ifr_on, ifr_off = dvs_handler.video_onoff(bin_size=200 * 1e3)
            img_dvs = (ifr_on[0] - ifr_off[0])

            fig, axs = plt.subplots(nrows=1, ncols=2)
            plt.suptitle(f'{dataset_type.upper()} sample {img_id} from the {img_split.upper()} set - label {img_label}')
            axs[0].set_title(f'DVS recording {img_dvs.shape}\nmotion seed {fem_seed}')
            axs[0].imshow(img_dvs, cmap='gray')
            axs[0].axis('off')
            axs[1].set_title(f'Original image {img.shape[:2]}\nstatic display')
            axs[1].imshow(img[..., 0], cmap='gray')
            axs[1].axis('off')
            plt.show()

    # View the corresponding movement (FEM sequence)
    if fem_file is not None:
        _ = load_fem(fem_file,
                     verbose=verbose, plot=plot)


# TODO: new function!!!!
def load_preprocessed_recording(file2load: str, shape: (int, int) = (34, 34), bin_size: float = 10,
                                verbose: bool = True, plot: bool = True):
    # Recordings
    _, dvs_file, _, err_file = recording_info(file2load)
    if err_file is not None:
        manage_recording_issue(err_file)
    # Movement
    fem_file, fem_seed = movement_info(file2load)
    # Stimulus
    _, img_id, img_label, img_split, dataset_type = stimulus_info(file2load)

    # View the movement (FEM sequence)
    if fem_file is not None:
        _ = load_fem(fem_file,
                     verbose=verbose, plot=plot)

    # View the DVS recording
    dvs_handler = load_dvs_final(dvs_file, shape=shape, bin_size=bin_size, plot=True, verbose=True)

    # View the stimulus (dataset sample)
    img = load_img_dataset(dataset_type, split=img_split, index=img_id, verbose=verbose, plot=plot)

    # View them together
    ifr_on, ifr_off = dvs_handler.video_onoff(bin_size=1 * 1e3)
    img_dvs = (ifr_on[0] - ifr_off[0])
    fig, axs = plt.subplots(nrows=1, ncols=2)
    plt.suptitle(f'{dataset_type.upper()} sample {img_id} from the {img_split.upper()} set - label {img_label}')
    axs[0].set_title(f'DVS recording {img_dvs.shape}\nmotion seed {fem_seed}')
    axs[0].imshow(img_dvs, cmap='gray')
    axs[0].axis('off')
    axs[1].set_title(f'Original image {img.shape[:2]}\nstatic display')
    axs[1].imshow(img[..., 0], cmap='gray')
    axs[1].axis('off')
    plt.show()


def manage_recording_issue(errfile: str):
    print(f'\nThere were issues during the recording of this file, check the file {errfile} for more information:')
    if os.path.isfile(errfile):
        with open(errfile) as txt_err:
            msg = ''.join(txt_err.readlines())
            print(msg)
    view_anyway = input('\nDo you still want to view this recording? (y/n)  ')
    if view_anyway.lower() in ['y', 'yes']:
        pass
    else:
        sys.exit()
