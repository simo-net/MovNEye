import os
import sys
import warnings
import numpy as np
from tqdm import tqdm
from visionart.sensor import imu, dvs
from visionart.functional.read import read_json, recording_imufile, recording_dvsfile, recording_errfile
from visionart.utils.add2os import listdir_flatten, keep_list_extension


# --------------------------------------------------- All Problems ---------------------------------------------------

def detect_all_problems(rec_dir: str = None, rec_list: list = None,
                        rec_burnin: float = 0, rec_burnout: float = 0,
                        thr_num_events: int = 5000, min_fem_delay: int = 5000, max_hole_duration: float = 5,
                        min_speed: float = 1e-3, min_temperature: float = 15, max_temperature: float = 55,
                        verbose: bool = False, show_pbars: bool = True) -> list or None:
    if rec_list is None and rec_dir is not None:
        assert isinstance(rec_dir, str), 'The directory must be a string.'
        assert os.path.isdir(rec_dir), f'The given directory must actually exist. Check if {rec_dir} is present.'
        all_files = sorted(keep_list_extension(listdir_flatten(rec_dir), '.json', empty_error=False))
    elif rec_dir is None and rec_list is not None:
        assert isinstance(rec_list, list), 'The list of recordings to check must be a list of strings.'
        all_files = sorted(rec_list)
    else:
        raise ValueError('You must specify either a directory where the recordings to check are stored or a list '
                         'of full paths of files to check.')
    num_files = len(all_files)
    issue_files = detect_issues(all_files,
                                verbose=False, show_pbar=show_pbars)
    all_files = sorted(list(set(all_files)-set(issue_files)))
    empty_files = detect_nodata(all_files,
                                check_speed=True, check_temperature=True, check_events=True,
                                thr_num_events=thr_num_events,
                                min_speed=min_speed, min_temp=min_temperature, max_temp=max_temperature,
                                verbose=False, show_pbar=show_pbars)
    all_files = sorted(list(set(all_files)-set(empty_files)))
    delay_files = detect_delay(all_files,
                               min_fem_delay=min_fem_delay,
                               verbose=False, show_pbar=show_pbars)
    all_files = sorted(list(set(all_files)-set(delay_files)))
    holes_files = detect_holes(all_files,
                               rec_burnin=rec_burnin, rec_burnout=rec_burnout,
                               max_hole_duration=max_hole_duration,
                               verbose=False, show_pbar=show_pbars)
    problem_files = sorted(issue_files + empty_files + delay_files + holes_files)
    if not problem_files:
        if verbose:
            print('\nNo problems were detected in the recordings.\n')
        return None
    if verbose:
        print('\nThe following recordings have bad data due to at least one of these reasons:\n' +
              '--> an acquisition-timeout/process-order issue\n' +
              '--> too low FEM delay (wrt start of recording)\n' +
              '--> holes in the event stream (both DVS and IMU)\n' +
              '--> missing IMU/DVS data (no DVS events, no IMU speed or low/high temperature)\n')
        for problem in problem_files:
            print(f'  - {problem}')
        print(f'There are {len(problem_files)}/{num_files} files with issues detected.\n')
    return problem_files


# ---------------------------------- 1) Acquisition issue (process order or timeout) ---------------------------------

def detect_issues(all_files: list,
                  verbose: bool = True, show_pbar: bool = False) -> list or None:
    files_with_issues = []
    pbar = tqdm(all_files, total=len(all_files), desc='Detecting issues during recording', file=sys.stdout,
                disable=not show_pbar)
    for file in all_files:
        errfile = recording_errfile(read_json(file))
        if errfile is not None and os.path.isfile(errfile):
            files_with_issues.append(file)
            if verbose:
                with open(errfile) as txt_err:
                    print(f' --> File: {file}')
                    msg = ''.join(txt_err.readlines())
                    print(msg)
        pbar.update(1)
    pbar.close()
    return files_with_issues


# ---------------------------------------------------- 2) No Data ----------------------------------------------------

def detect_nodata(all_files: list,
                  check_speed: bool = True, min_speed: float = 1e-3,
                  check_temperature: bool = True, min_temp: float = 15, max_temp: float = 55,
                  check_events: bool = True, thr_num_events: int = 5000,
                  verbose: bool = True, show_pbar: bool = False) -> list or None:
    warnings.filterwarnings('error')
    pbar = tqdm(all_files, total=len(all_files), desc='Detecting missing IMU data in recordings', file=sys.stdout,
                disable=not show_pbar)
    files_with_nodata = []
    for file in all_files:
        imufile = recording_imufile(read_json(file))
        if imufile is not None and os.path.isfile(imufile):
            with imu.handler(reset_timestamps=False) as imu_handler:
                imu_handler.load_file(imufile)
                try:
                    temp = float(np.median(imu_handler.temperature))
                    speed = float(np.abs(np.median(imu_handler.angular_speed)))
                except RuntimeWarning:
                    temp, speed = np.nan, np.nan
                speed_issue = False
                if check_speed and not speed > min_speed:
                    if verbose:
                        print(f' --> File: {file}\n'
                              f'     The median IMU speed of the DAVIS device during the recording was '
                              f'{round(speed, 2)}°/s.')
                    speed_issue = True
                    files_with_nodata.append(file)
                if check_temperature and not (min_temp < temp < max_temp):
                    if verbose:
                        if not check_speed and not speed_issue:
                            print(f' --> File: {file}')
                        print(f'     The median IMU temperature of the DAVIS device during the recording was '
                              f'{round(temp, 2)}°C.')
                    if not speed_issue:
                        files_with_nodata.append(file)
        else:
            raise Exception(f'The recording {file} does not have an IMU file and it is required for finding the speed in the recording.')
        dvsfile = recording_dvsfile(read_json(file))
        if check_events and file not in files_with_nodata:
            if dvsfile is not None and os.path.isfile(dvsfile):
                with dvs.handler(reset_timestamps=False) as dvs_handler:
                    dvs_handler.load_file(dvsfile)
                    num_events = dvs_handler.num_events
                if num_events < thr_num_events:
                    if verbose:
                        print(f' --> File: {file}\n'
                              f'     The number of DVS events of the DAVIS device during the recording was {num_events}.')
                    files_with_nodata.append(file)
            else:
                raise Exception(f'The recording {file} does not have an DVS file and it is required for finding the speed in the recording.')
        pbar.update(1)
    pbar.close()
    return files_with_nodata


# ----------------------------------------------------- 3) Holes -----------------------------------------------------

def detect_holes(all_files: list,
                 rec_burnin: float = 0, rec_burnout: float = 0,
                 max_hole_duration: float = 5,
                 verbose: bool = True, show_pbar: bool = False) -> (list, list, list):
    pbar = tqdm(all_files, total=len(all_files), desc='Detecting holes in recordings', file=sys.stdout,
                disable=not show_pbar)
    files_with_holes = []
    for file in all_files:
        imufile = recording_imufile(read_json(file))
        if imufile is not None and os.path.isfile(imufile):
            with imu.handler(reset_timestamps=False) as imu_handler:
                imu_handler.load_file(imufile)
                imu_handler.cut_timewindow(start=imu_handler.start + rec_burnin*1e3,
                                           stop=imu_handler.stop - rec_burnout*1e3)
                # all_dts = np.diff(imu_handler.ts) * 1e-3
                all_dts = np.round(np.diff(imu_handler.ts) * 1e-3)
                median_dt = int(np.median(all_dts))
                # hole = float(np.sum(all_dts[all_dts > (median_dt + 0.5)]))
                hole = float(np.sum(all_dts[all_dts > median_dt]))
                if hole >= max_hole_duration:
                    if verbose:
                        print(f' --> File: {file}\n'
                              f'   - Sampling rate of IMU device is: {int(imu_handler.num_events / (imu_handler.duration * 1e-6))} Hz\n'
                              f'   - Total hole duration is: {int(round(hole))} ms\n')
                    files_with_holes.append(file)
        else:
            raise Exception(f'The recording {file} does not have an IMU file and it is required for finding holes '
                            'in the recording.')
        pbar.update(1)
    pbar.close()
    return files_with_holes


# # TODO: this function was deprecated, updated with the upper one
# def detect_holes_old(all_files: list,
#                      rec_burnin: float = 0, rec_burnout: float = 0,
#                      samples4hole: int = 2,
#                      verbose: bool = True, show_pbar: bool = False) -> list or None:
#     pbar = tqdm(all_files, total=len(all_files), desc='Detecting holes in recordings', file=sys.stdout,
#                 disable=not show_pbar)
#     files_with_holes = []
#     for file in all_files:
#         imufile = recording_imufile(read_json(file))
#         if imufile is not None and os.path.isfile(imufile):
#             with imu.handler(reset_timestamps=False) as imu_handler:
#                 imu_handler.load_file(imufile)
#                 imu_handler.cut_timewindow(start=imu_handler.start + rec_burnin*1e3,
#                                            stop=imu_handler.stop - rec_burnout*1e3)
#                 max_dt = imu_handler.compute_dt_max()
#                 if max_dt > samples4hole * imu_handler.compute_dt():
#                     if verbose:
#                         imu_dur = int(round(imu_handler.duration * 1e-3))
#                         num_events = imu_handler.num_events
#                         print(f' --> File: {file}\n'
#                               f'   - Sampling rate of IMU device: '
#                               f'{int(num_events / imu_dur * 1e3) if imu_dur > 0 else "..."} Hz\n'
#                               f'   - Maximum IMU time-step is: {int(round(max_dt * 1e-3))} ms\n')
#                     files_with_holes.append(file)
#         else:
#             raise Exception(f'The recording {file} does not have an IMU file and it is required for finding holes '
#                             'in the recording.')
#         pbar.update(1)
#     pbar.close()
#     return files_with_holes

# # TODO: still work in progress...
# def detect_holes_new(all_files: list,
#                      rec_burnin: float = 0, rec_burnout: float = 0,
#                      samples4hole: int = 2, maxnumholes: int = 5, target_sampling: float = 999,
#                      verbose: bool = True, show_pbar: bool = False) -> (list, list, list):
#     pbar = tqdm(all_files, total=len(all_files), desc='Detecting holes in recordings', file=sys.stdout,
#                 disable=not show_pbar)
#     files_with_holes = []
#     for file in all_files:
#         imufile = recording_imufile(read_json(file))
#         if imufile is not None and os.path.isfile(imufile):
#             with imu.handler(reset_timestamps=False) as imu_handler:
#                 imu_handler.load_file(imufile)
#                 imu_handler.cut_timewindow(start=imu_handler.start + rec_burnin*1e3,
#                                            stop=imu_handler.stop - rec_burnout*1e3)
#                 imu_dur = imu_handler.duration * 1e-6  # seconds
#                 try:
#                     sampling = int(imu_handler.num_events / imu_dur)
#                 except ZeroDivisionError:
#                     sampling = imu_handler.sampling
#                 all_dts = np.diff(imu_handler.ts)
#                 max_dt = np.max(all_dts)
#                 median_dt = int(np.median(all_dts))
#                 n_holes = np.sum(all_dts >= (samples4hole - 1) * median_dt)
#                 if max_dt > samples4hole * median_dt or n_holes >= maxnumholes or sampling < target_sampling:
#                     if verbose:
#                         print(f' --> File: {file}\n'
#                               f'   - Sampling rate of IMU device: {sampling if imu_dur > 0 else "..."} Hz\n'
#                               f'   - Maximum IMU time-step is: {int(round(max_dt * 1e-3))} ms\n'
#                               f'   - Number of holes of at least {samples4hole * int(median_dt*1e-3)} ms: {n_holes}\n'
#                               )
#                     files_with_holes.append(file)
#         else:
#             raise Exception(f'The recording {file} does not have an IMU file and it is required for finding holes '
#                             'in the recording.')
#         pbar.update(1)
#     pbar.close()
#     return files_with_holes


def detect_holes_distribution(all_files: list,
                              rec_burnin: float = 0, rec_burnout: float = 0,
                              num_bins: int = 100, max_hole_duration: float = 20,  # in milliseconds
                              show_pbar: bool = False) -> (np.ndarray, np.ndarray):
    pbar = tqdm(all_files, total=len(all_files), desc='Detecting holes in recordings', file=sys.stdout,
                disable=not show_pbar)
    all_distributions = np.zeros((len(all_files), num_bins), dtype=np.uint16)
    bins = np.linspace(0, max_hole_duration, num_bins, endpoint=False)
    for k, file in enumerate(all_files):
        imufile = recording_imufile(read_json(file))
        if imufile is not None and os.path.isfile(imufile):
            with imu.handler(reset_timestamps=False) as imu_handler:
                imu_handler.load_file(imufile)
                imu_handler.cut_timewindow(start=imu_handler.start + rec_burnin*1e3,
                                           stop=imu_handler.stop - rec_burnout*1e3)
                all_dts = np.diff(imu_handler.ts)
            hist, _ = np.histogram(all_dts * 1e-3, bins=num_bins, range=(0, max_hole_duration))
            all_distributions[k] = hist
        else:
            raise Exception(f'The recording {file} does not have an IMU file and it is required for finding holes '
                            'in the recording.')
        pbar.update(1)
    pbar.close()
    return all_distributions, bins


# # TODO: still work in progress...
# def detect_holes_distribution_new(all_files: list,
#                                   rec_burnin: float = 0, rec_burnout: float = 0,
#                                   show_pbar: bool = False) -> (np.ndarray, np.ndarray):
#     pbar = tqdm(all_files, total=len(all_files), desc='Detecting holes in recordings', file=sys.stdout,
#                 disable=not show_pbar)
#     holes_distribution = np.zeros(len(all_files), dtype=np.uint16)
#     for k, file in enumerate(all_files):
#         imufile = recording_imufile(read_json(file))
#         if imufile is not None and os.path.isfile(imufile):
#             with imu.handler(reset_timestamps=False) as imu_handler:
#                 imu_handler.load_file(imufile)
#                 imu_handler.cut_timewindow(start=imu_handler.start + rec_burnin*1e3,
#                                            stop=imu_handler.stop - rec_burnout*1e3)
#                 # all_dts = np.diff(imu_handler.ts) * 1e-3
#                 all_dts = np.round(np.diff(imu_handler.ts) * 1e-3)
#                 median_dt = int(np.median(all_dts))
#                 # hole = float(np.sum(all_dts[all_dts > (median_dt + 0.5)]))
#                 hole = float(np.sum(all_dts[all_dts > median_dt]))
#             holes_distribution[k] = hole
#         else:
#             raise Exception(f'The recording {file} does not have an IMU file and it is required for finding holes '
#                             'in the recording.')
#         pbar.update(1)
#     pbar.close()
#     return holes_distribution
#
#
# def detect_holes_distribution_newnew(all_files: list,
#                                      rec_burnin: float = 0, rec_burnout: float = 0,
#                                      num_bins: int = 1000,
#                                      show_pbar: bool = False) -> (np.ndarray, np.ndarray):
#     holes_distribution_inrecs = detect_holes_distribution_new(all_files=all_files,
#                                                               rec_burnin=rec_burnin, rec_burnout=rec_burnout,
#                                                               show_pbar=show_pbar)
#     holes_distribution, bins = np.histogram(holes_distribution_inrecs,
#                                             bins=num_bins, range=(0, max(holes_distribution_inrecs)+1))
#     return holes_distribution, bins[-1]


def detect_sampling_distribution(all_files: list,
                                 rec_burnin: float = 0, rec_burnout: float = 0,
                                 show_pbar: bool = False) -> (list, list) or (None, None):
    pbar = tqdm(all_files, total=len(all_files), desc='Detecting holes in recordings', file=sys.stdout,
                disable=not show_pbar)
    all_samplings = np.zeros(len(all_files), dtype=np.uint16)
    for k, file in enumerate(all_files):
        imufile = recording_imufile(read_json(file))
        if imufile is not None and os.path.isfile(imufile):
            with imu.handler(reset_timestamps=False) as imu_handler:
                imu_handler.load_file(imufile)
                imu_handler.cut_timewindow(start=imu_handler.start + rec_burnin*1e3,
                                           stop=imu_handler.stop - rec_burnout*1e3)
                try:
                    sampling = int(imu_handler.num_events / (imu_handler.duration * 1e-6))
                except ZeroDivisionError:
                    sampling = imu_handler.sampling
                all_samplings[k] = sampling
        else:
            raise Exception(f'The recording {file} does not have an IMU file and it is required for finding holes '
                            'in the recording.')
        pbar.update(1)
    pbar.close()
    sampling_distribution, bins = np.histogram(all_samplings, bins=1000, range=(1, 1001))
    return sampling_distribution, bins[:-1]


def detect_holes_sampling(all_files: list,
                          rec_burnin: float = 0, rec_burnout: float = 0,
                          target_sampling: float = 999,
                          show_pbar: bool = False) -> list:
    pbar = tqdm(all_files, total=len(all_files), desc='Detecting holes in recordings', file=sys.stdout,
                disable=not show_pbar)
    files_with_holes = []
    for k, file in enumerate(all_files):
        imufile = recording_imufile(read_json(file))
        if imufile is not None and os.path.isfile(imufile):
            with imu.handler(reset_timestamps=False) as imu_handler:
                imu_handler.load_file(imufile)
                imu_handler.cut_timewindow(start=imu_handler.start + rec_burnin*1e3,
                                           stop=imu_handler.stop - rec_burnout*1e3)
                imu_dur = imu_handler.duration * 1e-6  # seconds
                try:
                    sampling = int(imu_handler.num_events / imu_dur)
                except ZeroDivisionError:
                    sampling = imu_handler.sampling
                if sampling < target_sampling:
                    files_with_holes.append(file)
        else:
            raise Exception(f'The recording {file} does not have an IMU file and it is required for finding holes '
                            'in the recording.')
        pbar.update(1)
    pbar.close()
    return files_with_holes


def detect_holes_info(all_files: list,
                      rec_burnin: float = 0, rec_burnout: float = 0,
                      samples4hole: int = 2, maxnumholes: int = 5, target_sampling: float = 999,
                      verbose: bool = True, show_pbar: bool = False) -> (list, list, list):
    pbar = tqdm(all_files, total=len(all_files), desc='Detecting holes in recordings', file=sys.stdout,
                disable=not show_pbar)
    files_with_holes = []
    max_holes_dur = []
    num_holes = []
    for file in all_files:
        imufile = recording_imufile(read_json(file))
        if imufile is not None and os.path.isfile(imufile):
            with imu.handler(reset_timestamps=False) as imu_handler:
                imu_handler.load_file(imufile)
                imu_handler.cut_timewindow(start=imu_handler.start + rec_burnin*1e3,
                                           stop=imu_handler.stop - rec_burnout*1e3)
                imu_dur = imu_handler.duration * 1e-6  # seconds
                try:
                    sampling = int(imu_handler.num_events / imu_dur)
                except ZeroDivisionError:
                    sampling = imu_handler.sampling
                all_dts = np.diff(imu_handler.ts)
                max_dt = np.max(all_dts)
                median_dt = int(np.median(all_dts))
                n_holes = np.sum(all_dts >= (samples4hole - 1) * median_dt)
                # n_holes = np.sum(all_dts >= samples4hole * median_dt)
                if max_dt > samples4hole * median_dt or n_holes >= maxnumholes or sampling < target_sampling:
                    if verbose:
                        print(f' --> File: {file}\n'
                              f'   - Sampling rate of IMU device: {sampling if imu_dur > 0 else "..."} Hz\n'
                              f'   - Maximum IMU time-step is: {int(round(max_dt * 1e-3))} ms\n'
                              f'   - Number of holes of at least {samples4hole * int(median_dt*1e-3)} ms: {n_holes}\n'
                              )
                    files_with_holes.append(file)
                    max_holes_dur.append(max_dt * 1e-3)  # in ms
                    num_holes.append(n_holes)
        else:
            raise Exception(f'The recording {file} does not have an IMU file and it is required for finding holes '
                            'in the recording.')
        pbar.update(1)
    pbar.close()
    return files_with_holes, max_holes_dur, num_holes


# ---------------------------------------------------- 4) No Data ----------------------------------------------------

def detect_delay(all_files: list,
                 min_fem_delay: int = 5000,
                 verbose: bool = True, show_pbar: bool = False) -> list or None:
    warnings.filterwarnings('error')
    pbar = tqdm(all_files, total=len(all_files), desc='Detecting FEM delay in recordings', file=sys.stdout,
                disable=not show_pbar)
    files_low_delay = []
    for file in all_files:
        imufile = recording_imufile(read_json(file))
        if imufile is not None and os.path.isfile(imufile):
            with imu.handler(reset_timestamps=False) as imu_handler:
                imu_handler.load_file(imufile)
                # Detect FEM interval from IMU data
                imu_handler.find_fem(cut_init=100 * 1e3)
                fem_start = imu_handler.fem_finder.fem_start
                rec_start = imu_handler.start
            delay = fem_start - rec_start
            if delay <= min_fem_delay*1e3:
                if verbose:
                    print(f' --> File: {file}\n'
                          f'   - Delay between recording and FEM start is: {int(round(delay*1e-3))} ms\n')
                    print()
                files_low_delay.append(file)
        else:
            raise Exception(f'The recording {file} does not have an IMU file and it is required for finding FEM delay '
                            'in the recording.')
        pbar.update(1)
    pbar.close()
    return files_low_delay
