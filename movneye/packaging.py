import os
import numpy as np
from movneye.sensor import dvs, imu
from movneye.functional.read import sensor_info


def read_imu(imu_file: str, rows4header: int = 2):
    imu_events = None
    if imu_file is not None and os.path.isfile(imu_file):
        with imu.handler(reset_timestamps=False) as imu_handler:
            imu_handler.load_file(imu_file, rows4header=rows4header)
            imu_events = imu_handler.data
    return imu_events


def find_fem_period(imufile: str,
                    burnin: int = 0, burnout: int = 0) -> (int, int):
    # Return first and last timestamps of FEM sequence
    with imu.handler(reset_timestamps=False) as imu_handler:
        imu_handler.load_file(imufile)
        imu_handler.cut_timewindow(start=imu_handler.start + burnin * 1e3,
                                   stop=imu_handler.stop - burnout * 1e3)
        # Detect FEM interval from IMU data
        imu_handler.find_fem(cut_init=(0 if burnin != 0 else 100 * 1e3))
        fem_start = imu_handler.fem_finder.fem_start
        fem_stop = imu_handler.fem_finder.fem_stop
    return fem_start, fem_stop


def return_preprocessed_events(dvs_events: np.ndarray,
                               fem_start: float = None, fem_stop: float = None,
                               calib_file: str = None, roi: [[int, int], [int, int]] = None,
                               burnin: float = 0, burnout: float = 0,
                               refractory: float = None, hotpix_space_window: int = None):
    with dvs.handler(data=dvs_events, reset_timestamps=False) as dvs_handler:
        dvs_handler.cut_timewindow(start=dvs_handler.start + burnin * 1e3,
                                   stop=dvs_handler.stop - burnout * 1e3)
        if calib_file is not None:
            _, model, serial = sensor_info()
            dvs_handler.undistort(calib_file, model=model, serial=serial)
        if roi is not None:
            dvs_handler.crop_region(start=roi[0], end=roi[1])
        if refractory is not None:
            dvs_handler.refractory_filter(refractory=int(refractory * 1e3))
        if hotpix_space_window is not None:
            dvs_handler.hot_pixels_filter(space_window=hotpix_space_window)
        # Take only DVS events in FEM interval
        if fem_start and fem_stop:
            dvs_handler.cut_timewindow(start=fem_start, stop=fem_stop)
            dvs_handler.timereset(reference_timestamp=fem_start)
        else:
            dvs_handler.timereset()
        dvs_events = dvs_handler.data
    return dvs_events
