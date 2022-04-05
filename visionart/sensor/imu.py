import os
import numpy as np
from pandas import read_csv
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class handler:

    def __init__(self, data: np.ndarray = None, reset_timestamps: bool = False):
        """
        This class helps handling events from the IMU device. A set of methods for taking info, visualizing and finding
        motion intervals is available.
        Inputs:
            data (np.ndarray): IMU info in the form of a N*8 np.ndarray where N is the total number of events and the
                other dimensions stand for: timestamp (int,us), X linear acceleration (float,ms/s), Y acc. (float,ms/s),
                Z acc. (float,ms/s), X angular velocity (float,°/s), Y ang. vel. (float,°/s), Z ang. vel. (float,°/s),
                temperature (float,°C).
        Protected Attributes:
            data (np.ndarray): Array of imu info with shape N*8.
            sampling (int): Sampling rate of the device in Hz.
            fem_finder (object): Instance of the class fem_finder useful for finding the FEM interval from
                gyroscope data (angular speeds). It is not empty only after a call to the find_fem() method.
        """
        self.reset_timestamps = reset_timestamps
        self._sampling = None
        self._data = data
        if data is not None:
            self._sampling = 1 / (self.compute_dt() * 1e-6)
            if self.reset_timestamps:
                self.timereset()
        self._fem_finder = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._sampling = 1 / (self.compute_dt() * 1e-6)
        if self.reset_timestamps:
            self.timereset()

    @property
    def ts(self):  # in micro-seconds (us)
        return self._data[:, 0]

    @property
    def x_acc(self):
        return self._data[:, 1]

    @property
    def y_acc(self):
        return self._data[:, 2]

    @property
    def z_acc(self):
        return self._data[:, 3]

    @property
    def linear_acceleration(self):
        return np.sqrt((self._data[:, 1:4] ** 2).sum(axis=1))

    @property
    def x_gyr(self):
        return self._data[:, 4]

    @property
    def y_gyr(self):
        return self._data[:, 5]

    @property
    def z_gyr(self):
        return self._data[:, 6]

    @property
    def temperature(self):
        return self._data[:, 7]

    @property
    def angular_speed(self):
        return np.sqrt((self._data[:, 4:7] ** 2).sum(axis=1))

    @property
    def num_events(self):
        return self._data.shape[0]

    @property
    def start(self):  # in micro-seconds (us)
        return self.ts[0]

    @property
    def stop(self):  # in micro-seconds (us)
        return self.ts[-1]

    @property
    def duration(self):  # in micro-seconds (us)
        return self.stop - self.start

    @property
    def dt(self):  # in micro-seconds (us)
        return self.compute_dt()

    @property
    def sampling(self):  # in Hz
        return self._sampling

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    # -------------------------------------------------- Data Loading --------------------------------------------------
    def load_file(self, file: str,
                  rows4header: int = 2):
        """Load (and update) data from a TXT/CSV or AEDAT4 or NPY file.
        Args:
            file (str): full path of the file where events were recorded.
            rows4header (int): number of rows for the header in the CSV file.
        """
        assert isinstance(file, str) and file is not None,\
            'You must specify a string as full path for loading event-based data.'
        assert os.path.isfile(file), 'The given path is not a file.\n' + file
        extension = os.path.splitext(file)[-1]
        if extension in ['.txt', '.csv']:
            self._data = read_csv(file, delim_whitespace=True, header=None,
                                  engine='c', skiprows=rows4header, nrows=None, memory_map=True,
                                  names=['t', 'x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'temp'],
                                  dtype={'t': np.int64,
                                         'x_acc': np.float64, 'y_acc': np.float64, 'z_acc': np.float64,
                                         'x_gyr': np.float64, 'y_gyr': np.float64, 'z_gyr': np.float64,
                                         'temp': np.float64}
                                  ).values
            # self._data = np.genfromtxt(file, skip_header=rows4header, delimiter=' ', dtype=np.float64)
        elif extension == '.aedat4':
            from visionart.sensor.readaedat import AedatIMUReader
            self._data = AedatIMUReader(file).events_Nx8()
        elif extension == '.npy':
            self._data = np.load(file)
        else:
            raise ValueError('Type of file not supported. It must be a .txt/.csv or .npy or .aedat4 file.')
        self._sampling = 1 / (self.compute_dt() * 1e-6)
        if self.reset_timestamps:
            self.timereset()

    # ------------------------------------------------ Basic Utilities -------------------------------------------------
    def timereset(self, data: np.ndarray or None = None, reference_timestamp: int or None = None) -> np.ndarray:
        """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
        timestamps, this function returns the same array but resetting all timestamps according to the first event or
        to a given reference timestamp (if reference_timestamp is not None).
        Args:
            data (np.ndarray, required): events array with N rows (number of events) and M columns where the first one
                represents the timestamps.
            reference_timestamp (int, optional): the timestamp of a given reference phenomenon by which to reset all the
                timestamps of events.
        Returns:
            (np.ndarray): events array as input data but resetting all timestamps according to the first one (so that
                the timestamp of the first event is 0 if reference_timestamp is None, else it depends by such value).
        """
        if data is not None:
            data_reset = np.copy(data)
            if reference_timestamp is None:
                reference_timestamp = data[0, 0]
            data_reset[:, 0] -= reference_timestamp
            return data_reset
        else:
            if reference_timestamp is None:
                reference_timestamp = self._data[0, 0]
            self._data[:, 0] -= reference_timestamp

    def compute_dt(self):
        if not self.ts.size:
            return 1000
        return int(np.median(np.diff(self.ts)))

    def compute_dt_max(self):
        if not self.ts.size:
            return 100000
        return np.max(np.diff(self.ts))

    def cut_timewindow(self, start: float, stop: float):
        """This function cuts out all imu data outside the given time period.
        Args:
            start (float, required): first moment (in us) for the recorded events.
            stop (float, required): last moment (in us) for the recorded events.
        """
        idx_keep = np.where(np.logical_and(self._data[:, 0] >= start,
                                           self._data[:, 0] <= stop))[0]
        self._data = self._data[idx_keep]
        if self.reset_timestamps:
            self.timereset()

    # ---------------------------------------------- Find motion from IMU ----------------------------------------------
    def find_fem(self, cut_init: float = 50 * 1e3, plateau_interval: float = 200 * 1e3, smoothing_window: int = 6,
                 min_step_interval: float = 10 * 1e3, min_step_speed: float = 10.):
        self._fem_finder = fem_finder(imu=self._data, sampling_rate=self._sampling, cut_init=cut_init,
                                      plateau_interval=plateau_interval, smoothing_window=smoothing_window,
                                      min_step_interval=min_step_interval, min_step_speed=min_step_speed)

    def plot_fem_interval(self, show: bool = True, title='FEM activity from IMU data'):
        if self._fem_finder is None:
            self.find_fem()
        self._fem_finder.plot_fem_interval(show=show, title=title)

    @property
    def fem_finder(self):
        if self._fem_finder is None:
            self.find_fem()
        return self._fem_finder

    @property
    def fem_start(self):
        if self._fem_finder is None:
            self.find_fem()
        return self._fem_finder.fem_start

    @property
    def fem_stop(self):
        if self._fem_finder is None:
            self.find_fem()
        return self._fem_finder.fem_stop

    @property
    def fem_steps_start(self):
        if self._fem_finder is None:
            self.find_fem()
        return self._fem_finder.steps_start

    @property
    def fem_steps_stop(self):
        if self._fem_finder is None:
            self.find_fem()
        return self._fem_finder.steps_stop

    @property
    def fem_speed(self):
        if self._fem_finder is None:
            self.find_fem()
        return self._fem_finder.fem_speed

    @property
    def fem_ts(self):
        if self._fem_finder is None:
            self.find_fem()
        return self._fem_finder.fem_timestamps


class fem_finder:

    def __init__(self, imu: np.ndarray or str, sampling_rate: float = 1000, cut_init: float = 100 * 1e3,
                 plateau_interval: float = 200 * 1e3, smoothing_window: int = 6,
                 min_step_interval: float = 10 * 1e3, min_step_speed: float = 10.):
        """
        Arguments:
        :param imu (required): numpy array (or full path to npy file) with IMU info having shape (N, 8) where N is the
            total number of IMU events (samples) and the 8 columns represent the timestamp of the event, the
            accelerations on the X, Y, and Z axes (i.e. tilt, pan and roll accelerations respectively), the angular
            speed on the X, Y and Z axes (i.e. tilt, pan and roll velocity respectively) and the temperature in Celsius.
        :param sampling_rate (optional, default=1 kHz): the rate at which IMU info was acquired.
        :param cut_init (optional, default=100 ms): the time interval (in us) to remove from IMU recordings, both at the
            beginning and at the end. During such intervals, the recording should only contain noisy and useless
            information, such as motor initialization at the beginning of the recording or movement back to the origin
            after the FEM sequence.
        :param plateau_interval (optional, default=200 ms): time interval (in us), at the end of the recording, during
            which the angular speed is at its plateau value, i.e. no big oscillation due to FEM but only noise.
        :param smoothing_window (optional, default=6 samples): the standard deviation of the Gaussian filter used for
            smoothing data through convolution.
        :param min_step_interval (optional, default=10 ms): minimum time interval (in us) between consecutive steps of
            the FEM sequence.
        :param min_step_speed (optional, default=10 deg/s): minimum height of the angular speed (in deg/s) during a
            single step of the FEM sequence.

        Main computations:
        - the timestamps of start and stop (i.e. first and last moments) of the whole FEM sequence (as 2 integers:
          fem_start, fem_stop).
        - the timestamps of start and stop (i.e. first and last moments) of each FEM step (as 2 numpy arrays:
          steps_start, steps_stop).
        In both cases, timestamps are in micro-seconds but the precision is determined by the sampling rate of the IMU
        (which should be 1 kHz, thus 1 ms precision for the identification of these instants).
        """

        # Load imu data (if necessary)
        if isinstance(imu, str):
            imu = np.load(imu)
        elif isinstance(imu, np.ndarray):
            pass
        else:
            raise TypeError('The imu parameter must be a numpy array or a string with full path to npy file.')

        # Sampling rate of the IMU
        self.sampling_rate = sampling_rate  # Hz
        # Parameters for finding FEM interval
        self.smoothing_window = smoothing_window
        self.plateau_interval = plateau_interval
        # Parameters for finding single steps of FEM
        self.min_step_interval = min_step_interval
        self.min_step_speed = min_step_speed  # or 8 for seed=2 ?

        # Take relevant IMU data
        self.timestamps = imu[:, 0].astype(int)
        # self.acc_components = imu[:, 1:4]
        # self.speed_components = imu[:, 4:7]
        self.speeds = np.sqrt((imu[:, 4:6] ** 2).sum(axis=1))
        # Note: in order to compute the modulus of the angular speed, we only take the X (=TILT) and Y (=PAN) components
        # from the gyroscope since the movements induced have no ROLL (Z component), meaning that this information is
        # not useful (it brings only noise) for identifying FEMs in IMU data.

        # Cut off some samples at the beginning and end of the recording
        cut_samples = int(round(cut_init * 1e-6 * self.sampling_rate))
        if cut_samples > 0:
            self.timestamps = self.timestamps[cut_samples: -cut_samples]
            self.speeds = self.speeds[cut_samples: -cut_samples]

        # Find FEM interval
        first_fem_sample, last_fem_sample = self.find_fem_interval()
        self.fem_start = self.timestamps[first_fem_sample]
        self.fem_stop = self.timestamps[last_fem_sample]

        # Cut IMU data to exclude all samples outside the FEM interval
        self.fem_timestamps, self.fem_speed = self.select_fem_interval()

        # Find interval of single steps in FEM
        first_steps_samples, last_steps_samples = self.find_fem_steps()
        self.steps_start = self.fem_timestamps[first_steps_samples]
        self.steps_stop = self.fem_timestamps[last_steps_samples]

        # # Reset sample values to original IMU data
        # first_fem_sample += cut_samples
        # last_fem_sample += cut_samples
        # first_steps_samples += first_fem_sample
        # last_steps_samples += first_fem_sample

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _gaussian_smoothing(self, window: int = None):
        """Smooth IMU data through convolution with a Gaussian filter."""
        if window is None:
            window = self.smoothing_window
        gauss = np.exp(-(np.arange(-4 * window, 4 * window) / window) ** 2 / 2)
        smooth_gauss = np.convolve(self.speeds, gauss, mode='same')
        return (smooth_gauss - smooth_gauss.min()) / (smooth_gauss.max() - smooth_gauss.min()) *\
               (self.speeds.max() - self.speeds.min()) + self.speeds.min()

    def find_fem_interval(self, plateau_interval: float = None, smoothing_window: int = None):
        """Identify, from IMU data, the time interval in which FEM movements occur. We use a threshold level for the
        angular speed above which speed oscillations are considered to be originating from FEMs. Such threshold is
        defined as the mean+4std of the final plateau in the angular speed (a time interval corresponding to the last
        200 ms of recording are considered as default).
        This function returns the indices of first and last IMU samples during which FEMs occur."""
        if plateau_interval is None:
            plateau_interval = self.plateau_interval
        speed_smooth = self._gaussian_smoothing(window=smoothing_window)
        num_samples_plateau = int(round(plateau_interval * 1e-6 * self.sampling_rate))
        motion_threshold = float(speed_smooth[-num_samples_plateau:].mean() +
                                 4 * speed_smooth[-num_samples_plateau:].std())
        first_sample = np.argmax(speed_smooth > motion_threshold)
        last_sample = len(speed_smooth) - np.argmax(speed_smooth[first_sample:][::-1] > motion_threshold)
        if last_sample >= self.speeds.shape[0]:
            last_sample = self.speeds.shape[0] - 1
        return first_sample, last_sample

    def select_fem_interval(self):
        """Select IMU data simultaneous to FEM execution."""
        fem_mask = np.logical_and(self.timestamps >= self.fem_start,
                                  self.timestamps <= self.fem_stop)
        # fem_mask = range(self._first_fem_sample, self._last_fem_sample)
        return self.timestamps[fem_mask], self.speeds[fem_mask]

    def find_fem_peaks(self, min_step_interval: float = None, min_step_speed: float = None):
        min_step_speed = self.min_step_speed if min_step_speed is None else min_step_speed
        min_step_interval = self.min_step_interval if min_step_interval is None else min_step_interval
        min_step_samples = int(round(min_step_interval * 1e-6 * self.sampling_rate))
        return find_peaks(self.fem_speed, height=min_step_speed, distance=min_step_samples)[0]

    def find_fem_steps(self, min_step_interval: float = None, min_step_speed: float = None):
        """Identify, from IMU data during FEM, the time interval of each single FEM step. We first find the most
         significant peaks of the angular speed during FEM. Then we identify as the starting moment of a step the
         first sample, previous to a peak, from which the speed starts raising. The last sample of each step equals
         to the first sample of the next step."""
        peaks_speed = self.find_fem_peaks(min_step_interval=min_step_interval, min_step_speed=min_step_speed)
        speed_derivative = np.diff(self.fem_speed)
        first_samples = np.array([peak - np.argmax(speed_derivative[:peak][::-1] < 0) for peak in peaks_speed])
        last_samples = np.append(first_samples[1:] - 1, -1)
        return first_samples, last_samples

    def cut_dvs_data_in_fem_interval(self, dvs_data: np.ndarray or str, return_steps_start: bool = False):
        """Cut off DVS events falling outside the time interval in which the neuromorphic sensor is subject to FEMs,
        and reset the timestamps of events according to the timestamp at which FEMs start.
        The method accepts as argument the np.array of full path of DVS events (and a bool parameter determining whether
        to return new timestamps of FEM-steps' start in DVS data)."""
        if isinstance(dvs_data, str):
            dvs_data = np.load(dvs_data)
        elif isinstance(dvs_data, np.ndarray):
            pass
        else:
            raise TypeError('The dvs_data parameter must be a numpy array or a string with full path to npy file.')
        cut_data = dvs_data[np.logical_and(dvs_data[:, 0] >= self.fem_start,
                                           dvs_data[:, 0] <= self.fem_stop)]
        cut_data[:, 0] -= self.fem_start
        if return_steps_start:
            return cut_data, self.steps_start - self.fem_start
        return cut_data

    def plot_fem_interval(self, show: bool = True, title='FEM activity from IMU data'):
        """Some informative and self-explanatory plots."""
        peaks = self.find_fem_peaks()
        upper_speed = self.speeds.max() * 1.2
        plt.figure(figsize=(16, 10))
        plt.suptitle(title)
        plt.subplot(211)
        plt.title('Angular speed during whole recording')
        plt.xlabel('Time (ms)')
        plt.ylabel('Angular Speed (deg/s)')
        plt.ylim(0, upper_speed)
        plt.plot((self.timestamps - self.timestamps[0]) * 1e-3, self.speeds)
        plt.plot((self.timestamps - self.timestamps[0]) * 1e-3, self._gaussian_smoothing())
        plt.axvline(x=(self.fem_start - self.timestamps[0]) * 1e-3, color='g', lw=2)
        plt.axvline(x=(self.fem_stop - self.timestamps[0]) * 1e-3, color='r', lw=2)
        plt.subplot(212)
        plt.title('Angular speed during FEM')
        plt.xlabel('Time (ms)')
        plt.ylabel('Angular Speed (deg/s)')
        plt.ylim(0, upper_speed)
        plt.plot((self.fem_timestamps - self.fem_timestamps[0]) * 1e-3, self.fem_speed)
        plt.plot((self.fem_timestamps[peaks] - self.fem_timestamps[0]) * 1e-3, self.fem_speed[peaks], '.r')
        plt.text(x=len(self.fem_speed), y=upper_speed/2, s=str(len(peaks)) + ' steps', fontsize=10)
        for start_step in self.steps_start:
            plt.axvline(x=(start_step - self.fem_timestamps[0]) * 1e-3, color='g', lw=1.2, ls='--')
        plt.axvline(x=(self.steps_stop[-1] - self.fem_timestamps[0]) * 1e-3, color='r', lw=1.2, ls='--')
        if show:
            plt.show()
