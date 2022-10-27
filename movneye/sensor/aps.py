import os
import cv2
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from movneye.utils.add2cv import load_calibration


class handler:

    def __init__(self, frames: np.ndarray = None, timestamps: np.ndarray = None, reset_timestamps: bool = False):
        """
        This class helps handling frame-based data. A set of methods for transforming, taking info and visualizing
        single frame or video is available.
        Inputs:
            frames (np.ndarray): Video in the form of a N*Nx*Ny np.ndarray where N is the total number of frames and the
                other dimensions match the shape of the pixel array.
            timestamps (np.ndarray): Vector of N elements representing the timestamp of each frame.
        Protected Attributes:
            N_x, N_y (int, int): Width and height of the camera pixel array.
            data (np.ndarray): Array of frames with shape N*Nx*Ny.
            ts (np.ndarray): Vector of timestamps with length N.
            sampling (int): Frame rate of the device in Hz.
            camera_mtx (np.ndarray): 2D array with shape (N_y,N_x) keeping the intrinsic parameters of the camera.
            distortion_coeffs (np.ndarray): 1D array keeping the distortion coefficients given by lens' optical effects.
        """
        self.reset_timestamps = reset_timestamps
        self._N_y, self._N_x = None, None
        self._data = frames
        if frames is not None:
            self._N_y, self._N_x = frames[0].shape[:2]
            self._data = ((frames - frames.min()) / (frames.max() - frames.min()) * 255).astype(np.uint8)
        self._ts = timestamps
        if timestamps is not None:
            self._sampling = self.compute_sampling_rate()
            if self.reset_timestamps:
                self.timereset()
            if frames is not None:
                assert len(self._ts) == self.num_frames,\
                    'The number of timestamps must be equal to the number of frames.'

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, frames):
        self._data = ((frames - frames.min()) / (frames.max() - frames.min()) * 255).astype(np.uint8)
        self._N_y, self._N_x = self._data[0].shape[:2]

    @property
    def ts(self):
        return self._ts

    @ts.setter
    def ts(self, timestamps):
        self._ts = timestamps
        self._sampling = self.compute_sampling_rate()
        if self.reset_timestamps:
            self.timereset()
        if self._data is not None:
            assert len(self._ts) == self.num_frames, \
                'The number of timestamps must be equal to the number of frames.'

    @property
    def shape_pixel_array(self):
        return self._N_y, self._N_x

    @property
    def num_frames(self):
        return self._data.shape[0]

    @property
    def start(self):  # in micro-seconds (us)
        if self._ts is None:
            return None
        return self._ts[0]

    @property
    def stop(self):  # in micro-seconds (us)
        if self._ts is None:
            return None
        return self._ts[-1]

    @property
    def dt(self):  # in micro-seconds (us)
        return self.compute_dt()

    @property
    def sampling(self):  # in Hz
        if self._ts is None:
            return None
        return self._sampling

    @property
    def duration(self):  # in micro-seconds (us)
        # Underestimated duration of recording (40 Hz is the maximum frame rate)
        if self._ts is None:
            return None
        return self._ts[-1] - self._ts[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    # -------------------------------------------------- Data Loading --------------------------------------------------
    def load_file(self, file: str):
        """Load (and update) data from a TXT/CSV or AEDAT4 file.
        Args:
            file (str, required): full path of the file where frames were recorded.
        """

        assert isinstance(file, str) and file is not None,\
            'You must specify a string as full path for loading event-based data.'
        assert os.path.isfile(file), 'The given path is not a file.'
        extension = os.path.splitext(file)[-1]
        if extension in ['.txt', '.csv']:
            _, self._N_x, self._N_y = read_csv(file, delim_whitespace=True, header=None, nrows=1,
                                               dtype={'text': str, 'width': int, 'height': int},
                                               names=['text', 'width', 'height']).values[0]
            n_skip = 2
            frames, ts = [], []
            while True:
                try:
                    ts.append(int(read_csv(file, delim_whitespace=True, header=None, nrows=1, skiprows=n_skip,
                                           dtype={'ts': int}, names=['ts']).values[0]))
                    frames.append(np.loadtxt(file, dtype=int, skiprows=n_skip+1, max_rows=self._N_y))
                    n_skip += self._N_y + 1
                except:
                    break
            frames = np.array(frames)
            if frames is not None:
                self._data = ((frames - frames.min()) / (frames.max() - frames.min()) * 255).astype(np.uint8)
            self._ts = np.array(ts)
        elif extension == '.avi':
            file_ts = os.path.splitext(file)[0] + 'ts.csv'
            if not os.path.isfile(file_ts):
                file_ts = os.path.splitext(file)[0] + 'ts.txt'
            _, self._N_x, self._N_y = read_csv(file_ts, delim_whitespace=True, header=None, nrows=1,
                                               dtype={'text': str, 'width': int, 'height': int},
                                               names=['text', 'width', 'height']).values[0]
            self._ts = read_csv(file_ts, delim_whitespace=True, header=None, skiprows=2,
                                dtype={'ts': int}, names=['ts']).values.ravel()
            cap = cv2.VideoCapture(file)
            ret, frame = cap.read()
            frames = np.copy(frame[None, :, :, 0])
            while ret:
                frames = np.vstack((frames, frame[None, :, :, 0]))
                ret, frame = cap.read()
            cap.release()
            frames = frames[1:]
            if frames is not None:
                self._data = ((frames - frames.min()) / (frames.max() - frames.min()) * 255).astype(np.uint8)
        elif extension == '.aedat4':
            from movneye.sensor.readaedat import AedatAPSReader
            aedat_loader = AedatAPSReader(file)
            self._data = aedat_loader.frames
            self._N_y, self._N_x = self._data.shape[1:]
            self._ts = aedat_loader.frames_timestamps
        else:
            raise ValueError('Type of file not supported. It must be a .txt/.csv or .avi or .aedat4 file.')
        self._sampling = self.compute_sampling_rate()

    # ------------------------------------------------ Basic Utilities -------------------------------------------------
    def timereset(self, reference_timestamp: int or None = None):
        """Given an array of DVS events, with N rows (number of events) and M columns where the first one represents the
        timestamps, this function returns the same array but resetting all timestamps according to the first event or
        to a given reference timestamp (if reference_timestamp is not None).
        Args:
            reference_timestamp (int, optional): the timestamp of a given reference phenomenon by which to reset all the
                timestamps of events.
        """
        if self._ts is None:
            pass
        else:
            if reference_timestamp is None:
                reference_timestamp = self._ts[0]
            self._ts -= reference_timestamp

    def compute_dt(self):
        if self._ts is None:
            return None
        return int(np.median(np.diff(self._ts)))

    def compute_sampling_rate(self):
        if self._ts is None:
            return None
        return 1 / (self.compute_dt() * 1e-6)

    def select_frame(self, frame_num: int = 0):
        return self._data[frame_num]

    # ---------------------------------------------- Optics undistortion -----------------------------------------------
    def _undistortion_maps(self, camera_mtx: np.ndarray, distortion_coeffs: np.ndarray) -> (np.ndarray, np.ndarray):
        """Compute the undistortion maps from camera calibration parameters."""
        map1, map2 = cv2.initUndistortRectifyMap(camera_mtx, distortion_coeffs, R=np.eye(3, dtype=int),
                                                 newCameraMatrix=camera_mtx, size=(self._N_x, self._N_y),
                                                 m1type=cv2.CV_32FC1)
        return map1, map2

    def _remap(self, map_1: np.ndarray, map_2: np.ndarray, interpolate: bool = False):
        """Remap frames for removing radial and tangential lens distortion. It takes as input the undistortion maps and
        updates frames. We remove info falling on those pixels for which there is no correspondence in the
        original pixel array."""
        if interpolate:
            dst_video = np.zeros_like(self._data)
            for k, frame in enumerate(self._data):
                dst_video[k] = cv2.remap(frame, map_1, map_2, interpolation=cv2.INTER_LINEAR)
            self._data = dst_video
        else:
            self._data = self._data[:, np.round(map_2).astype(int), np.round(map_1).astype(int)]

    def undistort(self, calib_file: str, model: str = 'DAVIS346', serial: str = '00000336',
                  interpolate: bool = False):
        """Compute undistortion maps and remap frames accordingly in order to correct lens effects. Note that this
        method should be called BEFORE cropping!!
        Args:
             calib_file (str, required): the full path of the xml file where info from calibration are stored
                in open-CV format (e.g. from calibration procedure on the DV software).
             model (str): the model of the neuromorphic sensor.
             serial (str): the serial number of the sensor.
             interpolate (bool, optional): whether to smoothen the result through interpolation.
        """
        camera_mtx, dist_coeffs = load_calibration(calib_file, model=model, serial=serial)
        self._remap(*self._undistortion_maps(camera_mtx, dist_coeffs), interpolate=interpolate)

    # ---------------------------------------------------- Cut data ----------------------------------------------------
    def crop(self, shape: (int, int)):
        """Given an array of APS frames (each frame having number of rows y_dim and number of columns x_dim), this
        method returns the same array but with each frame having the given shape.
        Args:
            shape (tuple, required): number of pixels along the y and x dimensions of the new pixel array.
        """
        if shape[1] > self._N_x or shape[0] > self._N_y:
            raise ValueError('Cannot crop data with the given shape.')
        x_start, y_start = (self._N_x - shape[1]) // 2, (self._N_y - shape[0]) // 2
        x_end, y_end = x_start + shape[1], y_start + shape[0]
        self._data = self._data[:, y_start:y_end, x_start:x_end]
        self._N_y, self._N_x = shape

    def crop_square(self, dim: int):
        """Given an array of APS frames (each frame having number of rows y_dim and number of columns x_dim), this
        method returns the same array but with each frame having squared (cut) size (number of rows and columns equals
        to dim).
        Args:
            dim (int, required): number of pixels along the x and y dimensions of the new squared pixel array.
        """
        if dim > self._N_x or dim > self._N_y or dim < 0:
            raise ValueError('Cannot crop data with the given dimension.')
        x_start, y_start = (self._N_x - dim) // 2, (self._N_y - dim) // 2
        x_end, y_end = x_start + dim, y_start + dim
        self._data = self._data[:, y_start:y_end, x_start:x_end]
        self._N_x, self._N_y = dim, dim

    def crop_region(self, start: (int, int), end: (int, int)):
        """Given a region of interest of the pixel array and an array of APS frames (each frame having number of rows
        y_dim and number of columns x_dim), this method returns the same array but cutting off info outside of the
        given region.
        Args:
             start (tuple, required): starting point (x, y) of the selected region.
             end (tuple, required): ending point (x, y) of the selected region.
        """
        x_start, x_end = start[0], end[0]
        # Note: when defining the y coordinate of the starting point, consider that y=0 is on top,
        # while y=N_y-1 is at the bottom. For this reason we do the following:
        y_start, y_end = self._N_y - end[1], self._N_y - start[1]
        self._data = self._data[:, y_start:y_end, x_start:x_end]
        self._N_x, self._N_y = (x_end - x_start), (y_end - y_start)

    def cut_duration(self, duration: float, refresh: float = 40.):
        """This method cut the last frames in the APS video according to the given duration. The video is supposed to
        start from 0 timestamp and to have a fixed refresh rate (given as input).
        Args:
            duration (float, required): wanted duration (in us) for the recorded frames.
            refresh (float, optional): refresh rate of the sensor useful for deriving the timestamp of each frame. If
                the correct timestamps of the frames were provided, this argument is ignored.
        """
        if self._ts is None:
            dt = 10e6 / refresh
            timestamps = np.linspace(0, dt * self.num_frames, self.num_frames, dtype=int)
            self._data = self._data[timestamps <= duration, ...]
        else:
            mask = ((self._ts - self._ts[0]) <= duration)
            self._data = self._data[mask, ...]
            self._ts = self._ts[mask]
            self._sampling = self.compute_sampling_rate()

    def cut_timewindow(self, start: float, stop: float, refresh: float = 40):
        """Given the relative timestamps of the APS video, this method cuts all frames having timestamps outside the
         given time window defined by start and stop parameters."""
        if self._ts is None:
            dt = 10e6 / refresh
            timestamps = np.linspace(0, dt * self.num_frames, self.num_frames, dtype=int)
            self._data = self._data[timestamps <= (stop - start), ...]
        else:
            mask = np.logical_and(self._ts >= start, self._ts <= stop)
            self._data = self._data[mask, ...]
            self._ts = self._ts[mask]
            self._sampling = self.compute_sampling_rate()

    # --------------------------------------------------- Show frames --------------------------------------------------
    def show_frame(self, frame_num: int or str = 0, title: str or None = None,
                   show: bool = True, figsize: (int, int) = (5, 4)):
        """Show a single frame.
        Args:
            frame_num (int, optional): the frame to show.
            title (str, optional): title of the plot.
            show (bool, optional): whether to show or not the window.
            figsize (tuple, optional): size of the window.
        """
        if not title:
            title = 'APS frame ' + str(frame_num)
        if frame_num == 'mean':
            img = np.round(np.mean(self._data, axis=0)).astype(np.uint8)
        elif isinstance(frame_num, int):
            img = self._data[frame_num]
        else:
            raise ValueError('Parameter not understood, num_frame should be an integer or "mean".')
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        if isinstance(show, bool):
            if show:
                plt.show()
            else:
                pass
        else:
            plt.show(block=False)
            plt.pause(show)
            plt.close()

    def show_video(self, refresh: float = 40, title: str = 'APS video', position: (int, int) = (0, 0), zoom: float = 1):
        """Show the video from an array of APS frames.
        Args:
            refresh (int, optional): refresh rate of the video.
            title (str, optional): title of the plot.
            position (tuple, optional): position of the cv2 window.
            zoom (float, optional): how much to zoom the window.
        """
        dt = int(round(1e3 / refresh)) if self._sampling is None else int(round(1e3 / self._sampling))
        # Visualise video
        cv2.namedWindow(title)
        cv2.moveWindow(title, *position)
        plot_shape = (int(self._N_x * zoom), int(self._N_y * zoom))
        for frame in self._data:
            frame = cv2.resize(frame, plot_shape, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, frame)
            cv2.waitKey(dt)
        cv2.destroyAllWindows()
