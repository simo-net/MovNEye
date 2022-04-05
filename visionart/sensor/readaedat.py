import os
import aedat
import numpy as np

dvs_dtype = np.dtype({'names': ['t', 'x', 'y', 'p'],
                      'formats': [np.uint64, np.uint16, np.uint16, np.bool]})
imu_dtype = np.dtype({'names': ['t', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
                                'magnet_x', 'magnet_y', 'magnet_z', 'temperature'],
                      'formats': [np.uint64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,
                                  np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]})


class AedatDVSReader(object):
    """
    BaseReader for .aedat4 format.
    We write a base iterator returning each event packet as a numpy array with the 'events_dtype' format, i.e. with
    names (t, x, y, p) having types (int64, int16, int16, bool) respectively.
    The full array of all events in the recording (i.e. all packets together) is returned by the method 'events_info()'.
    If you want a full array with shape (N,4) [where N is the total number of events], call the method 'events_Nx4()'.
    """

    def __init__(self, file: str, reset_timestamps: bool = False):
        assert os.path.exists(file), "The given file does not exist."
        file_extension = os.path.splitext(file)[1]
        assert (file_extension in ['.aedat', '.aedat4']), "The given file must be either a .aedat or .aedat4 file."

        self.decoder = aedat.Decoder(file)
        self._t0 = None if reset_timestamps else 0
        self.dvs_events = None

    def __del__(self):
        pass

    # @property
    # def size(self):
    #     for packet in self.decoder:
    #         if 'frame' in packet:
    #             h, w = packet['frame']['height'], packet['frame']['width']
    #             return h, w
    #     raise Exception('frame size not found')

    def __iter__(self):
        for packet in self.decoder:
            if 'events' in packet:
                events = packet['events']
                num = len(events)
                if self._t0 is None:
                    self._t0 = events['t'][0]
                event_buffer = np.zeros((num,), dtype=dvs_dtype)
                event_buffer['t'][:num] = (events['t'] - self._t0)
                event_buffer['x'][:num] = events['x']
                event_buffer['y'][:num] = events['y']
                event_buffer['p'][:num] = events['on']
                yield event_buffer
            else:
                continue

    def collect_info(self):
        events = []
        for evts in self.__iter__():
            events.append(evts)
        self.dvs_events = np.hstack(events)
        return self.dvs_events

    def events_Nx4(self, dtype=int):
        if self.dvs_events is None:
            self.collect_info()
        events = np.array([self.dvs_events['t'].astype(dtype),
                           self.dvs_events['x'].astype(dtype),
                           self.dvs_events['y'].astype(dtype),
                           self.dvs_events['p'].astype(dtype)]).T
        return events


class AedatAPSReader(object):
    """
    BaseReader for .aedat4 format.
    We write a base iterator returning each event packet as a numpy array with the 'events_dtype' format, i.e. with
    names (t, x, y, p) having types (int64, int16, int16, bool) respectively.
    The full array of all events in the recording (i.e. all packets together) is returned by the method 'array()'.
    If you want a full array with shape (N,4) [where N is the total number of events], call the method 'array_Nx4()'.
    """

    def __init__(self, file: str):
        assert os.path.exists(file), "The given file does not exist."
        file_extension = os.path.splitext(file)[1]
        assert (file_extension in ['.aedat', '.aedat4']), "The given file must be either a .aedat or .aedat4 file."

        self.decoder = aedat.Decoder(file)
        self._frames = None
        self._frames_timestamps = None

    def __del__(self):
        pass

    # @property
    # def size(self):
    #     for packet in self.decoder:
    #         if 'frame' in packet:
    #             h, w = packet['frame']['height'], packet['frame']['width']
    #             return h, w
    #     raise Exception('frame size not found')

    def __iter__(self):
        for packet in self.decoder:
            if 'frame' in packet:
                frame_info = packet['frame']
                yield frame_info['pixels'], frame_info['t']
            else:
                continue

    def collect_info(self):
        frames, timestamps = [], []
        for frm in self.__iter__():
            frames.append(frm[0])
            timestamps.append(frm[1])
        self._frames = np.stack(frames)
        self._frames_timestamps = np.hstack(timestamps)
        return self._frames, self._frames_timestamps

    @property
    def frames(self, dtype=np.uint8):
        if self._frames is None:
            self.collect_info()
        return self._frames.astype(dtype)

    @property
    def frames_timestamps(self, dtype=int):
        if self._frames_timestamps is None:
            self.collect_info()
        return self._frames_timestamps.astype(dtype)


class AedatIMUReader(object):
    """
    BaseReader for .aedat4 format.
    We write a base iterator returning each event packet as a numpy array with the 'events_dtype' format, i.e. with
    names (t, x, y, p) having types (int64, int16, int16, bool) respectively.
    The full array of all events in the recording (i.e. all packets together) is returned by the method 'array()'.
    If you want a full array with shape (N,4) [where N is the total number of events], call the method 'array_Nx4()'.
    """

    def __init__(self, file: str):
        assert os.path.exists(file), "The given file does not exist."
        file_extension = os.path.splitext(file)[1]
        assert (file_extension in ['.aedat', '.aedat4']), "The given file must be either a .aedat or .aedat4 file."

        self.decoder = aedat.Decoder(file)
        self.imu_events = None

    def __del__(self):
        pass

    # @property
    # def size(self):
    #     for packet in self.decoder:
    #         if 'frame' in packet:
    #             h, w = packet['frame']['height'], packet['frame']['width']
    #             return h, w
    #     raise Exception('frame size not found')

    def __iter__(self):
        for packet in self.decoder:
            if 'imus' in packet:
                events = packet['imus']
                num = len(events)
                event_buffer = np.zeros((num,), dtype=imu_dtype)
                event_buffer['t'][:num] = events['t']
                event_buffer['acc_x'][:num] = events['accelerometer_x']
                event_buffer['acc_y'][:num] = events['accelerometer_y']
                event_buffer['acc_z'][:num] = events['accelerometer_z']
                event_buffer['gyro_x'][:num] = events['gyroscope_x']
                event_buffer['gyro_y'][:num] = events['gyroscope_y']
                event_buffer['gyro_z'][:num] = events['gyroscope_z']
                event_buffer['magnet_x'][:num] = events['magnetometer_x']
                event_buffer['magnet_y'][:num] = events['magnetometer_y']
                event_buffer['magnet_z'][:num] = events['magnetometer_z']
                event_buffer['temperature'][:num] = events['temperature']
                yield event_buffer
            else:
                continue

    def collect_info(self):
        events = []
        for evt in self.__iter__():
            events.append(evt)
        self.imu_events = np.hstack(events)
        return self.imu_events

    def events_Nx8(self, dtype=int):
        if self.imu_events is None:
            self.collect_info()
        events = np.array([self.imu_events['t'].astype(dtype),
                           self.imu_events['acc_x'].astype(dtype),
                           self.imu_events['acc_y'].astype(dtype),
                           self.imu_events['acc_z'].astype(dtype),
                           self.imu_events['gyro_x'].astype(dtype),
                           self.imu_events['gyro_y'].astype(dtype),
                           self.imu_events['gyro_z'].astype(dtype),
                           self.imu_events['temperature'].astype(dtype)]).T
        return events

    def events_Nx11(self, dtype=int):
        if self.imu_events is None:
            self.collect_info()
        events = np.array([self.imu_events['t'].astype(dtype),
                           self.imu_events['acc_x'].astype(dtype),
                           self.imu_events['acc_y'].astype(dtype),
                           self.imu_events['acc_z'].astype(dtype),
                           self.imu_events['gyro_x'].astype(dtype),
                           self.imu_events['gyro_y'].astype(dtype),
                           self.imu_events['gyro_z'].astype(dtype),
                           self.imu_events['magnet_x'].astype(dtype),
                           self.imu_events['magnet_y'].astype(dtype),
                           self.imu_events['magnet_z'].astype(dtype),
                           self.imu_events['temperature'].astype(dtype)]).T
        return events
