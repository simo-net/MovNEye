import os
from pandas import read_csv


def check_file(path_to_file: str):
    # Check if the file exists and if its extension is valid
    assert os.path.exists(path_to_file), "The given file does not exist."
    file_extension = os.path.splitext(path_to_file)[1]
    assert (file_extension in ['.txt', '.csv']), "The given file must be either a .txt/.csv file."


def read_header(path_to_file: str, num_header_lines: int):
    # Read the header from the file
    with open(path_to_file, 'r') as event_file:
        header = list(map(lambda _: event_file.readline(), range(num_header_lines)))
    header = ''.join(header)
    return header


def compute_num_events(path_to_file: str, skip: int = 0):
    # Compute the total number of events in the file
    with open(path_to_file, 'r') as event_file:
        for _ in range(skip):
            event_file.readline()
        tot_events = 0
        for _ in event_file:
            tot_events += 1
    return tot_events


class LoadEventsInPackets(object):
    """
    Reads events from a '.txt' or '.csv' file, and packages the events into
    non-overlapping event windows (packets), each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file: str, events_per_packet: int = 8000,
                 start_index: int = 0, num_header_lines: int = 2):

        check_file(path_to_event_file)
        self.path_to_event_file = path_to_event_file
        self.num_header_lines = num_header_lines
        self.start_index = start_index

        # Define the width and height of the sensor from the header


        # Define an iterator over the events in the file
        self._iterator = read_csv(path_to_event_file, compression=None,
                                  iterator=True, chunksize=events_per_packet, engine='c', memory_map=True,
                                  delim_whitespace=True, header=None, nrows=None, skiprows=start_index+num_header_lines,
                                  dtype={'t': int, 'x': int, 'y': int, 'p': int},
                                  names=['t', 'x', 'y', 'p'])

    def __iter__(self):
        return self

    def __next__(self):
        events = self._iterator.__next__().values
        return events.astype(int)

    @property
    def header(self) -> str:
        return read_header(self.path_to_event_file, num_header_lines=self.num_header_lines)

    @property
    def tot_events(self) -> int:
        return compute_num_events(self.path_to_event_file, skip=self.num_header_lines+self.start_index)

    @property
    def sensor_size(self) -> (int, int):
        _, width, height = read_csv(self.path_to_event_file, compression=None,
                                    delim_whitespace=True, header=None, nrows=1, skiprows=0,
                                    dtype={'text': str, 'width': int, 'height': int},
                                    names=['text', 'width', 'height']).values[0]
        return width, height


class LoadEventsInPacketsOfDuration:
    """
    Reads events from a '.txt' or '.csv' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the LoadEventsInPackets.
              The reason is that the latter can use Pandas' very efficient chunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file: str, duration_per_packet: int = 50,
                 start_index: int = 0, num_header_lines: int = 2):

        check_file(path_to_event_file)
        self.path_to_event_file = path_to_event_file
        self.duration_per_packet = duration_per_packet
        self.num_header_lines = num_header_lines
        self.start_index = start_index

        # Grub the first timestamp of the events
        self._last_timestamp = None
        self._first_timestamp = read_csv(path_to_event_file, compression=None,
                                         delim_whitespace=True, header=None, nrows=1, skiprows=start_index+num_header_lines,
                                         dtype={'t': int, 'x': int, 'y': int, 'p': int},
                                         names=['t', 'x', 'y', 'p'])['t'].iloc[0]

        # Open the event file for reading (ignore the first start_index lines and the header)
        self.event_file = open(path_to_event_file, 'r')
        for _ in range(start_index + num_header_lines):
            self.event_file.readline()
        self._eof = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._eof:
            raise StopIteration
        event_list = []
        for line in self.event_file:
            t, x, y, p = line.split(' ')
            t, x, y, p = float(t) * 1e-6, int(x), int(y), int(p)
            event_list.append([t, x, y, p])
            if self._last_timestamp is None:
                self._last_timestamp = t
            if t > self._last_timestamp + self.duration_per_packet:
                self._last_timestamp = t
                return event_list
        self._eof = True
        return event_list

    def __del__(self):
        self.event_file.close()

    @property
    def start_ts(self):
        return self._first_timestamp

    @property
    def last_ts(self):
        return self._last_timestamp

    @property
    def header(self) -> str:
        return read_header(self.path_to_event_file, num_header_lines=self.num_header_lines)

    @property
    def tot_events(self) -> int:
        return compute_num_events(self.path_to_event_file, skip=self.num_header_lines + self.start_index)

    @property
    def sensor_size(self) -> (int, int):
        _, width, height = read_csv(self.path_to_event_file, compression=None,
                                    delim_whitespace=True, header=None, nrows=1, skiprows=0,
                                    dtype={'text': str, 'width': int, 'height': int},
                                    names=['text', 'width', 'height']).values[0]
        return width, height
