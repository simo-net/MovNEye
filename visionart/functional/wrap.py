import os
import numpy as np


def read_bin(bin_file: str) -> np.ndarray:
    """
    Reads events from a binary file where N spikes are stored as an unsigned 8 bit Nx5 array and each event is 40 bits.
    Returns an array Nx4 where columns are organized as t,x,y,p.
    Code adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py and
    https://bitbucket.org/bamsumit/spikefilesreadwrite/src/master/Read_Ndataset.m
    """
    f = open(bin_file, "rb")
    raw_data = np.uint32(np.fromfile(f, dtype=np.uint8))
    f.close()

    # X-COORDINATES are stored in all 8 bits of array[0::5]
    all_x = (raw_data[0::5] & 255)
    # Y-COORDINATES are stored in all 8 bits of array[1::5]
    all_y = (raw_data[1::5] & 255)
    # POLARITY are stored in only 1 of the 8 bits of array[2::5], the other 7 bits are for the timestamp
    all_p = (raw_data[2::5] & 128) >> 7
    # TIMESTAMPS are stored in 23 bits in total: the last 7 bits of array[2::5] (together with the polarity) +
    # all 8 bits in array[3::5] + all 8 bits in array[4::5]
    all_t = ((raw_data[2::5] & 127) << 16) | ((raw_data[3::5] & 255) << 8) | (raw_data[4::5] & 255)

    return np.c_[all_t, all_x, all_y, all_p].astype(int)


def write_bin(bin_file: str, events: np.ndarray):
    """
    Writes spikes to a binary file. Each event will occupy 40 bits: x (8 bits), y (8 bits), p (1 bit), t (23 bits).
    An unsigned 8 bit array will be stored in the binary file with shape Nx5, where N is the total number of events.
    The 5 columns are: 1) 8 bits of x address, 2) 8 bits of y address, 3) 1 bit of polarity + last 7 bits of timestamp,
    4) middle 8 bits of timestamp, 5) first 8 bits of timestamp.
    The input events array should be a Nx4 array where columns are organized as t,x,y,p. Note that all values of x and y
    address should be < 2^8, polarity < 2 and timestamp < 2^23.
    Code adapted from https://bitbucket.org/bamsumit/spikefilesreadwrite/src/master/Encode_Ndataset.m
    """
    assert int(events[-1, 0]).bit_length() <= 23
    assert int(events[:, 1:3].max()).bit_length() <= 8
    assert int(events[:, 3].max()).bit_length() <= 1
    byte_events = np.zeros(events.shape[0]*5, dtype=np.uint8)

    # X-COORDINATES are stored in all 8 bits of byte_events[0::5]
    byte_events[0::5] = (events[:, 1] & 255)
    # Y-COORDINATES are stored in all 8 bits of byte_events[1::5]
    byte_events[1::5] = (events[:, 2] & 255)
    # POLARITY are stored in only 1 of the 8 bits of byte_events[2::5], the other 7 bits are for the timestamp
    byte_events[2::5] = (((events[:, 3] << 7) & 128) | ((events[:, 0] >> 16) & 127))
    # TIMESTAMPS are stored in 23 bits in total: the last 7 bits of byte_events[2::5] (together with the polarity) +
    # all 8 bits in byte_events[3::5] + all 8 bits in byte_events[4::5]
    byte_events[3::5] = ((events[:, 0] >> 8) & 255)
    byte_events[4::5] = (events[:, 0] & 255)

    f = open(bin_file, "wb")
    byte_events.tofile(f)
    f.close()

    check_file_size(bin_file, num_events=events.shape[0])


def write_csv(csv_file: str, events: np.ndarray):
    f = open(csv_file, 'w+')
    np.savetxt(f, events, fmt='%i %i %i %i', newline="\n")
    f.close()


def read_csv(csv_file: str):
    return np.genfromtxt(csv_file, delimiter=' ', dtype=int)


def check_file_size(bin_file: str, num_events: int):
    real_size, expected_size = os.path.getsize(bin_file), num_events * 40 // 8
    assert real_size == expected_size,\
        f'The size of the binary file is different from expected: it is {real_size} but should be {expected_size} Bytes'
