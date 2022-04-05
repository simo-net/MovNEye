import numpy as np
from tqdm import tqdm
from pyaer import libcaer
from pyaer.filters import DVSNoise
from visionart.sensor.loadpacket import LoadEventsInPackets


EVENT_CAPACITY = 2**18
packet_header = libcaer.caerEventPacketAllocate(EVENT_CAPACITY,              # eventCapacity
                                                1, 0,                        # eventSource, eventTSOverflow
                                                libcaer.POLARITY_EVENT,      # eventType
                                                8, 4)                        # eventSize, eventTSOffset


def set_packet(event_number):
    libcaer.caerEventPacketHeaderSetEventNumber(packet_header, event_number)
    libcaer.caerEventPacketClear(packet_header)
    packet = libcaer.caerPolarityEventPacketFromPacketHeader(packet_header)
    return packet


def get_packet():
    events = libcaer.caerPolarityEventPacketFromPacketHeader(packet_header)
    return events


def filter_packet(packet, num_events):
    events = libcaer.get_filtered_polarity_event(packet, num_events * 5).reshape(num_events, 5)
    return events[events[:, 4] == 1][:, :-1]


def set_event(packet, idx, event):
    t, x, y, p = event
    event_info = libcaer.caerPolarityEventPacketGetEvent(packet, idx)
    libcaer.caerPolarityEventSetTimestamp(event_info, int(t))
    libcaer.caerPolarityEventSetX(event_info, int(x))
    libcaer.caerPolarityEventSetY(event_info, int(y))
    libcaer.caerPolarityEventSetPolarity(event_info, bool(p))
    libcaer.caerPolarityEventValidate(event_info, packet)


def yield_filtered_packet_from_file(packet_iterator: LoadEventsInPackets, filter_bias: str,
                                    show_progress: bool = True) -> np.ndarray:
    """Reads the events in packets from the given input file, it filters them and then yields them packet-by-packet,
    so that they can be used and released from memory before the operation (loading and noise-filtering) is repeated
    on the next packet."""

    # Define the DVS noise filter from libcaer
    width, height = packet_iterator.sensor_size
    noise_filter = DVSNoise(width, height)
    noise_filter.set_bias_from_json(filter_bias, verbose=False)

    # Loop through all packets (each one will contain EVENT_CAPACITY-1 events, except for the last one)
    pbar = tqdm(total=packet_iterator.tot_events, desc='Progress', disable=not show_progress)
    for packed_events in packet_iterator:
        num_events = packed_events.shape[0]

        # Define the empty packet
        packet = set_packet(num_events)

        # Fill it with the events from the original recording
        for event_id in range(num_events):
            event = packed_events[event_id]
            set_event(packet, event_id, event)
            pbar.update(1)
        packet = get_packet()

        # Filter out noisy events
        packet_filt = noise_filter.apply(packet)
        packed_events_filt = filter_packet(packet_filt, num_events)

        yield packed_events_filt

    noise_filter.destroy()
    pbar.close()


def return_filtered_events(dvs_file: str, filter_bias: str, return_header: bool = False,
                           show_progress: bool = True) -> np.ndarray or (np.ndarray, str):
    """Reads the events in packets from the given input file, it filters them and then returns them all together (i.e.
     ir returns an array of all the noise-filtered events from the input file).
    """

    # Define an iterator over events in the CSV file of DVS recording: read them in packets of size EVENT_CAPACITY-1
    file_iterator = LoadEventsInPackets(dvs_file, events_per_packet=EVENT_CAPACITY - 1,
                                        num_header_lines=2, start_index=0)
    file_header = file_iterator.header

    # Loop through all packets (each one will contain EVENT_CAPACITY-1 events, except for the last one)
    filtered_events = np.empty((0, 4), dtype=int)
    for filtered_packet in yield_filtered_packet_from_file(packet_iterator=file_iterator, filter_bias=filter_bias,
                                                           show_progress=show_progress):
        filtered_events = np.vstack((filtered_events, filtered_packet))

    if return_header:
        return filtered_events, file_header
    return filtered_events


def store_filtered_events(old_dvs_file: str, new_dvs_file: str, filter_bias: str, show_progress: bool = True):
    """Reads the events in packets from the given input file, it filters them and then saves them to the output file
    in chunks (i.e. packet-by-packet, meaning that filtered events are stored to file and released from memory when a
    new packet is picked up). The advantage of using this function instead of the previous one is that the memory will
    never be saturated, neither for large files (long recordings).
    """

    if old_dvs_file == new_dvs_file:
        raise ValueError('Cannot overwrite the existing file using this function! If this is what you want, use '
                         'overwrite_filtered_events_from_file() instead.')

    # Define an iterator over events in the CSV file of DVS recording: read them in packets of size EVENT_CAPACITY-1
    file_iterator = LoadEventsInPackets(old_dvs_file, events_per_packet=EVENT_CAPACITY - 1,
                                        num_header_lines=2, start_index=0)
    file_header = file_iterator.header

    # Open the .output csv file for writing (it will contain DVS filtered events)
    with open(new_dvs_file, 'w+') as new_txt:

        # Write the header to file
        new_txt.write(file_header)

        # Loop through all filtered packets and write them to file
        for filtered_packet in yield_filtered_packet_from_file(packet_iterator=file_iterator, filter_bias=filter_bias,
                                                               show_progress=show_progress):
            np.savetxt(new_txt, filtered_packet, fmt='%i %i %i %i', newline="\n")


def overwrite_filtered_events(dvs_file: str, filter_bias: str, show_progress: bool = True):
    """Reads the events in packets from the given input file, it filters them and then saves them to the output file
    all together (i.e. an array of all output events are stored to file all together).
    """

    # Filter all the events in the given input file
    filtered_events, header = return_filtered_events(dvs_file=dvs_file, filter_bias=filter_bias,
                                                     return_header=True, show_progress=show_progress)

    # Open the .output csv file for writing (it will contain DVS filtered events)
    with open(dvs_file, 'w+') as new_txt:

        # Write the header to file
        new_txt.write(header)

        # Write filtered events to file
        np.savetxt(new_txt, filtered_events, fmt='%i %i %i %i', newline="\n")
