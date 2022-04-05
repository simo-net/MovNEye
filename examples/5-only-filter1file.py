import os
import argparse
from visionart.functional.read import read_json
from visionart.functional.store import store_json
from visionart.filtering import store_filtered_events

# Usage:
# >>    python3 5-only-filter1file.py  -i ./data/rec/fem/seed_0/imk00002.json
#                                      -o ./data/rec_filt/fem/seed_0/imk00002.json
#                                      --filter_file ./sensor/configs/dvsnoisefilter_biases.json


def parse_args():
    parser = argparse.ArgumentParser(description='Offline filtering of DAVIS events.')

    parser.add_argument('-i', '--file2load', type=str, required=True,
                        help='Full path of the input .json file of recording info to load.\nNote: the names '
                             'for all the recorded events saved (APS, DVS and IMU, as well as FEM and IMG info) '
                             'are taken from here.')
    parser.add_argument('-o', '--file2save', type=str, required=True,
                        help='Full path of the .json file where to update the reference to the file of output\n'
                             'filtered DVS events (wrt the input file).')
    parser.add_argument('--filter_file', type=str, default='./sensor/configs/dvsnoisefilter_biases.json',
                        help='Full path of the .json file where DVS background-noise filter parameters are stored.')

    return parser.parse_args()


def main(old_file: str, new_file: str,
         filter_file: str):

    # Recover old file of DVS recording
    old_info = read_json(old_file)
    old_file_dvs = old_info['rec']['dvs']

    if old_file == new_file:
        print('Cannot overwrite old file.')
        return None

    # Update the info structure (only the DVS recording file, now pointing to the filtered events)....
    new_info = old_info
    new_file_dvs = os.path.splitext(new_file)[0] + '_dvs.csv'
    new_info['rec']['dvs'] = new_file_dvs
    # ...and save it to the new output json file
    os.makedirs(os.path.split(new_file)[0], exist_ok=True)
    store_json(file=new_file, info=new_info)

    # Read events from old file and store the filtered events to new file
    store_filtered_events(old_dvs_file=old_file_dvs, new_dvs_file=new_file_dvs, filter_bias=filter_file)


if __name__ == "__main__":

    args = parse_args()
    main(old_file=args.file2load, new_file=args.file2save, filter_file=args.filter_file)
