import os
import sys
import argparse
from visionart.utils.add2os import move_file, listdir_flatten, keep_list_extension
from visionart.functional.read import recording_files
from visionart.functional.store import update_json


# Usage:
# >>    python3 3-rename.py  --old_dir ./data/rec/seed_0
#                            --new_dir ./data/rec/batch_1


def parse_args():
    parser = argparse.ArgumentParser(description='Rename the main path of all recordings in a given path.')

    parser.add_argument('--old_dir', type=str, required=True,
                        help='')
    parser.add_argument('--new_dir', type=str, required=True,
                        help='')

    return parser.parse_args()


def main(old_dir: str, new_dir: str):
    old_dir = os.path.join(old_dir, '')
    new_dir = os.path.join(new_dir, '')
    assert isinstance(old_dir, str), 'The old directory must be a string!'
    assert isinstance(new_dir, str), 'The new directory must be a string!'

    if old_dir == new_dir:
        print('Nothing will be done since the new directory equals the old one.')
        sys.exit()

    all_files = keep_list_extension(listdir_flatten(old_dir), extension='.json', empty_error=True)
    k, n = 1, len(all_files)

    print(f'\nRenaming the directory {old_dir} to {new_dir} in all json files...')
    for old_file_info in all_files:
        print(f'Progress: {round(k / n * 100)} %', end="\r", flush=True)

        aps_old, dvs_old, imu_old, err_old = recording_files(old_file_info)
        old_files = {'aps': aps_old, 'dvs': dvs_old, 'imu': imu_old, 'err': err_old}

        new_files = dict()
        for key, file_old in zip(old_files.keys(), old_files.values()):
            file_new = move_file(file_old, old_dir, new_dir)
            new_files[key] = file_new

        update_json(old_file_info, new_entries={'rec': new_files})
        _ = move_file(old_file_info, old_dir, new_dir)

        k += 1
    print('Progress: 100 %  -->  Done!\n')

    # Clear all empty directories
    folders = os.listdir(old_dir)
    for fold in folders:
        fold_dir = os.path.join(os.path.split(old_dir)[0], fold)
        if os.path.exists(fold_dir) and os.path.isdir(fold_dir):
            if not os.listdir(fold_dir):
                os.removedirs(fold_dir)


if __name__ == "__main__":

    args = parse_args()

    main(old_dir=args.old_dir, new_dir=args.new_dir)
