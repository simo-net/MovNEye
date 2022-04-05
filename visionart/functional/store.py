import os
import sys
import json
from visionart.functional.read import read_json
from visionart.calib.paths import INFO_DIR, CHESS_FILE, IMG_FILE, SCREEN_FILE, SENSOR_FILE


def store_json(file: str, info: dict):
    with open(file, 'w+') as info_json:
        json.dump(info, info_json, indent=4)


def update_json(file: str, new_entries: dict):
    info = read_json(file)
    for k, v in zip(new_entries.keys(), new_entries.values()):
        if isinstance(v, dict):
            if not isinstance(info[k], dict):
                info[k] = {}
            for k2, v2 in zip(v.keys(), v.values()):
                info[k][k2] = v2
        else:
            info[k] = v
    store_json(file, info)


# ----------------------------------------------------- Sensor -------------------------------------------------------

def store_sensor_info(resolution: (int, int), model: str, serial: str):
    if os.path.isfile(SENSOR_FILE):
        print('Sensor information file already exists and will be overwritten!', file=sys.stderr)

    # Create and store monitor information structure
    davis_info = {'resolution': resolution,  # the resolution of the sensor
                  'model': model,  # the model of the sensor
                  'serial': serial}  # the serial number of the sensor
    with open(SENSOR_FILE, 'w+') as info_json:
        json.dump(davis_info, info_json, indent=4)
    return davis_info


def retrieve_sensor_info():
    from visionart.recording import RecordScene
    myexp = RecordScene()
    resolution = myexp.davis_shape
    model = myexp.davis_name + str(resolution[0])
    serial = myexp.davis_serial_number
    myexp.close()
    return resolution, model, serial


# ----------------------------------------------------- Monitor ------------------------------------------------------

def store_monitor_info(resolution: (int, int), diagonal: float, ppi: float):
    if os.path.isfile(SCREEN_FILE):
        print('Monitor information file already exists and will be overwritten!', file=sys.stderr)

    # Create and store monitor information structure
    screen_info = {'resolution': resolution,  # the resolution of the monitor
                   'diagonal': diagonal,  # the diagonal of the monitor (in inches)
                   'ppi': ppi}  # the number of pixels per inch
    with open(SCREEN_FILE, 'w+') as info_json:
        json.dump(screen_info, info_json, indent=4)
    return screen_info


# ---------------------------------------------------- Chessboard ----------------------------------------------------

def store_chessboard_info(shape: (int, int), square: int, border: (int, int)):
    if os.path.isfile(CHESS_FILE):
        print('Chessboard information file already exists and will be overwritten!', file=sys.stderr)

    # Create and store chessboard information structure
    chess_info = {'shape': shape,  # the shape of the chessboard (number of squares in x and y)
                  'square': square,  # the dimension of each square in the chessboard (in pixels)
                  'border': border,  # the white border around the chessboard (in x and y pixels)
                  'image': os.path.join(INFO_DIR, 'chessboard.npy'),  # the chessboard image as a numpy array
                  'corners': os.path.join(INFO_DIR, 'chesscorners.npy')}  # the positions of its internal corners
    with open(CHESS_FILE, 'w+') as info_json:
        json.dump(chess_info, info_json, indent=4)
    return chess_info


# ------------------------------------------------------ Image -------------------------------------------------------

def store_image_info(shape: (int, int), border: float, strinfo: str):
    assert shape is not None, 'Must specify shape (width, height) of the images to use as stimuli.'
    if os.path.isfile(IMG_FILE):
        print('Stimulus information file already exists and will be overwritten!', file=sys.stderr)

    # Create and store sample-image information structure
    img_info = {'shape': shape,  # the shape of the image (its (x,y) resolution)
                'border': border,  # the border surrounding the image (as float or string)
                'info': strinfo,  # some general information (string metadata) on the image stimuli
                'image': os.path.join(INFO_DIR, 'image.npy')}  # the sample image (black rectangle) as a numpy array
    with open(IMG_FILE, 'w+') as info_json:
        json.dump(img_info, info_json, indent=4)
    return img_info
