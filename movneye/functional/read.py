import os
import json
import numpy as np
from movneye.calib.paths import CHESS_FILE, IMG_FILE, SCREEN_FILE, SENSOR_FILE


def read_json(info_file: str = None):
    info_struct = {}
    if info_file is not None and os.path.isfile(info_file) and os.path.splitext(info_file)[-1] == '.json':
        with open(info_file, 'r') as info_json:
            info_struct = json.load(info_json)
    return info_struct


def read_json_decorator(func):
    def inner(info):
        if isinstance(info, dict) and info:
            return func(info)
        if isinstance(info, str) and os.path.isfile(info):
            return func(read_json(info))
        else:
            raise TypeError('You must pass either a non-empty dictionary or an existing json file as argument.')
    return inner


# ----------------------------------------------------- Sensor -------------------------------------------------------

def sensor_info(sensor_file: str = SENSOR_FILE) -> ((int, int), str, str):
    assert isinstance(sensor_file, str) and os.path.isfile(sensor_file), 'The argument must be an existing json file.'
    info_struct = read_json(sensor_file)
    resolution = sensor_resolution(info_struct)
    model = sensor_model(info_struct)
    serial = sensor_serial(info_struct)
    return resolution, model, serial


@read_json_decorator
def sensor_resolution(info: dict or str) -> (int, int) or (None, None):
    try:
        return tuple(info['resolution'])
    except (KeyError, TypeError):
        return None, None


@read_json_decorator
def sensor_model(info: dict or str) -> str or None:
    try:
        return info['model']
    except (KeyError, TypeError):
        return None


@read_json_decorator
def sensor_serial(info: dict or str) -> str or None:
    try:
        return info['serial']
    except (KeyError, TypeError):
        return None

# ----------------------------------------------------- Monitor ------------------------------------------------------

def monitor_info(screen_file: str = SCREEN_FILE) -> ((int, int), float, float):
    assert isinstance(screen_file, str) and os.path.isfile(screen_file), 'The argument must be an existing json file.'
    info_struct = read_json(screen_file)
    resolution = monitor_resolution(info_struct)
    diagonal = monitor_diagonal(info_struct)
    ppi = monitor_ppi(info_struct)
    return resolution, diagonal, ppi


@read_json_decorator
def monitor_resolution(info: dict or str) -> (int, int) or (None, None):
    try:
        return tuple(info['resolution'])
    except (KeyError, TypeError):
        return None, None


@read_json_decorator
def monitor_diagonal(info: dict or str) -> float or None:
    try:
        return float(info['diagonal'])
    except (KeyError, TypeError):
        return None


@read_json_decorator
def monitor_ppi(info: dict or str) -> float or None:
    try:
        return float(info['ppi'])
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------- Chessboard ----------------------------------------------------

def chessboard_info(chess_file: str = CHESS_FILE) -> ((int, int), (int, int), int, np.ndarray, np.ndarray):
    assert isinstance(chess_file, str) and os.path.isfile(chess_file), 'The given path is not a file.'
    info_struct = read_json(chess_file)
    shape = chessboard_shape(info_struct)          # number of internal corners in both dimensions
    border = chessboard_border(info_struct)        # x and y spaces between chess and monitor end in pixels
    dim = chessboard_squaredim(info_struct)        # dimension of each square in pixels
    chess = chessboard_image(info_struct)          # chessboard image
    corners3d = chessboard_corners3d(info_struct)  # 3D points of chess internal corners in pixels
    return shape, border, dim, chess, corners3d


@read_json_decorator
def chessboard_shape(info: dict or str) -> (int, int) or (None, None):
    try:
        return tuple(info['shape'])
    except (KeyError, TypeError):
        return None, None


@read_json_decorator
def chessboard_border(info: dict or str) -> (int, int) or (None, None):
    try:
        return tuple(info['border'])
    except (KeyError, TypeError):
        return None, None


@read_json_decorator
def chessboard_squaredim(info: dict or str) -> int or None:
    try:
        return int(info['square'])
    except (KeyError, TypeError):
        return None


@read_json_decorator
def chessboard_file(info: dict or str) -> str or None:
    try:
        chess_file = info['image']
        if not (isinstance(chess_file, str) and os.path.isfile(chess_file)):
            chess_file = None
    except (KeyError, TypeError):
        return None
    return chess_file


@read_json_decorator
def chessboard_image(info: dict or str) -> np.ndarray or None:
    try:
        return np.load(chessboard_file(info))
    except (KeyError, TypeError):
        return None


@read_json_decorator
def chessboard_corners_file(info: dict or str) -> str or None:
    try:
        corners_file = info['corners']
        if not (isinstance(corners_file, str) and os.path.isfile(corners_file)):
            corners_file = None
    except (KeyError, TypeError):
        return None
    return corners_file


@read_json_decorator
def chessboard_corners2d(info: dict or str) -> np.ndarray or None:
    try:
        return np.load(chessboard_corners_file(info))
    except (KeyError, TypeError):
        return None


@read_json_decorator
def chessboard_corners3d(info: dict or str) -> np.ndarray or None:
    try:
        corners2d = chessboard_corners2d(info)  # points of chess internal corners in pixels
        corners3d = np.hstack((corners2d, np.zeros((corners2d.shape[0], 1))))  # add Z=0
        return corners3d
    except (KeyError, TypeError):
        return None


# ----------------------------------------------- Stimulus-Calibration -----------------------------------------------

def stimcalib_info(img_file: str = IMG_FILE) -> ((int, int), float, str, np.ndarray):
    assert isinstance(img_file, str) and os.path.isfile(img_file), 'The given path is not a file.'
    info_struct = read_json(img_file)
    shape = stimcalib_shape(info_struct)          # the shape of the image (its (x,y) resolution)
    border = stimcalib_border(info_struct)        # the border surrounding the image (as float)
    strinfo = stimcalib_strinfo(info_struct)      # some general information (string metadata) on the image stimuli
    img = stimcalib_image(info_struct)            # the sample image (black rectangle) as a numpy array
    return shape, border, strinfo, img


@read_json_decorator
def stimcalib_shape(info: dict or str) -> (int, int) or (None, None):
    try:
        return tuple(info['shape'])
    except (KeyError, TypeError):
        return None, None


@read_json_decorator
def stimcalib_border(info: dict or str) -> float or None:
    try:
        return float(info['border'])
    except (KeyError, TypeError):
        return None


@read_json_decorator
def stimcalib_strinfo(info: dict or str) -> str or None:
    try:
        return str(info['info'])
    except (KeyError, TypeError):
        return None


@read_json_decorator
def stimcalib_file(info: dict or str) -> str or None:
    try:
        img_file = info['image']
        if not (isinstance(img_file, str) and os.path.isfile(img_file)):
            img_file = None
    except (KeyError, TypeError):
        return None
    return img_file


@read_json_decorator
def stimcalib_image(info: dict or str) -> np.ndarray or None:
    try:
        return np.load(stimcalib_file(info))
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------- Recording -----------------------------------------------------

def recording_info(rec_file: str) -> (str, str, str, str):
    assert isinstance(rec_file, str) and os.path.isfile(rec_file), 'The argument must be an existing json file.'
    info_struct = read_json(rec_file)
    return recording_files(info_struct)


@read_json_decorator
def recording_files(info: dict or str) -> (str, str, str, str):
    aps_file = recording_apsfile(info)
    dvs_file = recording_dvsfile(info)
    imu_file = recording_imufile(info)
    err_file = recording_errfile(info)
    return aps_file, dvs_file, imu_file, err_file


@read_json_decorator
def recording_apsfile(info: dict or str) -> str or None:
    try:
        aps_file = info['rec']['aps']
        if not (isinstance(aps_file, str) and os.path.isfile(aps_file)):
            aps_file = None
    except (KeyError, TypeError):
        return None
    return aps_file


@read_json_decorator
def recording_dvsfile(info: dict or str) -> str or None:
    try:
        dvs_file = info['rec']['dvs']
        if not (isinstance(dvs_file, str) and os.path.isfile(dvs_file)):
            dvs_file = None
    except (KeyError, TypeError):
        return None
    return dvs_file


@read_json_decorator
def recording_imufile(info: dict or str) -> str or None:
    try:
        imu_file = info['rec']['imu']
        if not (isinstance(imu_file, str) and os.path.isfile(imu_file)):
            imu_file = None
    except (KeyError, TypeError):
        return None
    return imu_file


@read_json_decorator
def recording_errfile(info: dict or str) -> str or None:
    try:
        err_file = info['rec']['err']
        if not (isinstance(err_file, str) and os.path.isfile(err_file)):
            err_file = None
    except (KeyError, TypeError):
        return None
    return err_file


# ----------------------------------------------------- Movement -----------------------------------------------------

def movement_info(rec_file: str) -> (str, int):
    info_struct = read_json(rec_file)
    fem_file = movement_file(info_struct)
    fem_seed = movement_seed(info_struct)
    return fem_file, fem_seed


@read_json_decorator
def movement_file(info: dict or str) -> str or None:
    try:
        fem_file = info['fem']['file']
        if not (isinstance(fem_file, str) and os.path.isfile(fem_file)):
            fem_file = None
    except (KeyError, TypeError):
        return None
    return fem_file


@read_json_decorator
def movement_seed(info: dict or str) -> int or None:
    try:
        fem_seed = info['fem']['seed']
    except (KeyError, TypeError):
        return None
    return fem_seed


# ----------------------------------------------------- Stimulus -----------------------------------------------------

def stimulus_info(rec_file: str) -> (str, int, int or str, str, str):
    assert isinstance(rec_file, str) and os.path.isfile(rec_file), 'The argument must be an existing json file.'
    info_struct = read_json(rec_file)
    img_file = stimulus_file(info_struct)
    img_index = stimulus_index(info_struct)
    img_label = stimulus_label(info_struct)
    img_split = stimulus_split(info_struct)
    img_dataset = stimulus_dataset(info_struct)
    img_transforms = stimulus_transforms(info_struct)
    return img_file, img_index, img_label, img_split, img_dataset, img_transforms


@read_json_decorator
def stimulus_file(info: dict or str) -> str or None:
    try:
        img_file = info['img']['file']
        if not (isinstance(img_file, str) and os.path.isfile(img_file)):
            img_file = None
    except (KeyError, TypeError):
        return None
    return img_file


@read_json_decorator
def stimulus_index(info: dict or str) -> int or None:
    try:
        img_index = info['img']['index']
    except (KeyError, TypeError):
        return None
    return img_index


@read_json_decorator
def stimulus_label(info: dict or str) -> int or str or None:
    try:
        img_label = info['img']['label']
    except (KeyError, TypeError):
        return None
    return img_label


@read_json_decorator
def stimulus_split(info: dict or str) -> str or None:
    try:
        img_split = info['img']['split']
    except (KeyError, TypeError):
        return None
    return img_split


@read_json_decorator
def stimulus_dataset(info: dict or str) -> str or None:
    try:
        img_dataset = info['img']['dataset']
    except (KeyError, TypeError):
        return None
    return img_dataset


@read_json_decorator
def stimulus_transforms(info: dict or str) -> list or None:
    try:
        img_transforms = info['img']['transform'].split(', ')
    except (KeyError, TypeError):
        return None
    return img_transforms
