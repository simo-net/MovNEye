INFO_DIR = './calib/info'
SENSOR_FILE = INFO_DIR + '/sensor.json'
SCREEN_FILE = INFO_DIR + '/screen.json'
CHESS_FILE = INFO_DIR + '/chess.json'
IMG_FILE = INFO_DIR + '/image.json'

CALIB2D_RECORD_DIR = INFO_DIR.replace('info', 'rec') + '/calib2D'       # './calib/rec/calib2D'
CALIB2D_RESULT_DIR = CALIB2D_RECORD_DIR.replace('rec', 'res')           # './calib/res/calib2D'
CALIB2D_CAM_FILE = CALIB2D_RESULT_DIR + '/lens_distortion.xml'

CALIB3D_RECORD_DIR = CALIB2D_RECORD_DIR.replace('calib2D', 'calib3D')   # './calib/rec/calib3D'
CALIB3D_RECimg_FILE = CALIB3D_RECORD_DIR + '/image.json'
CALIB3D_RECchess_FILE = CALIB3D_RECORD_DIR + '/chess.json'
CALIB3D_RESULT_DIR = CALIB3D_RECORD_DIR.replace('rec', 'res')           # './calib/res/calib3D'
CALIB3D_ROTATION_FILE = CALIB3D_RESULT_DIR + '/rotation.npy'
CALIB3D_TRANSLATION_FILE = CALIB3D_RESULT_DIR + '/translation.npy'
CALIB3D_FOVimg_FILE = CALIB3D_RESULT_DIR + '/img_fov.npy'
CALIB3D_ROIimg_FILE = CALIB3D_RESULT_DIR + '/img_roi.npy'
CALIB3D_FOVmon_FILE = CALIB3D_RESULT_DIR + '/mon_fov.npy'
CALIB3D_ROImon_FILE = CALIB3D_RESULT_DIR + '/mon_roi.npy'
