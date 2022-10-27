import os
import sys
import cv2
import time
import numpy as np
from datetime import datetime
from movneye.recording import RecordScene
from movneye.functional.store import store_sensor_info
from movneye.calib.paths import CALIB2D_CAM_FILE, CALIB2D_RECORD_DIR, CALIB2D_RESULT_DIR, SENSOR_FILE


def calibrate2d(n_recs: int, save_recs: bool,
                chess_shape: (int, int), square_length: int,
                config_file: str):

    # Create directory of the images to save (if it does not already exist)
    os.makedirs(CALIB2D_RECORD_DIR, exist_ok=True)

    # Create the object for recording data (communicate with DAVIS)
    myexp = RecordScene(
        # 1) Recording info
        rec_file=None,
        config_file=config_file,
        what2rec='aps'
    )
    davis_shape = myexp.davis_shape
    davis_name = myexp.davis_name + str(davis_shape[0])
    davis_serial = myexp.davis_serial_number
    if not os.path.isfile(SENSOR_FILE):
        store_sensor_info(resolution=davis_shape, model=davis_name, serial=davis_serial)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(chess_shape[0]-1,chess_shape[1]-1,0)
    objp = np.zeros((np.prod(chess_shape), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_shape[0], 0:chess_shape[1]].T.reshape(-1, 2)
    objp *= square_length

    # Loop for recording multiple views of the chessboard
    objpoints, imgpoints = [], []
    for k in range(n_recs):
        print(f'\n-Frame {k+1}')

        # Give me some time to reposition the physical chessboard
        time.sleep(0.5)

        # Take the snapshot
        shot = myexp.shot_aps()
        if shot is None:  # Try once again!
            shot = myexp.shot_aps()
            if shot is None:
                print('No image acquired.')
                continue
        print('Snapshot taken!')

        if save_recs:
            # Define the name of the file to save
            file2save = os.path.join(CALIB2D_RECORD_DIR, '%.3d' % (k + 1)) + '.jpg'
            if os.path.isfile(file2save):
                print('!!! CAUTION !!! The file to save already exists and will be over-written!', file=sys.stderr)
            # Log data
            cv2.imwrite(file2save, shot)

        # Find the chess board corners
        found, corners = cv2.findChessboardCorners(shot, chess_shape, None)

        # If found, add object points, image points (after refining them)
        if found:
            corners = cv2.cornerSubPix(shot, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(shot, chess_shape, corners, found)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Close communication with device and exit
    myexp.close()

    # Find camera matrix and distortion coefficients
    found, cam_mtx, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, davis_shape, None, None)

    # Store the resulting camera matrix and distortion coefficients found
    if found:
        # Create directory of the results to save (if it does not already exist)
        os.makedirs(CALIB2D_RESULT_DIR, exist_ok=True)

        # Define the structure of the xml file to save and store results in there
        with open(CALIB2D_CAM_FILE, 'w+') as writer:
            calib_structure = \
                '<?xml version="1.0"?>\n' \
                '<opencv_storage>\n' \
                f'<{davis_name.upper()}_{davis_serial}>\n' \
                '  <camera_matrix type_id="opencv-matrix">\n' \
                f'    <rows>{cam_mtx.shape[0]}</rows>\n' \
                f'    <cols>{cam_mtx.shape[1]}</cols>\n' \
                '    <dt>d</dt>\n' \
                '    <data>\n' \
                f'      {np.array2string(cam_mtx[0], separator=" ")[1:-1]}\n' \
                f'      {np.array2string(cam_mtx[1], separator=" ")[1:-1]}\n' \
                f'      {np.array2string(cam_mtx[2], separator=" ")[1:-1]}</data></camera_matrix>\n' \
                '  <distortion_coefficients type_id="opencv-matrix">\n' \
                '    <rows>5</rows>\n' \
                '    <cols>1</cols>\n' \
                '    <dt>d</dt>\n' \
                '    <data>\n' \
                f'      {np.array2string(dist_coeff.ravel(), separator=" ")[1:-1]}</data></distortion_coefficients>\n' \
                f'  <image_width>{davis_shape[0]}</image_width>\n' \
                f'  <image_height>{davis_shape[1]}</image_height></{davis_name.upper()}_{davis_serial}>\n' \
                '<use_fisheye_model>0</use_fisheye_model>\n' \
                '<type>camera</type>\n' \
                f'<pattern_width>{chess_shape[0]}</pattern_width>\n' \
                f'<pattern_height>{chess_shape[1]}</pattern_height>\n' \
                '<pattern_type>chessboard</pattern_type>\n' \
                f'<board_width>{chess_shape[0]+1}</board_width>\n' \
                f'<board_height>{chess_shape[1]+1}</board_height>\n' \
                f'<square_size_mm>{square_length}</square_size_mm>\n' \
                f'<calibration_error>{None}</calibration_error>\n' \
                f'<calibration_time>"{datetime.now().strftime("%a %b  %d %H:%M:%S %Y")}"</calibration_time>\n' \
                '</opencv_storage>\n' \
                ''
            writer.write(calib_structure)
        print(f'\nThe camera was successfully calibrated: the resulting parameters have been saved in '
              f'"{CALIB2D_CAM_FILE}".')
    else:
        raise Exception('The camera could not be calibrated. Calibration parameters not found.')
