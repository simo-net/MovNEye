import sys
from multiprocessing import Process
from movneye.sensor import aps
from movneye.recording import RecordImage
from movneye.functional.store import store_sensor_info, store_stimcalib_info
from movneye.functional.read import sensor_info, monitor_info, chessboard_info, recording_info, monitor_resolution, chessboard_file
from movneye.utils.add2np import normalize
from movneye.utils.add2cv import *
from movneye.calib.paths import *


def record_aps(rec_file: str, config_file: str, img_file: str = None, border_img: float = None):
    if os.path.isfile(rec_file):
        print('!!! CAUTION !!! The file to save already exists and will be over-written!', file=sys.stderr)

    # Define parameters for APS recordings of the chessboard on the monitor
    record_duration = 1e3  # ms
    record_timeout = 2e3   # ms
    img = np.zeros((500, 500), dtype=np.uint8) if (border_img is not None and img_file is None) else \
        np.uint8(normalize(np.load(img_file), 0, 255))

    # Create the object for recording data (communicate with both PTU and DAVIS)
    myexp = RecordImage(
        # 1) Recording info
        rec_file=rec_file,
        duration_rec=record_duration, timeout_rec=record_timeout,
        config_file=config_file, what2rec='aps',
        # 2) Stimulus info
        img_file=img_file, img=img, img_transforms=[],
        duration_img=None, second_monitor=True, monitor_resolution=monitor_resolution(SCREEN_FILE),
        border_img=border_img, border_img_color=(255, 255, 255),
    )

    if not os.path.isfile(SENSOR_FILE):
        davis_shape = myexp.davis_shape
        davis_name = myexp.davis_name + str(davis_shape[0])
        davis_serial = myexp.davis_serial_number
        store_sensor_info(resolution=davis_shape, model=davis_name, serial=davis_serial)

    # Run the experiment (recording)
    myexp.run()

    # Close communication with devices and exit
    myexp.close()


def load_aps_median(file_aps: str, frame_number: int = None, calib_file: str = None) -> np.ndarray:
    with aps.handler(reset_timestamps=False) as aps_handler:
        aps_handler.load_file(file_aps)
        if calib_file is not None:
            aps_handler.undistort(calib_file)
        if frame_number is None:
            image = np.round(np.median(aps_handler.data, axis=0)).astype(np.uint8)
        else:
            image = aps_handler.select_frame(frame_number)
    return image


def calibrate3d(config_file: str):

    # Record the chess stimulus (take some APS camera frames of it)
    os.makedirs(CALIB3D_RECORD_DIR, exist_ok=True)
    record_aps(rec_file=CALIB3D_RECchess_FILE, config_file=config_file,
               img_file=chessboard_file(CHESS_FILE), border_img=None)
    # Note: the chess must be displayed in full screen mode (no border surrounding the chessboard image)

    # Load chess and monitor information
    screen_resolution, _, _ = monitor_info(SCREEN_FILE)
    chess_shape, _, square_dim, _, corners_3d = chessboard_info(CHESS_FILE)
    origin = -np.copy(corners_3d[0])  # define origin on the monitor (first corner of the chessboard)
    corners_3d += origin

    # Load sensor information
    _, model, serial = sensor_info(SENSOR_FILE)

    # Load a single frame from APS recording of the chessboard
    aps_file, _, _, _ = recording_info(CALIB3D_RECchess_FILE)
    camera_img = load_aps_median(aps_file)
    # camera_img_undistorted = undistort_frames(camera_img, calib_file=CALIB2D_CAM_FILE, interpolate=True)
    # camera_resolution = camera_img.shape[::-1]

    # Load camera calibration info
    cam_mtx, dist_coeffs = load_calibration(CALIB2D_CAM_FILE, model=model, serial=serial)

    # Search for chessboard grid (internal point corners of its squares)
    found_pts, corners_2d_camera = cv2.findChessboardCorners(camera_img, tuple([v-1 for v in chess_shape]), None)

    if found_pts:
        # Refine it with sub-pixel precision
        corners_2d_camera = cv2.cornerSubPix(camera_img, corners_2d_camera, (11, 11), (-1, -1),
                                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                                             ).reshape((-1, 2))

        # Visualize them
        draw_corners(camera_img, corners_2d_camera, 3)
        # p = Process(target=draw_corners, args=(camera_img, corners_2d_camera, 3))
        # p.start()
        # p.join()

        # Find the rotation and translations vectors between the actual point
        found_p3p, rvecs, tvecs = cv2.solvePnP(corners_3d, corners_2d_camera, cam_mtx, dist_coeffs)

        if found_p3p:
            # Project 3D points to image plane (on the origin, which is the first corner of the chessboard)
            cam_origin_corner = corners_2d_camera[0]
            axis = np.float32([[square_dim, 0, 0],
                               [0, square_dim, 0],
                               [0, 0, square_dim]])  # points in 3D space for axes with same length as the chess squares
            axs_pts, _ = cv2.projectPoints(axis, rvecs, tvecs, cam_mtx, dist_coeffs)
            draw_axis(camera_img, cam_origin_corner, axs_pts)
            # p = Process(target=draw_axis, args=(camera_img, cam_origin_corner, axs_pts))
            # p.start()
            # p.join()

            # Now put the axis on the origin of the monitor (top left) instead than on the first corner
            cam_origin_monitor, _ = cv2.projectPoints(origin, rvecs, tvecs, cam_mtx, dist_coeffs)
            new_axis = axis + np.float32(origin)
            new_axs_pts, _ = cv2.projectPoints(new_axis, rvecs, tvecs, cam_mtx, dist_coeffs)
            draw_axis(camera_img, cam_origin_monitor, new_axs_pts)
            # p = Process(target=draw_axis, args=(camera_img, cam_origin_monitor, new_axs_pts))
            # p.start()
            # p.join()

            # Save the results of 3D camera-monitor calibration
            os.makedirs(CALIB3D_RESULT_DIR, exist_ok=True)
            np.save(CALIB3D_ROTATION_FILE, rvecs)
            np.save(CALIB3D_TRANSLATION_FILE, tvecs)
            print(f'\nRotation and translation vectors were successfully saved in "{CALIB3D_RESULT_DIR}".')
        else:
            raise Exception('\nRotation and translation vectors not found!')

    else:
        raise Exception('Chessboard corners not found!')


def find_monitor_roi(show_results: bool = True, store_results: bool = True):

    # Load monitor and chess information used in 3D camera-screen calibration
    screen_resolution, _, _ = monitor_info(SCREEN_FILE)
    _, _, _, _, corners_3d = chessboard_info(CHESS_FILE)
    origin = -corners_3d[0]  # define origin on the monitor (first corner of the chessboard)

    # Load sensor information
    _, model, serial = sensor_info(SENSOR_FILE)

    # Load a single frame from APS recording of the chessboard image
    aps_file, _, _, _ = recording_info(CALIB3D_RECchess_FILE)
    camera_img = load_aps_median(aps_file)
    camera_img_undistorted = undistort_frames(camera_img, calib_file=CALIB2D_CAM_FILE, model=model, serial=serial,
                                              interpolate=True)
    camera_resolution = camera_img.shape[::-1]

    # Load camera calibration info
    cam_mtx, dist_coeffs = load_calibration(CALIB2D_CAM_FILE, model=model, serial=serial)

    # Load rotation and translation vectors
    rvecs = np.load(CALIB3D_ROTATION_FILE)
    tvecs = np.load(CALIB3D_TRANSLATION_FILE)

    # Find the 4 points on the sensor array corresponding to the 4 corners of the full monitor screen
    monitor_bounds = origin + np.float32([
        (0, 0, 0),  # TOP-LEFT
        (screen_resolution[0], 0, 0),  # TOP-RIGHT
        (0, screen_resolution[1], 0),  # BOTTOM-LEFT
        (screen_resolution[0], screen_resolution[1], 0)  # BOTTOM-RIGHT
    ])
    camera_monbounds, _ = cv2.projectPoints(monitor_bounds, rvecs, tvecs, cam_mtx, dist_coeffs)

    if show_results:
        draw_corners(camera_img, camera_monbounds, 3)
        # p = Process(target=draw_corners, args=(camera_img, camera_monbounds, 3))
        # p.start()
        # p.join()
        # Also correct lens distortion in both the image and the relative boundary (for visualization only)
        cam_monbounds_undistorted = undistort_img_points(camera_monbounds, camera_resolution,
                                                         cam_mtx, dist_coeffs)
        draw_corners(camera_img_undistorted, cam_monbounds_undistorted, 3)
        # p = Process(target=draw_corners, args=(camera_img_undistorted, cam_monbounds_undistorted, 3))
        # p.start()
        # p.join()

    # Find the field of view
    fov_camscreen = np.array((float(2 * np.abs(np.arctan2(screen_resolution[0] / 2, tvecs[2]) * 180 / np.pi)),
                              float(2 * np.abs(np.arctan2(screen_resolution[1] / 2, tvecs[2]) * 180 / np.pi))))
    print(f'\nThe field of view of the whole monitor screen projected on the camera is '
          f'{round(fov_camscreen[0])}° on X and {round(fov_camscreen[1])}° on Y.')

    # Store results
    if store_results:
        np.save(CALIB3D_ROImon_FILE, camera_monbounds)
        np.save(CALIB3D_FOVmon_FILE, fov_camscreen)
    print(f'\nMonitor boundaries on the camera matrix and relative FOV were successfully saved in '
          f'"{CALIB3D_RESULT_DIR}".')

    return camera_monbounds


def find_image_roi(recorded_frames_file: str = None, config_file: str = None,
                   border_img: float = None, img_resolution: (int, int) = None, img_strinfo: str = None,
                   show_results: bool = True, store_results: bool = True):

    if recorded_frames_file is None:
        # Store information on the sample image to use to find the 4 corners
        if store_results:
            img_info = store_stimcalib_info(img_resolution, border_img, img_strinfo)
            np.save(img_info['image'], np.zeros(img_resolution[::-1], dtype=np.uint8))

        # Record the sample image stimulus (take some APS camera frames of it)
        record_aps(rec_file=CALIB3D_RECimg_FILE, config_file=config_file,
                   img_file=chessboard_file(IMG_FILE), border_img=border_img)
        recorded_frames_file, _, _, _ = recording_info(CALIB3D_RECimg_FILE)

    # Load monitor and chess information used in 3D camera-screen calibration
    screen_resolution, _, _ = monitor_info(SCREEN_FILE)
    _, _, _, _, corners_3d = chessboard_info(CHESS_FILE)
    origin = -corners_3d[0]  # define origin on the monitor (first corner of the chessboard)

    # Load sensor information
    _, model, serial = sensor_info(SENSOR_FILE)

    # Load a single frame from APS recording of the given image
    camera_img = load_aps_median(recorded_frames_file)
    camera_resolution = camera_img.shape[::-1]

    # Load camera calibration info
    cam_mtx, dist_coeffs = load_calibration(CALIB2D_CAM_FILE, model=model, serial=serial)

    # Load rotation and translation vectors
    rvecs = np.load(CALIB3D_ROTATION_FILE)
    tvecs = np.load(CALIB3D_TRANSLATION_FILE)

    # Find the 4 points on the sensor array corresponding to the 4 corners of the image on the monitor screen
    mon_imgborder = compute_img_border(img_resolution, screen_resolution, border_ratio=border_img)
    mon_imgshape = tuple([int(round(screen_resolution[k] - 2 * mon_imgborder[k])) for k in range(2)])
    mon_imgbounds = origin + np.float32([*mon_imgborder, 0]) + np.float32([
        (0, 0, 0),                              # TOP-LEFT
        (mon_imgshape[0], 0, 0),                # TOP-RIGHT
        (0, mon_imgshape[1], 0),                # BOTTOM-LEFT
        (mon_imgshape[0], mon_imgshape[1], 0)   # BOTTOM-RIGHT
    ])
    cam_imgbounds, _ = cv2.projectPoints(mon_imgbounds, rvecs, tvecs, cam_mtx, dist_coeffs)

    if show_results:
        draw_corners(camera_img, cam_imgbounds, 3)
        # p = Process(target=draw_corners, args=(camera_img, cam_imgbounds, 3))
        # p.start()
        # p.join()

        # Also correct lens distortion (undistort) in both the image and the relative boundary (for visualization only)
        camera_img_undistorted = undistort_frames(camera_img, calib_file=CALIB2D_CAM_FILE, model=model, serial=serial,
                                                  interpolate=True)
        cam_imgbounds_undistorted = undistort_img_points(cam_imgbounds, camera_resolution,
                                                         cam_mtx, dist_coeffs)
        draw_corners(camera_img_undistorted, cam_imgbounds_undistorted, 3)
        # p = Process(target=draw_corners, args=(camera_img_undistorted, cam_imgbounds_undistorted, 3))
        # p.start()
        # p.join()

    # Find the field of view
    fov_camimg = np.array((float(2 * np.abs(np.arctan2(mon_imgshape[0] / 2, tvecs[2]) * 180 / np.pi)),
                           float(2 * np.abs(np.arctan2(mon_imgshape[1] / 2, tvecs[2]) * 180 / np.pi))))
    print(f"\nThe whole IMAGE falls on a {round(fov_camimg[0])}° by {round(fov_camimg[1])}° FOV "
          f"(x by y field of view) on the sensor.")

    # Save the result (i.e. the coordinates of the 4 corners of the image on the camera matrix plane)
    if store_results:
        np.save(CALIB3D_ROIimg_FILE, cam_imgbounds)
        np.save(CALIB3D_FOVimg_FILE, fov_camimg)
        print(f'\nImage boundaries on the camera matrix and relative FOV were successfully saved in '
              f'"{CALIB3D_RESULT_DIR}".')

    return cam_imgbounds


def find_image_roi_with_margin(margin: float, recorded_frames_file: str = None,
                               show_results: bool = True):
    # Load the image ROI and FOV on the camera matrix
    cam_imgbounds = np.load(CALIB3D_ROIimg_FILE)
    roi_camimg = compute_roi_from_bounds(cam_imgbounds)
    fov_camimg = np.load(CALIB3D_FOVimg_FILE)

    # Load monitor and chess information used in 3D camera-screen calibration
    screen_resolution, _, _ = monitor_info(SCREEN_FILE)
    _, _, _, _, corners_3d = chessboard_info(CHESS_FILE)
    origin = -corners_3d[0]  # define origin on the monitor (first corner of the chessboard)

    # Load sensor information
    _, model, serial = sensor_info(SENSOR_FILE)

    # Load a single frame from APS recording of the sample image
    if recorded_frames_file is None:
        recorded_frames_file, _, _, _ = recording_info(CALIB3D_RECimg_FILE)
    camera_img = load_aps_median(recorded_frames_file)
    camera_resolution = camera_img.shape[::-1]

    # Load camera calibration info
    cam_mtx, dist_coeffs = load_calibration(CALIB2D_CAM_FILE, model=model, serial=serial)

    # Load rotation and translation vectors
    rvecs = np.load(CALIB3D_ROTATION_FILE)
    tvecs = np.load(CALIB3D_TRANSLATION_FILE)

    # Apply the margin on the FOV of the image to find the resulting shape of the image -with the margin- on the monitor
    mon_imgshape_withmargin = (int(2 * np.abs(np.tan((fov_camimg[0] + margin) * np.pi / 180 / 2)) * tvecs[2]),
                               int(2 * np.abs(np.tan((fov_camimg[1] + margin) * np.pi / 180 / 2)) * tvecs[2]))

    # Now find the 4 points on the sensor array corresponding to the 4 corners of the image on the monitor screen
    # (with the margin surrounding the image)
    mon_imgborder_withmargin = tuple([int((screen_resolution[k] - mon_imgshape_withmargin[k]) / 2) for k in range(2)])
    mon_imgbounds_withmargin = origin + np.float32([*mon_imgborder_withmargin, 0]) + np.float32([
        (0, 0, 0),                                                    # TOP-LEFT
        (mon_imgshape_withmargin[0], 0, 0),                           # TOP-RIGHT
        (0, mon_imgshape_withmargin[1], 0),                           # BOTTOM-LEFT
        (mon_imgshape_withmargin[0], mon_imgshape_withmargin[1], 0)   # BOTTOM-RIGHT
    ])
    cam_imgbounds_withmargin, _ = cv2.projectPoints(mon_imgbounds_withmargin, rvecs, tvecs, cam_mtx, dist_coeffs)

    # Adjsut the boundaries in order to have same margin in both dimensions
    cam_imgbounds_withmargin = adjust_bounds_same_xy_margin(cam_imgbounds_withmargin, roi_camimg)

    if show_results:
        draw_corners(camera_img, cam_imgbounds_withmargin, 3)
        # p = Process(target=draw_corners, args=(camera_img, cam_imgbounds_withmargin, 3))
        # p.start()
        # p.join()

        # Correct lens distortion (undistort) in both the image and the relative boundary
        camera_img_undistorted = undistort_frames(camera_img, calib_file=CALIB2D_CAM_FILE, model=model, serial=serial,
                                                  interpolate=True)
        cam_imgbounds_withmargin_undistorted = undistort_img_points(cam_imgbounds_withmargin, camera_resolution,
                                                                    cam_mtx, dist_coeffs)
        draw_corners(camera_img_undistorted, cam_imgbounds_withmargin_undistorted, 3)
        # p = Process(target=draw_corners, args=(camera_img_undistorted, cam_imgbounds_withmargin_undistorted, 3))
        # p.start()
        # p.join()

    return cam_imgbounds_withmargin


def adjust_bounds_same_xy_margin(cam_imgbounds_withmargin: np.ndarray, roi_camimg: list):
    roi_camimg_withmargin = compute_roi_from_bounds(cam_imgbounds_withmargin)
    (x1_raw_new, y1_raw_new), (x2_raw_new, y2_raw_new) = cam_imgbounds_withmargin[[0,3], 0, :]
    (x1_new, y1_new), (x2_new, y2_new) = roi_camimg_withmargin
    (x1, y1), (x2, y2) = roi_camimg
    margin_x_pix, margin_y_pix = (x2_new-x1_new) - (x2-x1), (y2_new-y1_new) - (y2-y1)
    if margin_x_pix != margin_y_pix:
        if margin_x_pix > margin_y_pix:
            if y2_raw_new-int(y2_raw_new) > y1_raw_new-int(y1_raw_new):
                y2_raw_new = round(y1_raw_new + (y2 - y1 + margin_x_pix))
            else:
                y1_raw_new = round(y2_raw_new - (y2 - y1 + margin_x_pix))
        else:  # margin_x < margin_y
            if x2_raw_new-int(x2_raw_new) > x1_raw_new-int(x1_raw_new):
                x2_raw_new = round(x1_raw_new + (x2 - x1 + margin_y_pix))
            else:
                x1_raw_new = round(x2_raw_new - (x2 - x1 + margin_y_pix))
        cam_imgbounds_withmargin[[0, 3], 0, :] = (x1_raw_new, y1_raw_new), (x2_raw_new, y2_raw_new)
    return cam_imgbounds_withmargin


def compute_roi_from_bounds(boundaries: np.ndarray) -> list:
    """The input boundaries must have shape (4,1,2) while the returned ROI will have shape (2,2) where only the
    coordinates of the top-left and bottom-right corners of the ROI are stored as:
    [[top_left[x,y]], [bottom_right[x,y]]]"""
    roi = (np.round(boundaries[[0, 3]].reshape((2, 2))).astype(int)).tolist()
    return roi


def compute_monitor_distance() -> float:
    _, _, ppi = monitor_info(SCREEN_FILE)
    return abs(float(np.load(CALIB3D_TRANSLATION_FILE)[-1] / ppi * 2.54))  # cm


def continuous_image_roi_search(config_file: str,
                                border_img: float = None, img_resolution: (int, int) = None,
                                stimulus_bounds: np.ndarray = None):

    # Load monitor information
    screen_resolution, _, ppi = monitor_info(SCREEN_FILE)
    cam_screen_distance = compute_monitor_distance()

    # Find the 4 points of the image on the monitor screen
    mon_imgborder = compute_img_border(img_resolution, screen_resolution, border_ratio=border_img)
    mon_imgshape = tuple([int(round(screen_resolution[k] - 2 * mon_imgborder[k])) for k in range(2)])

    if not stimulus_bounds:
        # Load chess information used in 3D camera-screen calibration
        _, _, _, _, corners_3d = chessboard_info(CHESS_FILE)
        origin = -corners_3d[0]  # define origin on the monitor (first corner of the chessboard)

        # Load camera calibration info
        cam_mtx, dist_coeffs = load_calibration(CALIB2D_CAM_FILE)

        # Load 3D rotation and translation vectors
        rvecs = np.load(CALIB3D_ROTATION_FILE)
        tvecs = np.load(CALIB3D_TRANSLATION_FILE)

        # Now we can find the 4 points on the sensor array corresponding to the 4 corners of the image on the monitor
        mon_imgbounds = origin + np.float32([*mon_imgborder, 0]) + np.float32([
            (0, 0, 0),  # TOP-LEFT
            (mon_imgshape[0], 0, 0),  # TOP-RIGHT
            (0, mon_imgshape[1], 0),  # BOTTOM-LEFT
            (mon_imgshape[0], mon_imgshape[1], 0)  # BOTTOM-RIGHT
        ])
        cam_imgbounds, _ = cv2.projectPoints(mon_imgbounds, rvecs, tvecs, cam_mtx, dist_coeffs)
    else:
        cam_imgbounds = stimulus_bounds
    roi_img = compute_roi_from_bounds(cam_imgbounds)
    fov_camimg = np.array((float(2 * abs(np.arctan2(mon_imgshape[0] / 2, cam_screen_distance*ppi/2.54) * 180 / np.pi)),
                           float(2 * abs(np.arctan2(mon_imgshape[1] / 2, cam_screen_distance*ppi/2.54) * 180 / np.pi))))
    print(f'\nStarting the continuous search of the image ROI.'
          f'\nYou should adjust the camera/monitor position in order to match the red dots (the ROI) with the borders'
          f'\nof the image displayed on the monitor. To this purpose, the program will keep on displaying frames from'
          f'\nthe DAVIS sensor in real time until a KeyboardInterrupt (CTRL+C) is sent. In this case, the program will'
          f'\nnot raise an error but it will exit from the never-ending while loop.'
          f'\nThe following information are available on the present camera-monitor configuration:'
          f'\n   - according to the 3D calibration results the camera-monitor distance is {round(cam_screen_distance, 1)} cm'
          f'\n   - the image ROI is {roi_img[1][0] - roi_img[0][0]} x {roi_img[1][1] - roi_img[0][1]} and FOV is {round(fov_camimg[0])}° x {round(fov_camimg[1])}°')

    # Create the object for recording data (communicate with DAVIS)
    myexp = RecordImage(
        # 1) Recording info
        rec_file=None,
        noise_filter=False, config_file=config_file,
        what2rec='aps',
        # 2) Stimulus info
        second_monitor=True, monitor_resolution=screen_resolution,
        img=np.zeros(img_resolution[::-1], dtype=np.uint8), img_transforms=[],
        border_img=border_img, border_img_color=(255, 255, 255),
        verbose=False
    )

    # Load and start displaying the image stimulus on a second monitor
    img_stimulus = myexp.img
    second_monitor_position = get_multi_screen_geometry()[0]
    stim_process = Process(target=draw_img, args=(img_stimulus, -1, second_monitor_position, True))
    stim_process.daemon = True
    stim_process.start()

    # Loop for recording multiple views of the image
    cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        try:
            # Take a snapshot from the DAVIS: a single APS frame
            camera_img = myexp.shot_aps()
            if camera_img is None:
                print('skip')
                continue
            print('done')
            camera_img = cv2.cvtColor(camera_img, cv2.COLOR_GRAY2BGR)

            # Add the corner points defining the ROI of the stimulus on top of the frame
            for corn in cam_imgbounds.reshape((-1, 2)):
                camera_img = cv2.circle(camera_img, tuple(np.round(corn).astype(int)), 3, (0, 0, 255), -1)

            # Display the results on the built-in monitor
            cv2.imshow('img', camera_img)
            cv2.waitKey(1)

        except KeyboardInterrupt:
            # End displaying the results
            cv2.destroyWindow('img')
            break

    # Close communication with device
    myexp.close()

    # End displaying the stimulus image on the second monitor
    cv2.destroyAllWindows()
    stim_process.join(timeout=2)
    try:
        stim_process.kill()
    except AttributeError:
        pass
    finally:
        stim_process.terminate()
    # try:
    #     stim_process.close()
    # except AttributeError:
    #     stim_process.terminate()
