import os
import cv2
import numpy as np
from tkinter import Tk


def draw_corners(im: np.ndarray, corner_pts: np.ndarray, radius: int = 3) -> np.ndarray:
    if len(im.shape) < 3:
        new_img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        new_img = np.copy(im)
    for corn in corner_pts.reshape((-1, 2)):
        new_img = cv2.circle(new_img, tuple(np.round(corn).astype(int)), radius, (0, 0, 255), -1)
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('image', new_img)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
    return np.array(new_img)


def draw_axis(im: np.ndarray, origin: np.ndarray, space_pts: np.ndarray) -> np.ndarray:
    orig = tuple(np.round(origin.ravel()).astype(int))[:2]
    new_img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cols = [(0, 0, 255),  # x = Red
            (0, 255, 0),  # y = Green
            (255, 0, 0)]  # z = Blue
    for k in range(3):
        target_pt = tuple(space_pts[k].ravel().astype(int))
        new_img = cv2.line(new_img, orig, target_pt, cols[k], 2)
    cv2.namedWindow('3D axis', cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty('3D axis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('3D axis', new_img)
    cv2.waitKey(0)
    cv2.destroyWindow('3D axis')
    return np.array(new_img)


def draw_img(img: np.ndarray, duration: float = 0, window_position: (int, int) = (0, 0), full_screen: bool = True):
    # Create the window where the image will be displayed in full screen
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow("image", *window_position)
    if full_screen:
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Display the image
    cv2.imshow("image", img)

    # Display it for some time, then stop and destroy the window
    cv2.waitKey(int(duration))  # waiting for a key to be pressed (if 0) or for some time to pass
    cv2.destroyWindow("image")


def load_calibration(calib_file: str, model: str = 'DAVIS346', serial: str = '00000336') -> (np.ndarray, np.ndarray):
    """Load open-cv's .xml calibration file of the neuromorphic camera, where the camera matrix and distortion
    coefficients are stored. The camera matrix keeps the intrinsic parameters of the camera, thus once calculated
    it can be stored for future purposes. It includes information like focal length (f_x, f_y), optical
    centers (c_x, c_y), etc (it takes the form of a 3x3 matrix like [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).
    Distortion coefficients are 4, 5, or 8 elements (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]]); if the vector is
    NULL/empty, zero distortion coefficients are assumed.
    Args:
        calib_file (str): the full path of the xml file.
        model (str): the model of the neuromorphic sensor.
        serial (str): the serial number of the sensor.
    """
    assert os.path.isfile(calib_file), f'Yuo must run the 2D camera calibration in order to undistort the recordings.' \
                                       f'\n{calib_file} is not a valid file!'
    assert os.path.splitext(calib_file)[-1] == '.xml', 'Type of calibration file must be xml.'
    file_read = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    cam_tree = file_read.getNode(model + '_' + serial)
    camera_mtx = np.array(cam_tree.getNode('camera_matrix').mat())
    distortion_coeffs = np.array(cam_tree.getNode('distortion_coefficients').mat()).T
    # width, height = int(cam_tree.getNode('image_width').real()), int(cam_tree.getNode('image_height').real())
    file_read.release()
    return camera_mtx, distortion_coeffs


def undistortion_maps(camera_shape: (int, int),
                      camera_mtx: np.ndarray, distortion_coeffs: np.ndarray) -> (np.ndarray, np.ndarray):
    """Compute the undistortion maps from camera calibration parameters."""
    map1, map2 = cv2.initUndistortRectifyMap(camera_mtx, distortion_coeffs, R=np.eye(3, dtype=int),
                                             newCameraMatrix=camera_mtx, size=camera_shape, m1type=cv2.CV_32FC1)
    return map1, map2


def remap_frames(data: np.ndarray, map_1: np.ndarray, map_2: np.ndarray, interpolate: bool = False) -> np.ndarray:
    """Remap frames (either single or multi-frame data) for removing radial and tangential lens distortion.
    It takes as input the undistortion maps and updates the frames. We remove info falling on those pixels for which
    there is no correspondence in the original pixel array."""
    if len(data.shape) <= 2:
        single_frame = True
        video = data[None, :, :]
    else:
        single_frame = False
        video = data
    if interpolate:
        dst_video = np.zeros_like(video)
        for k, frame in enumerate(video):
            dst_video[k] = cv2.remap(frame, map_1, map_2, interpolation=cv2.INTER_LINEAR)
    else:
        dst_video = video[:, np.round(map_2).astype(int), np.round(map_1).astype(int)]
    if single_frame:
        return dst_video[0]
    return dst_video


def undistort_frames(data: np.ndarray,
                     calib_file: str, model: str = 'DAVIS346', serial: str = '00000336',
                     interpolate: bool = False):
    """Compute undistortion maps and remap frames accordingly in order to correct lens effects.
    Args:
         data (np.ndarray, required): a single APS frame or multi-frame video sequence from a neuromorphic camera.
         calib_file (str, required): the full path of the xml file where info from camera calibration are stored
            in open-CV format.
         model (str, optional): the model of the neuromorphic sensor.
         serial (str, optional): the serial number of the sensor.
         interpolate (bool, optional): whether to smoothen the result through interpolation.
    """
    if len(data.shape) > 2:
        camera_shape = data.shape[1:][::-1]
    else:
        camera_shape = data.shape[::-1]
    camera_mtx, dist_coeffs = load_calibration(calib_file, model=model, serial=serial)
    return remap_frames(data, *undistortion_maps(camera_shape, camera_mtx, dist_coeffs), interpolate=interpolate)


def undistort_img_points(points: np.ndarray,         # the coordinate points to remap in the undistorted plane
                         camera_shape: (int, int),   # (x, y) shape of the camera
                         camera_mtx: np.ndarray, distortion_coeffs: np.ndarray):  # camera calibration parameters
    map_1, map_2 = undistortion_maps(camera_shape, camera_mtx, distortion_coeffs)
    pts = points.reshape((-1, 2))
    pts_undistorted = np.zeros((pts.shape[0], 2), dtype=int)
    for k, pt in enumerate(pts):
        options = np.where((np.round(map_1) == np.round(pt[0])) & (np.round(map_2) == np.round(pt[1])))
        pts_undistorted[k] = (options[1][0], options[0][0]) if len(options[0]) else (0, 0)
    return pts_undistorted


def get_screen_geometry():
    """
    Get the resolution of the monitor in a single-screen setup.
    """
    root = Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


def get_multi_screen_geometry():
    """
    Workaround to get the resolution, in a multi-screen setup, of both the main screen (first element of the returned
    list) and the summed resolution of the other monitors (second element).
    """
    root = Tk()
    width_tot = root.winfo_screenwidth()
    height_tot = root.winfo_screenheight()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    geometry = root.winfo_geometry()
    root.destroy()
    width_main, height_main = int(geometry.split('x')[0]), int(geometry.split('x')[1].split('+')[0])
    width_others, height_others = (width_tot - width_main), height_tot
    return [(width_main, height_main), (width_others, height_others)]


def compute_monitor_info(resolution: (int, int), diagonal: float) -> ((int, int), float):
    ratio = resolution[0] / resolution[1]  # x / y
    height = np.sqrt(diagonal ** 2 / (1 + ratio ** 2))
    width = ratio * height
    ppi = resolution[0] / width
    return (width, height), ppi


def compute_img_border(image_resolution: (int, int), screen_resolution: (int, int),
                       border_ratio: float = None) -> (int, int):
    """
    Compute the black border surrounding an image when displayed on a monitor in full-screen mode (if border_ratio is
    None, else with a predefined percentage of border surrounding the image). In both cases, the aspect-ratio of the
    original image is preserved.
    Both image and screen resolutions are supposed to be in the form (width, height).
    The output will be (x_border, y_border).
    """
    mon_ratio = screen_resolution[0] / screen_resolution[1]
    img_ratio = image_resolution[0] / image_resolution[1]
    if mon_ratio > img_ratio:
        scale_factor = screen_resolution[1] / image_resolution[1]
    else:
        scale_factor = screen_resolution[0] / image_resolution[0]
    img_resolution = tuple([int(image_resolution[k] * scale_factor) for k in range(2)])
    img_border = tuple([int((screen_resolution[k] - img_resolution[k]) / 2) for k in range(2)])
    if border_ratio is not None and 0 <= border_ratio <= 1:
        img_border = tuple([int(img_resolution[k] * border_ratio / 2 + img_border[k]) for k in range(2)])
    # img_border_cm = tuple([s/ppi*2.54 for s in img_border])  # in cm
    return img_border


def adapt_img2screen(image: np.ndarray, border_img: float or None = 0.3, border_color: (int, int, int) or str = 'infer',
                     monitor_resolution: (int, int) = (1920, 1080), return_resolution: bool = False) -> np.ndarray:
    if image.dtype != np.uint8:
        'Image data-type must be 8-bit unsigned integer.'

    # Define the color of the border surrounding the image
    if isinstance(border_color, str):
        assert border_color == 'infer',\
            'The color of the border must be a tuple of 3 integers or, if you want to adapt it to the image,\n' \
            'set it equal to "infer" so that it will be equal to the median color in the image.'
    border_col = np.uint8(np.median(image, axis=(0, 1))) if border_color == 'infer' \
        else np.uint8(border_color)

    # Adapt image to monitor: note that the final image being returned has the same shape as monitor_resolution (i.e.
    # the resolution of the monitor where will be displayed) while the aspect-ratio of the original image is preserved
    img_border = compute_img_border(image.shape[:2][::-1], monitor_resolution, border_ratio=border_img)
    img_resolution = tuple([int(round(monitor_resolution[k] - 2 * img_border[k])) for k in range(2)])
    if img_resolution == (0, 0):
        return np.ones((*monitor_resolution[::-1], (3 if len(image.shape) > 2 else 1)), dtype=np.uint8) * border_col
    img = cv2.resize(image, img_resolution, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(src=img,
                             top=img_border[1], bottom=img_border[1],
                             left=img_border[0], right=img_border[0],
                             borderType=cv2.BORDER_CONSTANT, value=border_col.tolist())
    if return_resolution:
        return img, img_resolution
    return img


def preprocess_image(img: np.ndarray,
                     transforms: list or None = None) -> np.ndarray:
    preprocessing_functions = {
        'bgr2rgb': lambda im: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        'bgr2gray': lambda im: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        'rgb2gray': lambda im: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
        'negative': lambda im: 255 - im,
    }
    if transforms:
        assert all(key in preprocessing_functions.keys() for key in transforms)
        for key in transforms:
            func = preprocessing_functions.get(key)
            img = func(img)
    return np.uint8(img)
