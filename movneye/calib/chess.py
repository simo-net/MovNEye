import os
import cv2
import numpy as np
from multiprocessing import Process
from movneye.calib.paths import INFO_DIR
from movneye.functional.store import store_monitor_info, store_chessboard_info
from movneye.utils.add2cv import draw_corners, compute_monitor_info


def create_chess(resolution: (int, int), shape: (int, int), square: int, border: (int, int)) -> np.ndarray:
    alternate = [list(map(lambda k: (k + 1) % 2 == 0, range(s))) for s in shape]
    chess = np.ones(resolution[::-1], dtype=bool)
    x = border[0]
    for kx in range(shape[0]):
        shift_y = square * alternate[0][kx]
        y = border[1]
        for ky in range(shape[1]):
            shift_x = square * alternate[1][ky]
            x_start, y_start = x + shift_x, y + shift_y
            chess[y_start: y_start + square, x_start: x_start + square] = False
            y += square
        x += square
    chess[resolution[1] - border[1]:, :] = True
    chess[:, resolution[0] - border[0]:] = True
    return np.uint8(chess * 255)


def adapt_chess2screen(resolution: (int, int), shape: (int, int), border: (int, int)) -> int:
    checker = tuple([(r - 2 * b) / s for r, s, b in zip(resolution, shape, border)])
    assert checker[0] == int(checker[0]) and checker[1] == int(checker[1]), \
        'The given chessboard (shape and border) does not match the resolution of the monitor!'
    checker = tuple([int(round(c)) for c in checker])
    assert checker[0] == checker[1], 'The checker in the chessboard must be a square!'
    return checker[0]


def chess4screen(resolution: (int, int), diagonal: float,     # monitor info
                 shape: (int, int), border: (int, int)):      # chessboard info

    # Compute more monitor info
    (width, height), ppi = compute_monitor_info(resolution, diagonal)

    # Compute more chessboard info
    square = adapt_chess2screen(resolution, shape, border)

    # Create the chessboard
    chess = create_chess(resolution, shape, square, border)

    # Print some information of the chessboard
    print(f'\n- The monitor has resolution {resolution[0]}x{resolution[1]} and dimensions '
          f'{int(round(width * 2.54))}x{int(round(height * 2.54))} cm, thus a PPI of {round(ppi, 3)}.')
    print(f'- The chessboard has {shape[0]}x{shape[1]} b/w squares, each one of length '
          f'{round(square * 2.554 / ppi)} cm.\n')

    # Find the corners
    found, corners = cv2.findChessboardCorners(chess, tuple([s - 1 for s in shape]), None)
    if found:
        criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(chess, corners, (11, 11), (-1, -1), criteria_subpix).reshape((-1, 2))

        # Visualize them
        p = Process(target=draw_corners, args=(chess, corners, 10))
        p.start()
        p.join()

        # Store the chessboard (in .npy file) and its information (in .json file)
        os.makedirs(INFO_DIR, exist_ok=True)
        _ = store_monitor_info(resolution, diagonal, ppi)
        chess_info = store_chessboard_info(shape, square, border)
        np.save(chess_info['image'], chess)
        np.save(chess_info['corners'], corners)
        print(f"\nThe chessboard and its corners' positions were successfully saved in '{INFO_DIR}'.")
    else:
        raise Exception('Corners not found. Nothing was done.')
