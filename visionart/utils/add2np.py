import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates


def crop(img: np.ndarray, shape: (int, int)) -> np.ndarray:
    start_x = img.shape[1] // 2 - shape[1] // 2
    start_y = img.shape[0] // 2 - shape[0] // 2
    return img[start_y:start_y + shape[0], start_x:start_x + shape[1]]


def normalize(func: np.ndarray, minimum: float = 0, maximum: float = 1,
              func_min: float = None, func_max: float = None) -> np.ndarray:
    func_min = func.min() if not func_min else func_min
    func_max = func.max() if not func_max else func_max
    if (func_max - func_min) == 0:
        return func
    func_norm = (func - func_min) / (func_max - func_min) * (maximum - minimum) + minimum
    return func_norm


def to_uint8(dat: np.ndarray, max_val: float = None, min_val: float = 0) -> np.ndarray:
    dat_norm = normalize(dat, minimum=0., maximum=255., func_min=min_val, func_max=max_val)
    return np.uint8(np.round(dat_norm))


def gauss2d(space_dim: (int, int), sigma: (float, float), center: (float, float) or None = None) -> np.ndarray:
    """Summary: function that calculates elongated gaussian 2D kernel.

    Args:
        space_dim (tuple or int): number of rows (y), cols (x) in 2d array of neurons.
        center (tuple): center of kernel.
        sigma (tuple): sigma (SD) of kernel.
    Returns:
        kernel (float 1d-array): value of kernel that can be set as a weight
    """
    if not center:
        center = (space_dim[0] // 2, space_dim[1] // 2)
    [x, y] = np.meshgrid(range(space_dim[0]), range(space_dim[1]))
    kernel = np.exp(-(((x - center[1]) / sigma[1]) ** 2 + ((y - center[0]) / sigma[0]) ** 2) / 2)
    return kernel / kernel.sum()


def gaussian_smoothing(array: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth 1D data through convolution with a Gaussian filter."""
    gauss = np.exp(-(np.arange(-4 * window, 4 * window) / window) ** 2 / 2)
    return normalize(np.convolve(array, gauss, mode='same'), minimum=array.min(), maximum=array.max())


def activity_interval(array: np.ndarray, samples_plateau: int = 20, std_tolerance: int = 4,
                      margin: int = None) -> (int, int):
    """Find first and last samples of higher activity wrt a plateau value. A plateau is supposed to be present both at
     the beginning and at the end of the given array with same extension."""
    plateau = np.append(array[:samples_plateau], array[-samples_plateau:])
    threshold = float(plateau.mean() + std_tolerance * plateau.std())
    first = np.argmax(array >= threshold)
    last = len(array) - np.argmax(array[first:][::-1] >= threshold)
    if margin is not None:
        first = 0 if first < margin else first - margin
        last = len(array) - 1 if last > len(array) - margin - 1 else last + margin
    return first, last


def cartesian2polar(x: np.ndarray, y: np.ndarray, grid: np.ndarray, rho: np.ndarray, theta: np.ndarray,
                    order: int = 3) -> np.ndarray:
    """Given a grid in cartesian coordinates (x, y), this function returns a grid in polar coordinates (rho, theta)
    with the given range."""
    rho_map, theta_map = np.meshgrid(rho, theta)
    new_x = rho_map * np.cos(theta_map)
    new_y = rho_map * np.sin(theta_map)
    ix = interp1d(x, np.arange(len(x)), bounds_error=False, fill_value="extrapolate")
    iy = interp1d(y, np.arange(len(y)), bounds_error=False, fill_value="extrapolate")
    return map_coordinates(grid, np.array([ix(new_x.ravel()), iy(new_y.ravel())]), order=order).reshape(new_x.shape)


def azimuthal_avg(image: np.ndarray, n_angles: int = 24, interp_order: int = 3) -> np.ndarray:
    """Calculate the azimuthal averaged radial profile of a 2D image. It takes the circular average of the
    2D array, resulting in a 1d function that represents the mean at each radial distance."""
    # Define original cartesian grid
    y, x = [np.arange(n) - n//2 for n in image.shape]
    # Remap the image on the polar grid (interpolate cartesian to polar grid with given order) and take its mean profile
    # along all the directions defined by 'theta'
    theta = np.linspace(0, np.pi, n_angles, endpoint=False)
    avg_profile = cartesian2polar(x=x, y=y, grid=image, rho=x, theta=theta, order=interp_order).mean(axis=0)
    return avg_profile
