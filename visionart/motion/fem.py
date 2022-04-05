import os
import numpy as np
import matplotlib.pyplot as plt
from visionart.utils.add2np import azimuthal_avg


class SAW(object):

    def __init__(self,
                 n_burnin: int = 10 ** 4, n_traj: int = 60,
                 max_step: float = 6., foveola: float = 60.,
                 ptu_resolution: float = 46.2857 / 60, grid: int = 301,
                 epsilon: float = 1e-4, lamda: float = .1, chi: float or None = None, hc: float = 2.1,
                 init_burnin: bool = True):
        """
        Extension of the Self-Avoiding random Walk (SAW) in a lattice proposed by Engbert et al. in 2011.
        In the original model, movements are driven by a self-generated activation field and confined in fovea by a
        quadratic potential (convex shaped). As a result, the walker avoids returning to recently visited lattice sites
        and, after many time steps, it reaches regions with high activation and potential values which push it back
        towards the center of the lattice. However, this model assumes that each drift step can only be towards one
        of the 4 neighbouring lattice sites. Doing so, each step is confined along the 2 cardinal directions.
        We extend this range by defining a less constrained neighborhood. Such neighbourhood is now described by a
        circular window with radius equal to the maximum biological step size for a drift movement (i.e. ~6 arcmins).

        For details about the original model and its implications please refer to:
         - Engbert R. et al. "An integrated model of fixational eye movements and microsaccades". Proceedings of the
           National Academy of Sciences (2011): E765-E770.
         - Herrmann C.J., Metzler R. and Engbert R. "A self-avoiding walk with neural delays as a model of fixational
           eye movements". Scientific Reports (2017): 1-17.

        :param n_burnin: number of iterations (steps) considered for the burn-in (best value in original model: 10**4)
        :param n_traj: number of steps actually considered for the FEM
        :param grid: useful grid size for FEM in PTU positions (best value in original model: 51)
        :param epsilon: relaxation factor (best value in original model: 1e-3)
        :param lamda: weight of the quadratic potential U (best value in original model: 1)
        :param chi: weight of the MS potential U1 (best value in original model: 2λ)
        :param hc: activation threshold for Micro-Saccade generation (best value in original model: 7.9-20)
        :param ptu_resolution: [arcmins] pan and tilt step resolution of the PTU
        :param foveola: [arcmins] diameter of the foveola, defining the circular area in which FEM must be confined
        :param max_step: [arcmins] maximum size of a single FEM step
        """

        # Parameters definition
        self.n_burnin = n_burnin
        self.n_traj = n_traj
        self._ptu_resolution = ptu_resolution
        self._foveola = foveola / self._ptu_resolution
        self._max_step = max_step / self._ptu_resolution
        self._grid = grid
        self._lamda = lamda
        self._chi = 2 * lamda if chi is None else chi
        self.epsilon = epsilon
        self.hc = hc

        # Initialize useful (main) variables
        self.i0 = self.j0 = None, None  # origin of the lattice, related to the rostral pole of superior colliculus
        self.J, self.I = None, None  # matrices of all x (J) & y (I) coordinates
        self.U, self.U1 = None, None  # quadratic potential & micro-saccadic potential
        self.init_vars()

        # Initialize burn-in movement variables (if you wish)
        self._H_burnin = None  # activation field after burn-in period
        self._mask_burnin = None  # mask for omitting current position when updating the burn-in activation field decay
        self._traj_burnin = None  # trajectory after the burn-in period
        if init_burnin:
            self.burn_in()

        # Initialize FEM sequence variables
        self._activation_field = None  # activation field of actual FEM
        self._mask = None  # mask for omitting current position when updating the activation field decay
        self._trajectory = None  # trajectory of actual FEM
        self._is_microsaccade = None  # whether or not each FEM step is an MS
        self._hit_activation_field = None  # the value of the activation field hit by each FEM step

    def init_vars(self):
        self.i0 = self.j0 = self._grid // 2
        self.J, self.I = np.meshgrid(range(self._grid), range(self._grid))
        self.U = self._lamda * self._grid * (((self.I - self.i0) / self.i0) ** 2 + ((self.J - self.j0) / self.j0) ** 2)
        self.U1 = self._chi * self._grid * ((self.I - self.i0) / self.i0) ** 2 * ((self.J - self.j0) / self.j0) ** 2

    def burn_in(self) -> (np.ndarray, np.ndarray):
        """This function executes the burn-in period necessary to initialize the activation field for further generating
        a FEM sequence. Note that this function is automatically run by the class constructor (__init__), but can also
        be run before generating a new FEM sequence for initializing the activation field to new random values (thus
        'increasing the randomness' of FEM generation)."""
        # Initialize useful variables
        self._H_burnin = np.random.uniform(size=(self._grid, self._grid))
        self._mask_burnin = np.ones((self._grid, self._grid), dtype=bool)
        i, j = self.i0, self.j0  # initialize first position
        self._traj_burnin = np.zeros((self.n_burnin, 2), dtype=int)  # initialize burn-in trajectory
        is_microsaccade_burnin = np.zeros(self.n_burnin, dtype=bool)  # check when a microsaccade is generated
        # ----------------------------------------------- Burn in period -----------------------------------------------
        for k in range(self.n_burnin):
            # Activation field update
            self._H_burnin[i, j] += 1
            self._mask_burnin[i, j] = False
            self._H_burnin[self._mask_burnin] *= (1 - self.epsilon)
            self._mask_burnin[i, j] = True

            # Find new drift position -> minimize H + U on neighbourhood N(i,j)
            neighbors_x, neighbors_y = self._find_neighbours(center=(i, j))
            drift_potential = self._H_burnin[neighbors_x, neighbors_y] + self.U[neighbors_x, neighbors_y]
            neighbor_idx = np.random.choice(np.where(drift_potential == drift_potential.min())[0])
            i = neighbors_x[neighbor_idx]
            j = neighbors_y[neighbor_idx]

            # Check condition for MS generation
            is_ms = False
            if self._H_burnin[i, j] >= self.hc:
                # Find new MS position -> minimize H + U + U1 on full lattice
                ms_potential = self._H_burnin + self.U + self.U1
                candidates = np.where(ms_potential == ms_potential.min())
                i = np.random.choice(candidates[0])
                j = candidates[1][candidates[0] == i]
                is_ms = True

            # Update trajectory
            self._traj_burnin[k, :] = [i, j]
            is_microsaccade_burnin[k] = is_ms
        return self._traj_burnin, is_microsaccade_burnin

    def generate_fem(self, reinit_burnin: bool = False, verbose: bool = False):
        """
        This function computes a FEM movement with 'n' steps in the trajectory (if n is given, else it uses the class
        variable n_traj). It computes the trajectory variable, the activation_field variable, the is_microsaccade
        variable and the hit_activation_field variable. You can access them as general class variables. Note that if
        this function is run a second time, these variables are not updated but over-written!
        """
        if reinit_burnin or self._H_burnin is None:
            self.burn_in()
        # Initialize useful variables
        self._activation_field = np.copy(self._H_burnin)  # initialize activation field according to the burn-in period
        self._mask = np.copy(self._mask_burnin)  # initialize mask according to the burn-in period
        self._trajectory = np.zeros((self.n_traj + 1, 2), dtype=int)  # initialize final trajectory
        self._is_microsaccade = np.zeros(self.n_traj + 1, dtype=bool)  # check when a microsaccade is generated
        self._hit_activation_field = np.zeros(self.n_traj + 1)  # check the values of the activation field hit by FEM
        # --------------------------------------------- Actual simulation ----------------------------------------------
        # Initialize first position of the FEM trajectory
        i, j = self.i0, self.j0
        self._trajectory[0, :] = [i, j]
        self._hit_activation_field[0] = self._activation_field[i, j]
        # Run the simulation and find all subsequent positions
        for k in range(1, self.n_traj + 1):
            # Activation field update
            self._activation_field[i, j] += 1
            self._mask[i, j] = False
            self._activation_field[self._mask] *= (1 - self.epsilon)
            self._mask[i, j] = True

            # Find new drift position -> minimize H + U on neighbourhood N(i,j)
            neighbors_x, neighbors_y = self._find_neighbours(center=(i, j))
            drift_potential = self._activation_field[neighbors_x, neighbors_y] + self.U[neighbors_x, neighbors_y]
            neighbor_idx = np.random.choice(np.where(drift_potential == drift_potential.min())[0])
            i = neighbors_x[neighbor_idx]
            j = neighbors_y[neighbor_idx]
            self._hit_activation_field[k] = self._activation_field[i, j]

            # Check condition for MS generation
            is_ms = False
            if self._activation_field[i, j] >= self.hc:
                # Find new MS position -> minimize H + U + U1 on full lattice
                ms_potential = self._activation_field + self.U + self.U1
                candidates = np.where(ms_potential == ms_potential.min())
                i = np.random.choice(candidates[0])
                j = candidates[1][candidates[0] == i]
                is_ms = True

            # Update trajectory
            self._trajectory[k, :] = [i, j]
            self._is_microsaccade[k] = is_ms

        if verbose:
            self.print_info()

    def _find_neighbours(self, center: (int, int)) -> np.ndarray:
        dist_from_center = np.round(np.sqrt((self.I - center[0]) ** 2 + (self.J - center[1]) ** 2))
        return np.where(dist_from_center <= self._max_step)

    @property
    def foveola(self) -> float:
        return self._foveola * self._ptu_resolution

    @foveola.setter
    def foveola(self, foveola: float):
        self._foveola = foveola / self._ptu_resolution

    @property
    def max_step(self) -> float:
        return self._max_step * self._ptu_resolution

    @max_step.setter
    def max_step(self, max_step: float):
        self._max_step = max_step / self._ptu_resolution

    @property
    def grid(self) -> int:
        return self._grid

    @grid.setter
    def grid(self, grid: int):
        self._grid = grid
        self.init_vars()

    @property
    def lamda(self) -> float:
        return self._lamda

    @lamda.setter
    def lamda(self, lamda: float):
        self._lamda = lamda
        self._chi = 2 * lamda
        self.init_vars()

    @property
    def chi(self) -> float:
        return self._chi

    @chi.setter
    def chi(self, chi: float):
        self._chi = chi
        self.init_vars()

    @property
    def ptu_resolution(self) -> float:
        return self._ptu_resolution

    @ptu_resolution.setter
    def ptu_resolution(self, ptu_resolution: float):
        self._foveola = self._foveola * self._ptu_resolution / ptu_resolution
        self._max_step = self._max_step * self._ptu_resolution / ptu_resolution
        self._ptu_resolution = ptu_resolution

    @property
    def activation_field_burnin(self) -> np.ndarray:
        return self._H_burnin

    @activation_field_burnin.setter
    def activation_field_burnin(self, new_val: np.ndarray):
        self._H_burnin = new_val

    @property
    def hit_activation_field(self) -> np.ndarray:
        return self._hit_activation_field

    @hit_activation_field.setter
    def hit_activation_field(self, new_val: np.ndarray):
        self._hit_activation_field = new_val

    @property
    def activation_field(self) -> np.ndarray:
        return self._activation_field

    @activation_field.setter
    def activation_field(self, new_val: np.ndarray):
        self._activation_field = new_val

    @property
    def is_microsaccade(self) -> np.ndarray:
        return self._is_microsaccade

    @is_microsaccade.setter
    def is_microsaccade(self, new_val: np.ndarray):
        self._is_microsaccade = new_val

    @property
    def trajectory(self) -> np.ndarray:
        return self._trajectory

    @trajectory.setter
    def trajectory(self, new_val: np.ndarray):
        self._trajectory = new_val
        if self._trajectory[0, :] == (0, 0):
            self._trajectory += (self.i0, self.j0)

    @property
    def steps(self) -> np.ndarray:
        return compute_steps(self._trajectory)

    @steps.setter
    def steps(self, new_val: np.ndarray):
        self._trajectory = compute_path(new_val)
        self._trajectory += (self.i0, self.j0)

    @property
    def angles(self) -> np.ndarray:
        return compute_angles(self.steps, half_circle=False)

    @property
    def steps_sizes(self) -> np.ndarray:
        return compute_stepsizes(self.steps)

    @property
    def maxdistance2origin(self) -> float:
        return maxdistance2origin(self._trajectory)

    @property
    def neighbour_diameter(self) -> float:
        """Diameter of the neighbourhood region where next step can occur, thus half this dimension equals to the actual
        maximum step size (as opposed to ideal value represented by the 'max_step' class variable)."""
        kernel = np.zeros((self._grid, self._grid), dtype=bool)
        kernel[self._find_neighbours(center=(self.i0, self.j0))] = 1
        return len(kernel[kernel[self.i0, :] > 0])

    @property
    def foveola_diameter(self) -> float:
        """Diameter of the foveola as computed from the FEM sequence, i.e. it represents the diameter of the circular
        area in which FEM are actually confined (as opposed to ideal value represented by the 'foveola' variable)."""
        H_profile = azimuthal_avg(image=self._activation_field, n_angles=24,
                                  interp_order=1)  # find azimuthal average of activation field profile (bell-shaped)
        first, last = find_bell_boundary(H_profile, sigma_smoothing=2.5,
                                         plot_operation=False, show_operation=False)  # find extremes of the bell
        return last - first

    def angles_distribution(self, ang_tolerance: int = 15, half_circle: bool = False, weighted: bool = True,
                            plot: bool = False, show: bool = True, return_gt: bool = False) -> \
            (np.ndarray, np.ndarray, np.ndarray):
        """Analyse the angles distribution of FEM and maximum deviation from isotropy."""
        distr, bins = angles_distribution(self.steps, ang_tolerance=ang_tolerance, half_circle=half_circle,
                                          weighted=weighted, plot_distribution=plot, show_distribution=show)
        if return_gt:
            iso = np.ones_like(bins) / len(bins) * 100
            return distr, bins, iso
        return distr, bins

    def isotropy_error(self, ang_tolerance: int = 15, half_circle: bool = False, weighted: bool = True):
        distr, _, iso = self.angles_distribution(ang_tolerance=ang_tolerance, half_circle=half_circle,
                                                 weighted=weighted, return_gt=True)
        return isotropy_error(distr, iso)

    def steps_sizes_distribution(self, num_bins: int = 12, unit: str = None,
                                 plot: bool = False, show: bool = True) -> (np.ndarray, np.ndarray):
        """Analyse the step-size distribution."""
        trajectory = self.convert2unit(self._trajectory, unit=unit)
        return stepsize_distribution(trajectory, num_bins=num_bins, unit=unit,
                                     plot_distribution=plot, show_distribution=show)

    def convert2arcmin(self, sequence: np.ndarray or float) -> np.ndarray or float:
        return sequence * self._ptu_resolution

    def convert2deg(self, sequence: np.ndarray or float) -> np.ndarray or float:
        return sequence * self._ptu_resolution / 60

    def convert2unit(self, val: np.ndarray or float, unit: str = None):
        func = {'arcmin': self.convert2arcmin, 'deg': self.convert2deg, None: lambda x: x}.get(unit)
        if func is None:
            raise ValueError('The unit must be either "arcmin", "deg" or None.')
        return func(val)

    def print_info(self):
        # Print out some information of the FEM model
        print('\nNumber of MS in the FEM trajectory is: ', self._is_microsaccade.sum())
        try:
            print('Maximum value of the activation field hit by the trajectory is',
                  np.round(self._hit_activation_field[1:].max(), 2), 'while threshold for MS generation is', self.hc)
        except TypeError:
            pass
        # Find neighbourhood dimension
        print('The radius of the circular neighbourhood (i.e. the maximum step size for a drift movement) is',
              round(self.neighbour_diameter / 2, 1), 'and it should be', round(self._max_step, 1))
        # Find diameter of foveola (in which FEM are confined) from the bell-shaped activation field H
        try:
            print('The diameter in which FEM is confined (i.e. the foveola dimension according to the activation field)'
                  f' is {self.foveola_diameter} and it should be equal to {int(round(self._foveola))}\n'
                  f'(the maximum distance of the FEM to the lattice origin is {int(round(self.maxdistance2origin))} '
                  f'instead)')
        except TypeError:
            pass
        # Analysis of angles distribution
        error_iso = self.isotropy_error()
        print('Maximum percentage deviation from a perfectly isotropic movement is ' + str(error_iso.max()) +
              '% while mean is ' + str(round(error_iso.mean(), 1)) + '%.')

    def save_fem(self, dir: str, seed: str):
        file = os.path.join(dir, seed)
        np.save(file + '_steps.npy', self.steps)
        np.save(file + '_activation-field.npy', self._activation_field)
        np.save(file + '_is-ms.npy', self._is_microsaccade)

    def show_fem(self, unit: str = None, view_foveola: bool = False, show: bool = True):
        grid_sz = self.convert2unit(self._grid, unit=unit) / 2
        trajectory = self.convert2unit(self._trajectory, unit=unit) - grid_sz
        x_unit = f'({unit if unit is not None else "pan positions"})'
        y_unit = f'({unit if unit is not None else "tilt positions"})'
        fig, ax = plt.subplots(1)
        plt.imshow(self._activation_field, origin='lower',
                   extent=(-grid_sz, grid_sz, -grid_sz, grid_sz))  # display the final activation field
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-k')
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'ob')  # START position (blue)
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'or')  # END position (red)
        if view_foveola:
            circle = plt.Circle((0, 0), radius=self.convert2unit(self.foveola_diameter, unit=unit) / 2,
                                color='r', fill=False, linewidth=2, linestyle='--')
            ax.add_patch(circle)
        plt.title('FEM with n=%d steps on top of\n'
                  'the activation field' % (self._trajectory.shape[0]-1))
        plt.xlabel('x ' + x_unit)
        plt.ylabel('y ' + y_unit)
        plt.axis('equal')
        plt.tight_layout()
        if show:
            plt.show()

    def show_fem_positions(self, unit: str = None, show: bool = True):
        grid_sz = self.convert2unit(self._grid, unit=unit) / 2
        trajectory = self.convert2unit(self._trajectory, unit=unit) - grid_sz
        steps_range = range(1, trajectory.shape[0]+1)
        y_label = f'position ({unit if unit is not None else "PTU positions"})'
        plt.figure()
        plt.suptitle('Horizontal and vertical FEM positions')
        plt.plot(steps_range, trajectory[:, 0], label='Horizontal')
        plt.plot(steps_range, trajectory[:, 1], label='Vertical')
        plt.legend()
        plt.ylim((-grid_sz, grid_sz))
        plt.ylabel(y_label)
        plt.xlabel('step')
        if show:
            plt.show()

    def show_fem_displacements(self, unit: str = None, show: bool = True):
        steps = self.convert2unit(self.steps, unit=unit)
        y_label = f'displacement ({unit if unit is not None else "PTU positions"})'
        plt.figure()
        plt.plot(range(1, len(steps)+1), steps[:, 0], label='x')
        plt.plot(range(1, len(steps)+1), steps[:, 1], label='y')
        plt.legend()
        plt.title('FEM steps along the 2 axes')
        plt.ylabel(y_label)
        plt.xlabel('step')
        if show:
            plt.show()

    def show_fem_stepsize(self, unit: str = None, show: bool = True):
        distances = self.convert2unit(self.steps_sizes, unit=unit)
        y_label = f'distance ({unit if unit is not None else "PTU positions"})'
        plt.figure()
        plt.plot(range(1, len(distances)+1), distances)
        plt.title('FEM steps dimensions')
        plt.ylabel(y_label)
        plt.xlabel('step')
        if show:
            plt.show()


# --------------------------------------------------- Main Functions ---------------------------------------------------
def load_femsteps(fem_file: str) -> np.ndarray:
    if fem_file is None or not os.path.isfile(fem_file):
        raise Exception('Could not load the motion sequence. Check the given fem_file.')
    fem = np.load(fem_file)
    if len(fem.shape) != 2 or fem.shape[1] != 2:
        raise Exception('The shape of the motion sequence loaded is not correct, it should be Nx2.')
    return fem


def compute_path(steps: np.ndarray) -> np.ndarray:
    path = np.cumsum(steps, axis=0)
    return np.insert(path, 0, 0, axis=0)  # add starting position [0, 0]


def compute_angles(steps: np.ndarray, half_circle: bool = True) -> np.ndarray:
    angles = np.arctan2(steps[:, 1], steps[:, 0]) * 180 / np.pi
    angles[angles < 0] += 180 if half_circle else 360
    return angles


def compute_steps(path: np.ndarray) -> np.ndarray:
    return np.diff(path, axis=0)


def compute_stepsizes(steps: np.ndarray) -> np.ndarray:
    return np.sqrt((steps ** 2).sum(axis=1))


def compute_distances2origin(path: np.ndarray) -> np.ndarray:
    return np.sqrt((path ** 2).sum(axis=1))


def maxdistance2origin(path: np.ndarray) -> float:
    return compute_distances2origin(path).max()


def stepsize_distribution(path: np.ndarray, num_bins: int = 12, unit: str or None = None,
                          plot_distribution: bool = False, show_distribution: bool = True) -> (np.ndarray, np.ndarray):
    distances = compute_stepsizes(compute_steps(path))
    hist, bins = np.histogram(distances, bins=num_bins)
    hist = hist.astype('float64') / hist.sum() * 100
    if plot_distribution:
        unit = 'lattice units' if unit is None else unit
        plt.figure()
        plt.plot(bins[:-1], hist, 'b--', lw=1)
        plt.bar(bins[:-1], hist, width=np.mean(np.diff(bins)), alpha=0.3)
        plt.xlabel('step size (' + unit + ')')
        plt.ylabel('probability (%)')
        plt.title('Histogram of FEM steps\' size distribution')
        if show_distribution:
            plt.show()
    return hist, bins[:-1]


def angles_distribution(steps: np.ndarray, half_circle: bool = False, ang_tolerance: int = 15, weighted: bool = True,
                        plot_distribution: bool = False, show_distribution: bool = True) -> (np.ndarray, np.ndarray):
    max_ang = 360 if not half_circle else 180
    bins = np.linspace(0, max_ang, int(round(max_ang/ang_tolerance))+1, endpoint=True, dtype=int)
    angles = np.round(compute_angles(steps, half_circle=half_circle)).astype(int)
    distances = compute_stepsizes(steps)
    if weighted:
        weights = np.exp(-(np.arange(0, max_ang+1) / (ang_tolerance/3))**2 / 2)
    else:
        weights = 1-np.heaviside(np.arange(0, max_ang+1)-ang_tolerance/2, 1)
    actual_distr = np.array(list(map(lambda ang: np.sum(weights[np.abs(angles - ang)] * distances), bins)))
    actual_distr[0] += actual_distr[-1]  # close the circle (0 deg == 360 deg or 180 if half_circle)
    actual_distr = actual_distr[:-1] * 100 / actual_distr[:-1].sum()
    bins = bins[:-1]
    if plot_distribution:
        isotropic_distr = np.ones_like(bins) / len(bins) * 100
        plt.figure()
        plt.plot(bins, isotropic_distr, 'k--', lw=2)
        plt.plot(bins, actual_distr, 'b--', lw=1)
        plt.bar(bins, actual_distr, width=max_ang / (len(bins)+1), alpha=0.3)
        if actual_distr.max() > 4 * isotropic_distr[0]:
            plt.ylim(0, actual_distr.max())
        else:
            plt.ylim(0, 4 * isotropic_distr[0])
        plt.xlabel('angle (deg)')
        plt.ylabel('probability (%)')
        plt.title('Histogram of FEM angles\' distribution')
        if show_distribution:
            plt.show()
    return actual_distr, bins


def isotropy_error(actual_distr, isotropic_distr):
    # error = np.round(np.sqrt(((actual_distr - isotropic_distr) ** 2).mean()), 1)  # RMSE
    error = np.round(np.abs(actual_distr - isotropic_distr), 1)  # MAE
    return error


# --------------------------------------------- Analysis-specific Functions --------------------------------------------
def find_bell_boundary(bell: np.ndarray, sigma_smoothing: float = 3.,
                       plot_operation: bool = False, show_operation: bool = True) -> (float, float):
    dim_smooth = round(4 * sigma_smoothing)
    gauss = np.exp(-(np.arange(-dim_smooth, dim_smooth) / sigma_smoothing) ** 2 / 2)
    bell_smooth = np.convolve(bell, gauss, mode='same') / len(gauss[gauss > 0])
    grad_bell_smooth = np.diff(bell_smooth)
    first_raw, last_raw = np.argmax(grad_bell_smooth), np.argmin(grad_bell_smooth)
    bell_stat = np.concatenate((bell[:first_raw], bell[last_raw + 1:]))
    median = np.median(bell_stat)
    mad = np.median(np.absolute(bell_stat - median)) * 1.4826
    th = median + 3 * mad
    first = np.where(bell[:first_raw] > th)[0]
    first = first[0] if len(first) else first
    # first = np.argmax(bell[:first_raw] > th)
    last = last_raw + np.where(bell[last_raw + 1:] < th)[0]
    last = last[0] if len(last) else last
    if plot_operation:
        plt.figure()
        plt.plot(bell, label='original bell')
        plt.plot(bell_smooth, label='smooth bell')
        plt.plot(grad_bell_smooth, label='smooth bell derivative')
        plt.plot(first_raw, grad_bell_smooth[first_raw], 'or')
        plt.plot(last_raw, grad_bell_smooth[last_raw], 'or')
        plt.plot(first, bell[first], '*k', ms=10, label='results')
        plt.plot(last, bell[last], '*k', ms=10)
        plt.plot(range(len(bell)), np.ones(len(bell)) * th, '-k')
        plt.plot(range(len(bell)), np.ones(len(bell)) * th, '-k')
        plt.xlabel(r"$x_{\theta}$")
        plt.ylabel('amplitude')
        plt.legend()
        if show_operation:
            plt.show()
    return first, last


if __name__ == '__main__':
    # Parameters definition
    ptu_res = 46.2857 / 60  # (arcmin) resolution of PTU-E46-17
    fem_args = dict(n_burnin=10**4, max_step=6, foveola=60, n_traj=60)
    # Parameters for analysis
    fem_saw_generator = SAW(**fem_args, init_burnin=True)
    fem_saw_generator.generate_fem(verbose=False)
    fem_saw_generator.show_fem(unit='arcmin')
    fem_saw_generator.angles_distribution(plot=True, show=False)
    fem_saw_generator.steps_sizes_distribution(unit='arcmin', plot=True, show=False)
    fem_saw_generator.show_fem_positions(unit='arcmin', show=False)
    fem_saw_generator.show_fem_stepsize(unit='arcmin', show=True)
