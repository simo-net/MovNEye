import sys
import time
import serial
import serial.tools.list_ports


class PTU(object):
    """
    A FLIR PTU E-series remote controller through serial communication.
    Note: Serial wired connection between PTU and host computer is required.
    """

    def __init__(self, port: int, baud: int = 9600, timeout: float = 6, steps: (str, str) = ('H', 'H'),
                 verbose: bool = True):
        """The central class to the PTU controller via serial connection.
        ==============
        ### Internal Communication-specific parameters:
            - port (int): The serial port that the PTU was connected to [required info].
            - baud (int): The baud rate of serial communication with the PTU [default: 9600].
            - timeout (float): Maximum time (in seconds) for waiting response from the device [default 60].
        > NOTE: All communication specific parameters are private members of this class and can be set only when at
        class initialization (when the instance is created).
        ==============
        ### Internal PTU-specific variables:
        --> Basic Settings
        All basic PTU variables, they cannot be modified on the fly. They are private members of the class that can be
        accessed (view current status) and directly modified. To query their current status or change it you can also
        call their specific get- or set- function.
            - steps (2 str tuple): Step mode determines the resolution of pan/tilt rotations, in fractions of the max
              resolution value: {'F'=full step, 'H'=half, 'Q'=quarter, 'E'=eighth, 'A'=automatic} [default: ('H','H')].
            - execution (str): execution mode ('I'=immediate, 'S'=slaved).
            - control_mode (str): speed control mode can be 'pos' (speed and position are independently controlled) or
              'vel' (pure velocity mode, where speed affects position).
            - echo (bool): either True if echo is enabled, or False otherwise.
            - hold_power (str): the power mode during holding state, i.e. the amount of power the PTU uses in order to
              hold the payload in a fixed position. It can be either 'OFF', 'LOW' or 'REG' (regular).
            - move_power (str): the power mode during motion state, i.e. the amount of power the PTU uses when moving
              the payload. It can be either 'LOW', 'REG' (regular) or 'HIGH'.
            - position_limits (4 int tuple): minimum and maximum allowable pan and tilt positions.
            - speed_bounds (6 int tuple): minimum, base (immediately reachable) and maximum allowable pan and tilt
              speeds in pos/sec.
            - resolutions (2 float tuple): resolution of pan and tilt displacements in arc-seconds.
        > NOTE: Resolutions cannot be directly modified but can be changed through steps property variables.
        --> On-The-Fly Settings
        All PTU variables that can be modified "on-the-fly" (i.e. while PTU is executing previous commands), such as
        positions (absolute and relative), speeds (absolute and relative) and accelerations. They can be accessed
        through their get- function, and modified through the set- function, as well as with specific and user-friendly
        property variables (class members).
        > NOTE: Position offsets, relative speeds (i.e. speed offsets) and trajectory cannot be queried but can only be
        set through their set- methods. Conversely, (actual) current speeds cannot be set but only queried through its
        custom get- method.
        """
        self.verbose = verbose
        self._closed = False

        # ### Communication-specific variables

        # Serial port where PTU is connected
        self.__port = port
        # Baud rate of data transmission
        self.__baud = baud
        # Timeout of data reading
        self.__timeout = timeout

        # Termination character of received data
        self._TERM = b'\r\n'

        # Open serial communication
        self._open()

        # ### PTU-specific variables

        # Step mode
        self.__steps = (self.getPanStep(), self.getTiltStep())
        if self.__steps != steps:
            self.setSteps(steps)
        # Execution mode
        self.__execution = self.getExecutionMode()
        # Control Mode
        self.__controlMode = self.getControlMode()
        # Echo State
        self.__echo = self.getEchoState()
        # Power Modes
        self.__HoldPowerMode = self.getHoldPowerMode()
        self.__MovePowerMode = self.getMovePowerMode()
        # Resolution
        self.__panResolution = self.getPanResolution()
        self.__tiltResolution = self.getTiltResolution()
        # Position Limits
        self.__panLimits = self.getPanPositionLimits()
        self.__tiltLimits = self.getTiltPositionLimits()
        # Speed Limits
        self.__panSpeedBounds = self.getPanSpeedBounds()
        self.__tiltSpeedBounds = self.getTiltSpeedBounds()

    @property
    def device_version(self) -> str:
        self.__send_command(b'V ')
        return self.__get_response().replace('V', '')[3:]

    @property
    def device_model(self) -> str:
        self.__send_command(b'VM ')
        return self.__get_response().split()[-1]

    @property
    def device_serial_number(self) -> int:
        self.__send_command(b'VS ')
        return int(self.__get_response().split()[-1])

    # =================================================================================================================
    # --> Open/Close communication
    # =================================================================================================================

    def _open(self):
        """Establish connection."""
        # Create a serial communication object
        self.serial = serial.Serial(port=self.__port, baudrate=9600, timeout=self.__timeout,
                                    bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE,
                                    parity=serial.PARITY_NONE)
        # Check if communication is properly working
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write(b'V ')
        resp = self.serial.read_until(self._TERM).decode()
        if not resp:
            self.serial.close()
            self._closed = True
            raise ConnectionError('\nCannot communicate with serial device. Check if it is powered-on and try again.\n')
        # Initialize data transmission (clear the buffer)
        size = self.serial.in_waiting
        if size:
            self.serial.read(size)
        # self.serial.flush()
        # Change baud rate if not default
        if self.__baud in [600, 1200, 2400, 4800, 19200, 38400, 57600, 115200]:
            self.__send_command(b'@(' + str(self.__baud).encode() + b',0,F) ')
            self.__get_response()
            self.serial.baudrate = self.__baud
        elif self.__baud != 9600:
            print('Baud should be one of {600, 1200, 2400, 4800, 9600, 19200, '
                  + '38400, 57600, 115200}. Defaulting to 9600.', file=sys.stderr)
        # Print out PTU information
        respV = self.device_version.split()
        if self.verbose:
            print("\n\n--> [PTU controller MSG] Starting up serial connection at port %s." % self.__port)
            respM = self.device_model
            respS = self.device_serial_number
            print("\n\n### " + ' '.join(respV[1:3]).upper() + "\n### " + ' '.join(respV[3:]) +
                  "\n### PTU model " + respM + ", serial number " + str(respS) + "\n\n")
            print('\nBaud rate of serial communication is set to {}, corresponding to a bit rate of {} bps.\n'
                  .format(self.serial.baudrate, int(self.serial.baudrate * self.serial.bytesize)))
        # Set feedback mode
        self.__send_command(b'FT ')
        self.__get_response()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._closed:
            self.close()

    def __del__(self):
        if not self._closed:
            self.close()

    def close(self):
        """Close the connection."""
        self._closed = True
        # Halt the motors in case the PTU is still moving
        self.halt()
        # Reset PTU to the origin
        if any(self.getPositions()):
            self.setPositions(0, 0)
        # Reset default baudrate
        if self.__baud in [600, 1200, 2400, 4800, 19200, 38400, 57600, 115200]:
            self.__send_command(b'@(9600,0,F) ')
            self.__get_response()
            self.serial.baudrate = 9600
        # Read last data in the buffer (in case there is)
        size = self.serial.in_waiting
        if size:
            self.serial.read(size)
        # Close the serial communication
        self.serial.close()
        if self.verbose:
            print("\n\n--> [PTU controller MSG] The serial connection is closed.\n\n")

    # =================================================================================================================
    # --> Send/Receive Data
    # =================================================================================================================

    def __send_command(self, command: bytes):
        """Write command to serial port."""
        # Check for any pending input: buffer should be empty before sending commands
        if self.serial.in_waiting:
            print('Device asynchronously displayed: ' + self.serial.read(self.serial.in_waiting).decode(), file=sys.stderr)
        # Send command to serial port
        self.serial.write(command)

    def __get_response(self) -> str:
        """Return device response as a string, without terminating characters."""
        # Read response from serial port
        response = self.serial.read_until(self._TERM).decode()
        return response[:-2]

    def __get_response_ntimes(self, n: int) -> str:
        """Return n device responses as a single string without terminating characters."""
        response = ''
        for k in range(n):
            response += self.serial.read_until(self._TERM).decode()
        return response.replace(self._TERM.decode(), '')

    def __send(self, command: bytes, verbose: bool = False) -> str:
        """Send and receive data and print all info. If enabled, echo is removed from response."""
        # Send command
        self.__send_command(command)
        # Get response
        response = self.__get_response()
        if self.echo:
            response = response[len(command):]
        if self.verbose and verbose:
            print(' -> Sent:', command.decode())
            print('Received:', response)
        return response

    def send(self, command: bytes or str, verbose: bool = True) -> str:
        """Wraps the previous two functions with some additional checks."""
        command = command.upper()
        # Check if command is of type bytes or string
        if isinstance(command, bytes):
            pass
        elif isinstance(command, str):
            command = command.encode()  # convert command to bytes
        else:
            raise TypeError('Commands should be of type bytes or string.')
        # Add blank space to command if not there
        if not command.endswith(b' '):
            command += b' '
        # Check if the command sent should make some internal parameters change
        check = {b'@(': "Change 'baud' parameter at class instantiation",
                 b'WP': "Change 'steps' parameter at class instantiation",
                 b'WT': "Change 'steps' parameter at class instantiation",
                 b'I ': "Use setExecution('I')", b'S ': "Use setExecution('S')",
                 b'CI': "Use setControlMode('pos')", b'CV': "Use setControlMode('vel')",
                 b'EE': "Use setEcho(True)", b'ED': "Use setEcho(False)",
                 b'PH': "Use setHoldPowerMode(pan, tilt)", b'TH': "Use setHoldPowerMode(pan, tilt)",
                 b'PM': "Use setMovePowerMode(pan, tilt)", b'TM': "Use setMovePowerMode(pan, tilt)",
                 b'PN': "Use setPanLimits(low, high)", b'PX': "Use setPanLimits(low, high)",
                 b'TN': "Use setTiltLimits(low, high)", b'TX': "Use setTiltLimits(low, high)",
                 b'PL': "Use setPanSpeedBounds(low, base, high)", b'PB': "Use setPanSpeedBounds(low, base, high)",
                 b'PU': "Use setPanSpeedBounds(low, base, high)",
                 b'TL': "Use setTiltSpeedBounds(low, base, high)", b'TB': "Use setTiltSpeedBounds(low, base, high)",
                 b'TU': "Use setTiltSpeedBounds(low, base, high)"
                 }.get(command[0:2])
        if check:
            print("Do not use send() method for this operation. %s instead." % check, file=sys.stderr)
            return 'Command not executed!'
        if command[0:1] == b'F':
            print("Feedback mode is set to 'ASCII terse mode' and cannot be changed.", file=sys.stderr)
            return 'Command not executed!'
        # Send command
        return self.__send(command, verbose)

    # =================================================================================================================
    # --> Query/Set BASIC SETTINGS
    # =================================================================================================================

    # ********************** Step-Modes/Resolution *********************
    @property
    def steps(self) -> (str, str):
        """Query panning and tilting step modes."""
        return self.__steps

    @steps.setter
    def steps(self, modes: (str, str)):
        """Set pan and tilt resolution (step modes) as a tuple (pan step, tilt step), where steps can be
        one of the following {'F'=full, 'H'=half, 'Q'=quarter, 'E'=eighth or 'A'=automatic}.
        NOTE: no setter is available for pan/tilt resolutions, you can only change these values by setting
        pan/tilt steps"""
        if self.__steps != modes:
            self.setSteps(modes)

    def getSteps(self) -> (str, str):
        """Query pan and tilt resolution as a tuple (pan step, tilt step), where steps can be
        one of the following {'F'=full, 'H'=half, 'Q'=quarter, 'E'=eighth or 'A'=automatic}."""
        return self.__steps

    def setSteps(self, modes: (str, str)):
        """Set pan and tilt resolution as a tuple (pan step, tilt step), where steps can be
        one of the following {'F'=full, 'H'=half, 'Q'=quarter, 'E'=eighth or 'A'=automatic}."""
        # Check if input is a tuple
        if type(modes) is tuple:
            # Check if elements in the tuple are string types and in the range of possible step modes
            if all(isinstance(k, str) and k.upper() in ['F', 'H', 'Q', 'E', 'A'] for k in modes):
                panstep, tiltstep = modes[0].upper(), modes[1].upper()
                # Set pan step mode if current one is different from desired
                if self.__steps[0] != panstep:
                    panstep = self.setPanStep(panstep)
                # Set tilt step mode if current one is different from desired
                if self.__steps[1] != tiltstep:
                    tiltstep = self.setTiltStep(tiltstep)
                self.__steps = (panstep, tiltstep)
            else:
                raise TypeError("Pan and tilt step modes in the tuple must be one of the following letters:"
                                "'F'=full, 'H'=half, 'Q'=quarter, 'E'=eighth or 'A'=automatic.")
        else:
            raise TypeError("The step parameter must be a tuple of string elements.")

    def getPanStep(self) -> str:
        """Query pan step mode: either F, H, Q, E or A."""
        self.__send_command(b'WP ')
        resp = self.__get_response()
        # response == '* H' (or F or Q or E or A)
        if resp.split()[-1] in ['F', 'H', 'Q', 'E', 'A']:
            return resp.split()[-1]
        else:
            raise RuntimeError('When queried for pan step, device gave unexpected reply:\n  ' + resp)

    def setPanStep(self, mode: str) -> str:
        """Set pan resolution choosing between 5 possible step modes
        {'F'=full, 'H'=half, 'Q'=quarter, 'E'=eighth or 'A'=automatic}."""
        mode = mode.upper()
        self.__send_command(b'WP' + mode.encode() + b' ')
        self.__get_response()
        # Pan recalibration is required (it will take some time!)
        self.recalibrate_pan()
        return mode

    def getTiltStep(self) -> str:
        """Query tilt step mode: either F, H, Q, E or A."""
        self.__send_command(b'WT ')
        resp = self.__get_response()
        # response == '* H' (or F or Q or E or A)
        if resp.split()[-1] in ['F', 'H', 'Q', 'E', 'A']:
            return resp.split()[-1]
        else:
            raise RuntimeError('When queried for tilt step, device gave unexpected reply:\n  ' + resp)

    def setTiltStep(self, mode: str) -> str:
        """Set tilt resolution choosing between 5 possible step modes
        {'F'=full, 'H'=half, 'Q'=quarter, 'E'=eighth or 'A'=automatic}."""
        mode = mode.upper()
        self.__send_command(b'WT' + mode.encode() + b' ')
        self.__get_response()
        # Tilt recalibration is required (it will take some time!)
        self.recalibrate_tilt()
        return mode

    @property
    def resolutions(self) -> (float, float):
        """Query pan and tilt resolution in arcseconds per position."""
        return self.__panResolution, self.__tiltResolution

    @property
    def pan_resolution(self) -> (float, float):
        """Query pan resolution in arcseconds per position."""
        return self.__panResolution

    @property
    def tilt_resolution(self) -> (float, float):
        """Query tilt resolution in arcseconds per position."""
        return self.__tiltResolution

    def getPanResolution(self) -> float:
        """Query panning resolution in arcseconds per position."""
        self.__send_command(b'PR ')
        resp = self.__get_response()
        # '* 46.2857 seconds arc per position' in verbose feedback mode, '* 46.2857' in terse feedback mode
        try:
            return float(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for pan resolution, device gave unexpected reply:\n  ' + resp)

    def getTiltResolution(self) -> float:
        """Query tilting resolution in arcseconds per position."""
        self.__send_command(b'TR ')
        resp = self.__get_response()
        # '* 46.2857 seconds arc per position' in verbose feedback mode, '* 46.2857' in terse feedback mode
        try:
            return float(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for tilt resolution, device gave unexpected reply:\n  ' + resp)

    # ************************ Execution-Mode **************************
    @property
    def execution(self) -> str:
        """Query whether the execution mode is immediate or slaved."""
        return self.__execution

    @execution.setter
    def execution(self, exemode: str):
        """Set the execution mode of the PTU. It can be either 'I'=immediate mode, or 'S'=slaved mode."""
        self.setExecutionMode(exemode)

    def getExecutionMode(self) -> str:
        """Query execution mode: 'I' if immediate execution mode, 'S' if slaved."""
        self.__send_command(b'IQ ')
        resp = self.__get_response()
        # '* I' or '* S'
        if resp.split()[-1] == 'I':
            return 'I'
        elif resp.split()[-1] == 'S':
            return 'S'
        else:
            raise RuntimeError('When queried for current execution mode, device gave unexpected reply:\n  ' + resp)

    def setExecutionMode(self, exemode: str):
        """Set the execution mode of the PTU. It can be either 'I'=immediate mode,
        or 'S'=slaved mode."""
        exemode = exemode.upper()
        if exemode in ['I', 'S']:
            self.__send_command(exemode.encode() + b' ')
            resp = self.__get_response()
            if resp[-1] != '*':
                print('When setting execution mode, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            else:
                self.__execution = exemode
        else:
            print("Execution mode should be either 'I'=immediate or 'S'=slaved.", file=sys.stderr)

    # ************************** Control-Mode **************************
    @property
    def control_mode(self) -> str:
        """Query control mode ('pos' or 'vel')."""
        return self.__controlMode

    @control_mode.setter
    def control_mode(self, control: str):
        """Set the speed control mode of the PTU. It can be either 'pos' (speed and position independently controlled)
        or 'vel' (pure velocity mode, where speed affects position)."""
        self.setControlMode(control)

    def getControlMode(self) -> str:
        """Query current control mode ('pos' or 'vel')."""
        self.__send_command(b'C ')
        resp = self.__get_response()
        # '* Independent speed control mode' or '* Pure Velocity speed control mode'
        if 'Independent' in resp:
            return 'pos'
        elif 'Velocity' in resp:
            return 'vel'
        else:
            raise RuntimeError('When queried for speed control mode, device gave unexpected reply:\n  ' + resp)

    def setControlMode(self, control: str):
        """Set the speed control mode of the PTU. It can be either 'pos' (speed and position independently controlled)
        or 'vel' (pure velocity mode, where speed affects position)."""
        if control in ['pos', 'vel']:
            mode = {'pos': 'I', 'vel': 'V'}.get(control)
            self.__send_command(b'C' + mode.encode() + b' ')
            resp = self.__get_response()
            if resp[-1] != '*':
                print('When setting execution mode, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            else:
                self.__controlMode = control
        else:
            print("Speed control mode should be either 'pos' or 'vel'.", file=sys.stderr)

    # ****************************** Echo ******************************
    @property
    def echo(self) -> bool:
        """Query whether PTU is configured to repeat commands back to host: True if echo state is enabled,
        False otherwise."""
        return self.__echo

    @echo.setter
    def echo(self, on: bool):
        """Set the echo state of the PTU (whether or not it should return the input it received).
        True to activate echo, False otherwise."""
        self.setEchoState(on)

    def getEchoState(self) -> bool:
        """Query echo state: True if echo enabled, False otherwise."""
        self.__send_command(b'E ')
        resp = self.__get_response()
        # 'E * Echoing ON' or '* Echoing OFF'
        if resp == 'E * Echoing ON':
            return True
        elif resp == '* Echoing OFF':
            return False
        else:
            raise RuntimeError('When queried for echo state, device gave unexpected reply:\n  ' + resp)

    def setEchoState(self, on: bool):
        """Set the echo state of the PTU (whether or not it should return the input it received).
        True to activate echo, False otherwise."""
        if not isinstance(on, bool):
            raise TypeError('Echo must be either True or False.')
        if on:
            self.__send_command(b'EE ')
        else:
            self.__send_command(b'ED ')
        resp = self.__get_response()
        if resp[-1] != '*':
            print('When setting echo mode, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
        else:
            self.__echo = on

    # ************************ Hold Power-Mode *************************
    @property
    def hold_power(self) -> (str, str):
        """Query pan and tilt hold power modes."""
        return self.__HoldPowerMode

    @hold_power.setter
    def hold_power(self, mode: (str, str)):
        """Set the power mode during holding state, i.e. the amount of power the PTU uses in order to hold the payload
        in a fixed position. It can be either 'OFF', 'LOW' or 'REG' (regular).
        Important Notes:
            - 'REG' hold power setting is intended only for intermittent duty cycles (<20%). Extended regular power
              cycles can overheat the PTU motors! It should be used for brief periods when very high holding torque
              may be required. Regular hold power mode should be avoided or used sparingly.
            - when using 'OFF' hold power mode, fully test that your load does not back-drive the unit when stationary.
              Back-driving will cause the controller to lose track of pan-tilt position, and this requires the unit to
              be reset. Back-driving is more likely on the tilt axis which has higher torque applied to it by the load.
        Notes:
            - if you change the hold power setting, the PTU must move before the new setting takes effect.
            - if you set hold power to 'OFF', there will be a 2‐second delay from the time the axis stops moving to when
              hold power shuts off. This allows the payload to come to a complete stop.
        """
        self.setHoldPowerMode(*mode)

    def getHoldPowerMode(self) -> (str, str):
        """Query the pan and tilt hold power modes."""
        power_modes = ['OFF', 'LOW', 'REG']
        self.__send_command(b'PH ')
        resp1 = self.__get_response().split()[-1]
        # '* Pan in REGULAR hold power mode' in verbose feedback mode, '* REG' in terse feedback mode
        self.__send_command(b'TH ')
        resp2 = self.__get_response().split()[-1]
        # '* Tilt in REGULAR hold power mode' in verbose feedback mode, '* REG' in terse feedback mode
        if resp1 not in power_modes or resp2 not in power_modes:
            raise RuntimeError('When queried for pan and tilt move power mode, device gave unexpected reply:\n  '
                               + resp1 + ' & ' + resp2)
        return resp1, resp2

    def setHoldPowerMode(self, pan: str, tilt: str):
        """Set the power mode during holding state, i.e. the amount of power the PTU uses in order to hold the payload
        in a fixed position. It can be either 'OFF', 'LOW' or 'REG' (regular).
        Important Notes:
            - 'REG' hold power setting is intended only for intermittent duty cycles (<20%). Extended regular power
              cycles can overheat the PTU motors! It should be used for brief periods when very high holding torque
              may be required. Regular hold power mode should be avoided or used sparingly.
            - when using 'OFF' hold power mode, fully test that your load does not backdrive the unit when stationary.
              Back-driving will cause the controller to lose track of pan-tilt position, and this requires the unit to
              be reset. Back-driving is more likely on the tilt axis which has higher torque applied to it by the load.
        Notes:
            - if you change the hold power setting, the PTU must move before the new setting takes effect.
            - if you set hold power to 'OFF', there will be a 2‐second delay from the time the axis stops moving to when
              hold power shuts off. This allows the payload to come to a complete stop.
        """
        pan, tilt = pan.upper(), tilt.upper()
        modes = {'OFF': 'O', 'LOW': 'L', 'REG': 'R'}
        if pan in modes.keys() and tilt in modes.keys():
            self.__send_command(b'PH' + modes.get(pan).encode() + b' ')
            resp1 = self.__get_response()
            self.__send_command(b'TH' + modes.get(tilt).encode() + b' ')
            resp2 = self.__get_response()
            if '!' in resp1 + resp2:
                print('When setting pan and tilt hold power mode, device gave unexpected reply:\n  '
                      + resp1 + ' & ' + resp2, file=sys.stderr)
                self.__HoldPowerMode = self.getHoldPowerMode()
            else:
                self.__HoldPowerMode = (pan, tilt)
                verbose = self.verbose
                self.verbose = False
                self.setTrajectory([[1, 1], [-1, -1]])  # the PTU must move before the new setting takes effect
                self.verbose = verbose
        else:
            print("Pan and tilt hold power mode should be either 'OFF', 'LOW' or 'REG'.", file=sys.stderr)

    # ************************ Move Power-Mode *************************
    @property
    def move_power(self) -> (str, str):
        """Query pan and tilt move power modes."""
        return self.__MovePowerMode

    @move_power.setter
    def move_power(self, mode: (str, str)):
        """Set the power mode during motion state, i.e. the amount of power the PTU uses when moving the payload.
        It can be either 'LOW', 'REG' (regular) or 'HIGH'.
        Important Note:
            - 'HIGH' motion power setting is intended only for intermittent duty cycles (<20%). Extended high power
              cycles can overheat the PTU motors!
        """
        self.setMovePowerMode(*mode)

    def getMovePowerMode(self) -> (str, str):
        """Query the pan and tilt move power modes."""
        power_modes = ['LOW', 'REG', 'HIGH']
        self.__send_command(b'PM ')
        resp1 = self.__get_response().split()[-1]
        # '* Pan in REGULAR move power mode' in verbose feedback mode, '* REG' in terse feedback mode
        self.__send_command(b'TM ')
        resp2 = self.__get_response().split()[-1]
        # '* Tilt in REGULAR move power mode' in verbose feedback mode, '* REG' in terse feedback mode
        if resp1 not in power_modes or resp2 not in power_modes:
            raise RuntimeError('When queried for pan and tilt move power mode, device gave unexpected reply:\n  '
                               + resp1 + ' & ' + resp2)
        return resp1, resp2

    def setMovePowerMode(self, pan: str, tilt: str):
        """Set the power mode during motion state, i.e. the amount of power the PTU uses when moving the payload.
        It can be either 'LOW', 'REG' (regular) or 'HIGH'.
        Important Note:
            - 'HIGH' motion power setting is intended only for intermittent duty cycles (<20%). Extended high power
              cycles can overheat the PTU motors!
        """
        pan, tilt = pan.upper(), tilt.upper()
        modes = {'LOW': 'L', 'REG': 'R', 'HIGH': 'H'}
        if pan in modes.keys() and tilt in modes.keys():
            self.__send_command(b'PM' + modes.get(pan).encode() + b' ')
            resp1 = self.__get_response()
            self.__send_command(b'TM' + modes.get(tilt).encode() + b' ')
            resp2 = self.__get_response()
            if '!' in resp1 + resp2:
                print('When setting pan and tilt move power mode, device gave unexpected reply:\n  '
                      + resp1 + ' & ' + resp2, file=sys.stderr)
                self.__MovePowerMode = self.getMovePowerMode()
            else:
                self.__MovePowerMode = (pan, tilt)
        else:
            print("Pan and tilt move power mode should be either 'LOW', 'REG' or 'HIGH'.", file=sys.stderr)

    # ************************ Position-Limits *************************
    @property
    def position_limits(self) -> ((int, int), (int, int)):
        """Query minimum and maximum pan and tilt positions: (pan lower, pan upper), (tilt lower, tilt upper)."""
        return self.__panLimits, self.__tiltLimits

    @position_limits.setter
    def position_limits(self, lims: ((int, int), (int, int))):
        """Set pan and tilt position limits: (pan lower, pan upper), (tilt lower, tilt upper)."""
        self.setPositionLimits(*lims[0], *lims[1])

    def getPositionLimits(self) -> ((int, int), (int, int)):
        """Query minimum and maximum pan and tilt positions: (pan lower, pan upper), (tilt lower, tilt upper)."""
        return self.__panLimits, self.__tiltLimits

    def setPositionLimits(self, lowP: int, upP: int, lowT: int, upT: int):
        """Set pan and tilt position limits: (pan lower, pan upper), (tilt lower, tilt upper)."""
        self.setPanPositionLimits(lowP, upP)
        self.setTiltPositionLimits(lowT, upT)

    @property
    def pan_position_limits(self) -> (int, int):
        """Query minimum and maximum pan positions."""
        return self.__panLimits

    @pan_position_limits.setter
    def pan_position_limits(self, lims: (int, int)):
        """Set pan position limits (lower, upper)."""
        self.setPanPositionLimits(*lims)

    def getPanPositionLimits(self) -> (int, int):
        """Query minimum and maximum pan positions."""
        self.__send_command(b'PN ')
        minpan = self.__get_response()
        # '* Minimum Pan position is -3000' in verbose feedback mode, '* -3000' in terse feedback mode
        self.__send_command(b'PX ')
        maxpan = self.__get_response()
        # '* Maximum Pan position is 3000' in verbose feedback mode, '* 3000' in terse feedback mode
        try:
            return int(minpan.split()[-1]), int(maxpan.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for pan limits, device gave unexpected reply:\n  '
                               + minpan + ' & ' + maxpan)

    def setPanPositionLimits(self, low: int, up: int):
        """Set pan position limits"""
        if not isinstance(low, int) or not isinstance(up, int):
            raise TypeError("Lower and upper pan position limits must be integer values.")
        # Enable user-defined limits.
        self.enableUserLimits()
        # Set lower limit
        self.__send_command(b'PNU' + str(low).encode() + b' ')
        resp1 = self.__get_response()
        # Set upper limit
        self.__send_command(b'PXU' + str(up).encode() + b' ')
        resp2 = self.__get_response()
        # Pan recalibration is required (it will take some time!)
        self.recalibrate_pan()
        if '!' in resp1 + resp2:
            print('When setting pan limits, device gave unexpected reply:\n  ' + resp1 + ' & ' + resp2,
                  file=sys.stderr)
            self.__panLimits = self.getPanPositionLimits()
        else:
            self.__panLimits = (low, up)

    @property
    def tilt_position_limits(self) -> (int, int):
        """Query minimum and maximum tilt positions."""
        return self.__tiltLimits

    @tilt_position_limits.setter
    def tilt_position_limits(self, lims: (int, int)):
        """Set tilt position limits (lower, upper)."""
        self.setTiltPositionLimits(*lims)

    def getTiltPositionLimits(self) -> (int, int):
        """Query minimum and maximum tilt positions."""
        self.__send_command(b'TN ')
        mintilt = self.__get_response()
        # '* Minimum Tilt position is -200' in verbose feedback mode, '* -200' in terse feedback mode
        self.__send_command(b'TX ')
        maxtilt = self.__get_response()
        # '* Maximum Tilt position is 400' in verbose feedback mode, '* 400' in terse feedback mode
        try:
            return int(mintilt.split()[-1]), int(maxtilt.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for tilt limits, device gave unexpected reply:\n  '
                               + mintilt + ' & ' + maxtilt)

    def setTiltPositionLimits(self, low: int, up: int):
        """Set tilt position limits"""
        if not isinstance(low, int) or not isinstance(up, int):
            raise TypeError("Lower and upper tilt position limits must be integer values.")
        # Enable user-defined limits.
        self.enableUserLimits()
        # Set lower limit
        self.__send_command(b'TNU' + str(low).encode() + b' ')
        resp1 = self.__get_response()
        # Set upper limit
        self.__send_command(b'TXU' + str(up).encode() + b' ')
        resp2 = self.__get_response()
        # Tilt recalibration is required (it will take some time!)
        self.recalibrate_tilt()
        if '!' in resp1 + resp2:
            print('When setting tilt limits, device gave unexpected reply:\n  ' + resp1 + ' & ' + resp2,
                  file=sys.stderr)
            self.__tiltLimits = self.getTiltPositionLimits()
        else:
            self.__tiltLimits = (low, up)

    def enableUserLimits(self):
        """Enable user-defined limits."""
        self.__send_command(b'LU ')
        self.__get_response()

    # ************************* Speed-Bounds ***************************
    @property
    def speed_bounds(self) -> ((int, int, int), (int, int, int)):
        """Query minimum, base and maximum panning and tilting speeds:
        (pan lower, pan base, pan upper), (tilt lower, tilt base, tilt upper)."""
        return self.__panSpeedBounds, self.__tiltSpeedBounds

    @speed_bounds.setter
    def speed_bounds(self, bounds: ((int, int, int), (int, int, int))):
        """Set minimum, base and maximum pan and tilt speed bounds:
        (pan lower, pan base, pan upper), (tilt lower, tilt base, tilt upper)."""
        self.setSpeedBounds(*bounds[0], *bounds[1])

    def getSpeedBounds(self) -> ((int, int, int), (int, int, int)):
        """Query pan and tilt speed bounds."""
        return self.__panSpeedBounds, self.__tiltSpeedBounds

    def setSpeedBounds(self, lowP: int, baseP: int, upP: int, lowT: int, baseT: int, upT: int):
        """Set pan and tilt speed bounds."""
        self.setPanSpeedBounds(lowP, baseP, upP)
        self.setTiltSpeedBounds(lowT, baseT, upT)

    @property
    def pan_speed_bounds(self) -> (int, int, int):
        """Query minimum, base and maximum panning speeds."""
        return self.__panSpeedBounds

    @pan_speed_bounds.setter
    def pan_speed_bounds(self, bounds: (int, int, int)):
        """Set minimum, base and maximum pan speed bounds."""
        self.setPanSpeedBounds(*bounds)

    def getPanSpeedBounds(self) -> (int, int, int):
        """Query minimum (lower), base (immediately reachable) and maximum (upper) pan speed bounds."""
        self.__send_command(b'PL ')
        minspeed = self.__get_response()
        # '* Minimum Pan speed is 0 positions/sec' in verbose feedback mode, '* 0' in terse feedback mode
        self.__send_command(b'PB ')
        basespeed = self.__get_response()
        # '* Current Pan base speed is 0 positions/sec' in verbose feedback mode, '* 0' in terse feedback mode
        self.__send_command(b'PU ')
        maxspeed = self.__get_response()
        # '* Maximum Pan speed is 12000 positions/sec' in verbose feedback mode, '* 12000' in terse feedback mode
        try:
            return int(minspeed.split()[-1]), int(basespeed.split()[-1]), int(maxspeed.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for pan speed bounds, device gave unexpected reply:\n  '
                               + minspeed + ' & ' + basespeed + ' & ' + maxspeed)

    def setPanSpeedBounds(self, low: int, base: int, up: int):
        """Set pan speed bounds: lower (minimum), base (immediately reachable)
        and upper (maximum) values."""
        if not all(isinstance(val, int) for val in [low, base, up]):
            raise TypeError("Lower, base and upper pan speed bounds must be integer values.")
        # Set lower bound
        self.__send_command(b'PL' + str(low).encode() + b' ')
        resp1 = self.__get_response()
        # Set upper bound
        self.__send_command(b'PU' + str(up).encode() + b' ')
        resp3 = self.__get_response()
        # Set base bound
        self.__send_command(b'PB' + str(base).encode() + b' ')
        resp2 = self.__get_response()
        if '!' in resp1 + resp2 + resp3:
            print('When setting pan speed bounds, device gave unexpected reply:\n  '
                  + resp1 + ' & ' + resp2 + ' & ' + resp3, file=sys.stderr)
            self.__panSpeedBounds = self.getPanSpeedBounds()
        else:
            self.__panSpeedBounds = (low, base, up)

    @property
    def tilt_speed_bounds(self) -> (int, int, int):
        """Query minimum, base and maximum tilting speeds."""
        return self.__tiltSpeedBounds

    @tilt_speed_bounds.setter
    def tilt_speed_bounds(self, bounds: (int, int, int)):
        """Set minimum, base and maximum tilt speed bounds."""
        self.setTiltSpeedBounds(*bounds)

    def getTiltSpeedBounds(self) -> (int, int, int):
        """Query minimum (lower), base (immediately reachable) and maximum (upper) tilt speed bounds."""
        self.__send_command(b'TL ')
        minspeed = self.__get_response()
        # '* Minimum Tilt speed is 0 positions/sec' in verbose feedback mode, '* 0' in terse feedback mode
        self.__send_command(b'TB ')
        basespeed = self.__get_response()
        # '* Current Tilt base speed is 0 positions/sec' in verbose feedback mode, '* 0' in terse feedback mode
        self.__send_command(b'TU ')
        maxspeed = self.__get_response()
        # '* Maximum Tilt speed is 12000 positions/sec' in verbose feedback mode, '* 12000' in terse feedback mode
        try:
            return int(minspeed.split()[-1]), int(basespeed.split()[-1]), int(maxspeed.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for tilt speed bounds, device gave unexpected reply:\n  '
                               + minspeed + ' & ' + basespeed + ' & ' + maxspeed)

    def setTiltSpeedBounds(self, low: int, base: int, up: int):
        """Set tilt speed bounds: lower (minimum), base (immediately reachable)
        and upper (maximum) values."""
        if not all(isinstance(val, int) for val in [low, base, up]):
            raise TypeError("Lower, base and upper tilt speed bounds must be integer values.")
        # Set lower bound
        self.__send_command(b'TL' + str(low).encode() + b' ')
        resp1 = self.__get_response()
        # Set upper bound
        self.__send_command(b'TU' + str(up).encode() + b' ')
        resp3 = self.__get_response()
        # Set base bound
        self.__send_command(b'TB' + str(base).encode() + b' ')
        resp2 = self.__get_response()
        if '!' in resp1 + resp2 + resp3:
            print('When setting tilt speed bounds, device gave unexpected reply:\n  '
                  + resp1 + ' & ' + resp2 + ' & ' + resp3, file=sys.stderr)
            self.__tiltSpeedBounds = self.getTiltSpeedBounds()
        else:
            self.__tiltSpeedBounds = (low, base, up)

    # =================================================================================================================
    # --> Query/Set ON-THE-FLY SETTINGS
    # =================================================================================================================

    # **************************** Position ****************************
    @property
    def positions(self) -> (int, int):
        """Query pan and tilt positions."""
        return self.getPanPosition(), self.getTiltPosition()

    @positions.setter
    def positions(self, pos: (int, int)):
        """Set next pan and tilt positions."""
        self.setPositions(*pos)

    def getPositions(self) -> (int, int):
        """Query current pan and tilt position."""
        return self.getPanPosition(), self.getTiltPosition()

    def setPositions(self, pan: int, tilt: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to go to a pan and tilt position."""
        resp1 = self.setPanPosition(pan, blocking=False, verbose=verbose)
        resp2 = self.setTiltPosition(tilt, blocking=blocking, verbose=verbose)
        return resp1 and resp2

    @property
    def pan_position(self) -> int:
        """Query pan position."""
        return self.getPanPosition()

    @pan_position.setter
    def pan_position(self, pos: int):
        """Set next pan position."""
        self.setPanPosition(pos)

    def getPanPosition(self) -> int:
        """Query current pan position."""
        self.__send_command(b'PP ')
        resp = self.__get_response()
        # '* Current Pan position is 100' in verbose feedback mode, '* 100' in terse feedback mode
        try:
            return int(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for current pan position, device gave unexpected reply:\n  ' + resp)

    def setPanPosition(self, pos: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to go to a pan position."""
        resp = self.__send(b'PP' + str(int(pos)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting pan position, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    @property
    def tilt_position(self) -> int:
        """Query tilt position."""
        return self.getTiltPosition()

    @tilt_position.setter
    def tilt_position(self, pos: int):
        """Set next tilt position."""
        self.setTiltPosition(pos)

    def getTiltPosition(self) -> int:
        """Query current tilt position."""
        self.__send_command(b'TP ')
        resp = self.__get_response()
        # '* Current Tilt position is 100' in verbose feedback mode, '* 100' in terse feedback mode
        try:
            return int(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for current tilt position, device gave unexpected reply:\n  ' + resp)

    def setTiltPosition(self, pos: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to go to a tilt position."""
        resp = self.__send(b'TP' + str(int(pos)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting tilt position, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    # ************************** Target-Speed **************************
    @property
    def target_speeds(self) -> (int, int):
        """Query desired panning and tilting speeds."""
        return self.getTargetSpeeds()

    @target_speeds.setter
    def target_speeds(self, speeds: (int, int)):
        """Set the desired panning and tilting target speeds."""
        self.setTargetSpeeds(*speeds)

    def getTargetSpeeds(self) -> (int, int):
        """Query desired panning and tilting speeds."""
        return self.getPanTargetSpeed(), self.getTiltTargetSpeed()

    def setTargetSpeeds(self, pan: int, tilt: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Set target panning and tilting speeds."""
        resp1 = self.setPanTargetSpeed(pan, blocking=False, verbose=verbose)
        resp2 = self.setTiltTargetSpeed(tilt, blocking=blocking, verbose=verbose)
        return resp1 and resp2

    @property
    def pan_target_speed(self) -> int:
        """Query desired panning speed."""
        return self.getPanTargetSpeed()

    @pan_target_speed.setter
    def pan_target_speed(self, speed: int):
        """Set the desired panning target speed."""
        self.setPanTargetSpeed(speed)

    def getPanTargetSpeed(self) -> int:
        """Query desired panning speed, i.e. the ona at which PTU should pan."""
        self.__send_command(b'PS ')
        resp = self.__get_response()
        # '* Desired Pan speed is 2000 positions/sec' in verbose feedback mode, '* 2000' in terse feedback mode
        try:
            return int(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for target pan speed, device gave unexpected reply:\n  ' + resp)

    def setPanTargetSpeed(self, speed: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Set a target panning speed."""
        resp = self.__send(b'PS' + str(int(speed)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting pan target speed, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    @property
    def tilt_target_speed(self) -> int:
        """Query desired tilting speed."""
        return self.getTiltTargetSpeed()

    @tilt_target_speed.setter
    def tilt_target_speed(self, speed: int):
        """Set the desired tilting target speed."""
        self.setTiltTargetSpeed(speed)

    def getTiltTargetSpeed(self) -> int:
        """Query desired tilting speed, i.e. the one at which PTU should tilt."""
        self.__send_command(b'TS ')
        resp = self.__get_response()
        # '* Desired Tilt speed is 2000 positions/sec' in verbose feedback mode, '* 2000' in terse feedback mode
        try:
            return int(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for target tilt speed, device gave unexpected reply:\n  ' + resp)

    def setTiltTargetSpeed(self, speed: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Set a target tilting speed."""
        resp = self.__send(b'TS' + str(int(speed)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting tilt target speed, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    # ************************** Acceleration **************************
    @property
    def accelerations(self) -> (int, int):
        """Query desired panning and tilting accelerations."""
        return self.getAccelerations()

    @accelerations.setter
    def accelerations(self, acc: (int, int)):
        """Set desired panning and tilting accelerations."""
        self.setAccelerations(*acc)

    def getAccelerations(self) -> (int, int):
        """Query desired panning and tilting accelerations."""
        return self.getPanAcceleration(), self.getTiltAcceleration()

    def setAccelerations(self, pan: int, tilt: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to pan and tilt a specified number of arcseconds/seconds faster
        than its currently defined speed."""
        resp1 = self.setPanAcceleration(pan, blocking=False, verbose=verbose)
        resp2 = self.setTiltAcceleration(tilt, blocking=blocking, verbose=verbose)
        return resp1 and resp2

    @property
    def pan_acceleration(self) -> int:
        """Query desired panning acceleration."""
        return self.getPanAcceleration()

    @pan_acceleration.setter
    def pan_acceleration(self, acc: int):
        """Set desired panning acceleration."""
        self.setPanAcceleration(acc)

    def getPanAcceleration(self) -> int:
        """Query current panning acceleration."""
        self.__send_command(b'PA ')
        resp = self.__get_response()
        # '* Pan acceleration is 1000 positions/sec/sec' in verbose feedback mode, '* 1000' in terse feedback mode
        try:
            return int(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for current pan acceleration, device gave unexpected reply:\n  ' + resp)

    def setPanAcceleration(self, acc: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to pan a specified number of arcseconds/seconds faster than its
        currently defined speed."""
        resp = self.__send(b'PA' + str(int(acc)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting pan acceleration, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    @property
    def tilt_acceleration(self) -> int:
        """Query desired tilting acceleration."""
        return self.getTiltAcceleration()

    @tilt_acceleration.setter
    def tilt_acceleration(self, acc: int):
        """Set desired tilting acceleration."""
        self.setTiltAcceleration(acc)

    def getTiltAcceleration(self) -> int:
        """Query current tilting acceleration."""
        self.__send_command(b'TA ')
        resp = self.__get_response()
        # '* Tilt acceleration is 1000 positions/sec/sec' in verbose feedback mode, '* 1000' in terse feedback mode
        try:
            return int(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for current tilt acceleration, device gave unexpected reply:\n  ' + resp)

    def setTiltAcceleration(self, acc: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to tilt a specified number of arcseconds/seconds faster than its
        currently defined speed."""
        resp = self.__send(b'TA' + str(int(acc)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting tilt acceleration, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    # =================================================================================================================
    # --> Query-Only (CANNOT Set) ON-THE-FLY SETTINGS
    # =================================================================================================================

    # ************************* Actual-Speed ***************************
    @property
    def current_speeds(self) -> (int, int):
        """Query the (actual) current panning and tilting speeds.
        NOTE: You can only query the actual current speed, you can not set it. You can only set the target speeds."""
        return self.getPanCurrentSpeed(), self.getTiltCurrentSpeed()

    def getCurrentSpeeds(self) -> (int, int):
        """Query current panning and tilting speeds."""
        return self.getPanCurrentSpeed(), self.getTiltCurrentSpeed()

    def getPanCurrentSpeed(self) -> int:
        """Query current panning speed."""
        self.__send_command(b'PD ')
        resp = self.__get_response()
        # '* Current Pan speed is 450 positions/sec' in verbose feedback mode, '* 450' in terse feedback mode
        try:
            return int(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for current pan speed, device gave unexpected reply:\n  ' + resp)

    def getTiltCurrentSpeed(self) -> int:
        """Query current tilting speed."""
        self.__send_command(b'TD ')
        resp = self.__get_response()
        # '* Current Tilt speed is 450 positions/sec' in verbose feedback mode, '* 450' in terse feedback mode
        try:
            return int(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for current tilt speed, device gave unexpected reply:\n  ' + resp)

    # =================================================================================================================
    # --> Set-Only (CANNOT Query) ON-THE-FLY SETTINGS
    # =================================================================================================================

    # ************************ Position-Offset *************************
    def setPositionOffset(self, pan: int, tilt: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to move pan and tilt positions by a specified number of
        positions from current position."""
        resp1 = self.setPanPositionOffset(pan, blocking=False, verbose=verbose)
        resp2 = self.setTiltPositionOffset(tilt, blocking=blocking, verbose=verbose)
        return resp1 and resp2

    def setPanPositionOffset(self, offset: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to move pan position by a specified number of positions
        from current position."""
        resp = self.__send(b'PO' + str(int(offset)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting pan position offset, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    def setTiltPositionOffset(self, offset: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to move tilt position by a specified number of positions
        from current position."""
        resp = self.__send(b'TO' + str(int(offset)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting tilt position offset, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    # ************************ Relative-Speed **************************
    def setRelativeSpeed(self, pan: int, tilt: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to pan and tilt a specified number of arcseconds faster
        than its currently defined speed."""
        resp1 = self.setPanRelativeSpeed(pan, blocking=False, verbose=verbose)
        resp2 = self.setTiltRelativeSpeed(tilt, blocking=blocking, verbose=verbose)
        return resp1 and resp2

    def setPanRelativeSpeed(self, offset: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to pan a specified number of arcseconds faster than its
        currently defined speed."""
        resp = self.__send(b'PD' + str(int(offset)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting pan speed offset, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    def setTiltRelativeSpeed(self, offset: int, blocking: bool = True, verbose: bool = False) -> bool:
        """Command PTU to tilt a specified number of arcseconds faster than its
        currently defined speed."""
        resp = self.__send(b'TD' + str(int(offset)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting tilt speed offset, device gave unexpected reply:\n  ' + resp, file=sys.stderr)
            return False
        else:
            return True

    # ********************** Move-in-trajectory ************************
    def setTrajectory(self, steps: list, return_response: bool = False) -> float or (float, str):
        """Command PTU to execute a continuous trajectory with 'n' steps. Return the duration of the whole movement and,
        if return_response is True, also the response of the PTU."""
        if not isinstance(steps, list) or not all(isinstance(sum(val), int) for val in steps):
            raise TypeError("Argument must be a list of integers.")
        if not all(len(steps[k]) == 2 for k in range(len(steps))):
            raise TypeError("The input list must have shape nx2, where 'n' is the number of steps of the trajectory "
                            "and each step is a list that contains the pan and tilt displacements, respectively.")
        # Set slaved execution mode, if not already set
        reset = False
        if self.__execution == 'I':
            reset = True
            self.setExecutionMode('S')
        # Create string of commands
        command = ''.join(['PO' + str(pan) + ' TO' + str(tilt) + ' A ' for pan, tilt in steps])
        n = command.count('A')  # total number of steps
        # Send commands
        start = time.time()  # start = self.timestamp()
        self.__send_command(command.encode())
        # Get response
        resp = self.__get_response_ntimes(n * 3)  # for each step, 3 commands are sent: PO, TO and A.
        end = time.time()  # end = self.timestamp()
        if self.verbose:
            print('\n -> Sent:', command)
            print('Received:', resp.replace('*', ''))
            n_exe = resp.count('*') // 3  # number of steps actually executed
            print('N.B.: ' + ('ALL' if n_exe == n else 'ONLY') +
                  ' {} / {} steps of the trajectory were executed.'.format(n_exe, n))
        # Reset previous execution mode
        if reset:
            self.setExecutionMode('I')
        if return_response:
            return end - start, resp.replace('*', '')
        return end - start

    # =================================================================================================================
    # --> Useful Methods
    # =================================================================================================================

    # ************************ Wait-Execution **************************
    def wait(self, verbose: bool = False):
        """Command PTU to wait the end of execution of previous command."""
        self.__send_command(b'A ')
        self.__get_response()
        if self.verbose and verbose:
            print('...Wait for command to be executed...')

    # ************************* Stop-Moving ****************************
    def halt_pan(self) -> bool:
        """Command PTU to stop panning."""
        self.__send_command(b'HP ')
        return self.__get_response()[-1] == '*'

    def halt_tilt(self) -> bool:
        """Command PTU to stop tilting."""
        self.__send_command(b'HT ')
        return self.__get_response()[-1] == '*'

    def halt(self) -> bool:
        """Command PTU to stop moving both in pan and tilt."""
        self.__send_command(b'H ')
        return self.__get_response()[-1] == '*'

    # ************************ Recalibration ***************************
    def recalibrate_pan(self):
        """Recalibrate pan motor."""
        if self.verbose:
            print('Pan recalibration... ', end='')
        self.serial.timeout = 60
        self.__send_command(b'RP ')
        resp = self.__get_response()
        self.serial.timeout = self.__timeout
        if self.verbose:
            print(resp.replace('RP', '') + '\n')

    def recalibrate_tilt(self):
        """Recalibrate tilt motor."""
        if self.verbose:
            print('Tilt recalibration... ', end='')
        self.serial.timeout = 60
        self.__send_command(b'RT ')
        resp = self.__get_response()
        self.serial.timeout = self.__timeout
        if self.verbose:
            print(resp.replace('RT', '') + '\n')

    def recalibrate(self):
        """Recalibrate pan and tilt motors."""
        if self.verbose:
            print('Pan and Tilt recalibration... ', end='')
        self.serial.timeout = 60
        self.__send_command(b'R ')
        resp = self.__get_response()
        self.serial.timeout = self.__timeout
        if self.verbose:
            print(resp.replace('R', '') + '\n')

    def disable_reset(self):
        """Disable reset (axes recalibration) upon power-up."""
        self.__send_command(b'RD ')
        self.__get_response()

    def enable_reset(self):
        """Enable reset (axes recalibration) upon power-up."""
        self.__send_command(b'RE ')
        self.__get_response()

    def reset_mode(self) -> str:
        """Query current reset type at power-up."""
        self.__send_command(b'RQ ')
        resp = self.__get_response()
        return {'E': 'enabled', 'D': 'disabled', 'P': 'pan only', 'T': 'tilt only'}.get(resp[-1])

    def query_reset_speed(self) -> (int, int):
        """Query pan and tilt recalibration speed."""
        self.__send_command(b'RPS ')
        resp1 = self.__get_response()
        self.__send_command(b'RTS ')
        resp2 = self.__get_response()
        try:
            return int(resp1.split()[-1]), int(resp2.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for pan and tilt recalibration speed, device gave unexpected reply:\n  '
                               + resp1 + ' & ' + resp2)

    def set_reset_speed(self, speedP: int, speedT: int):
        """Set pan and tilt recalibration speed."""
        self.__send_command(b'RPS' + str(speedP).encode() + b' ')
        resp1 = self.__get_response()
        self.__send_command(b'RTS' + str(speedT).encode() + b' ')
        resp2 = self.__get_response()
        if '!' in resp1 + resp2:
            print('When setting pan and tilt recalibration speed, device gave unexpected reply:\n  '
                  + resp1 + ' & ' + resp2, file=sys.stderr)

    # ************************** Timestamp *****************************
    def timestamp(self) -> int:
        """Returns current timestamp count in seconds."""
        self.__send_command(b'CNT ')
        resp = self.__get_response()
        try:
            return int(resp.split()[-1])
        except ValueError:
            raise RuntimeError('When queried for current timestamp count, device gave unexpected reply:\n  ' + resp)

    def countfreq(self) -> str:
        """Returns frequency of timestamp counter (typically 90 MHz)."""
        self.__send_command(b'CNF ')
        resp = self.__get_response()
        try:
            return str(int(resp.split()[-1] * 10 ** -6)) + ' MHz'
        except ValueError:
            raise RuntimeError('When queried for timestamp counter frequency, device gave unexpected reply:\n  ' + resp)

    # ************************ Convert-Angle ***************************
    def angle2position(self, rotation: str, angle: float) -> int:
        """Converts an angle (in degrees) to the corresponding pan/tilt position displacement."""
        if not isinstance(rotation, str) and isinstance(angle, (float, int)):
            raise TypeError("Argument 'rotation' must be of type string, while 'angle' of type int or float.")
        res = self.resolutions
        resolution = {'P': res[0], 'T': res[1]}.get(rotation.upper())
        if not resolution:
            raise TypeError("Argument 'rotation' must be either 'P'=pan or 'T'=tilt.")
        return int(round(angle / (resolution / 60 ** 2)))

    def position2angle(self, rotation: str, pos: int) -> float:
        """Converts a pan/tilt position displacement to the corresponding angle (in degrees)."""
        if not isinstance(rotation, str) and isinstance(pos, int):
            raise TypeError("Argument 'rotation' must be of type string, while 'pos' of type int.")
        res = self.resolutions
        resolution = {'P': res[0], 'T': res[1]}.get(rotation.upper())
        if not resolution:
            raise TypeError("Argument 'rotation' must be either 'P'=pan or 'T'=tilt.")
        return round(pos * resolution / 60 ** 2, 2)

    # ************************* Position+Speed *************************
    def getPositionAndSpeed(self) -> (int, int, int):
        """Returns both pan and tilt current position and speed, along with current timestamp."""
        self.__send_command(b'BT ')
        resplist = self.__get_response().replace('BT', '').split()
        # '* P(-2300,550) S(600,200) 9278'  # TODO: check if this output is different in terse feedback mode
        try:
            pos = tuple(map(int, resplist[1][2:-1].split(',')))
            speed = tuple(map(int, resplist[2][2:-1].split(',')))
            timestamp = int(resplist[-1])
            return pos, speed, timestamp
        except ValueError:
            raise RuntimeError('When queried for current position and speed, device gave unexpected reply:\n  '
                               + ' '.join(resplist))

    def setPositionAndSpeed(self, posP: int, posT: int, speedP: int, speedT: int, blocking: bool = True,
                            verbose: bool = False) -> bool:
        """Command PTU to go to pan and tilt positions at particular pan and tilt target speeds."""
        resp = self.__send(b'B' + str(int(posP)).encode() + b',' + str(int(posT)).encode() + b','
                           + str(int(speedP)).encode() + b',' + str(int(speedT)).encode() + b' ', verbose)
        if blocking:
            self.wait(verbose=verbose)
        if resp[0] != '*':
            print('When setting pan and tilt position and target speed, device gave unexpected reply:\n  ' + resp,
                  file=sys.stderr)
            return False
        else:
            return True


# ===================================================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Utility Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ===================================================================================================================

def get_ports():
    return list(serial.tools.list_ports.comports())


def find_ptuport(verbose=True):
    ports = get_ports()
    n_ports = len(ports)
    ptu_port = 0
    if n_ports:
        if verbose:
            print("\nAvailable serial ports:")
        for i in range(n_ports):
            str_port = str(ports[i])
            if verbose:
                print(str(i + 1) + ")      " + str_port)
            if 'RS422' in str_port:
                ptu_port = str_port.split(' ')[0]
    else:
        raise Exception('No serial port is available.')
    if ptu_port:
        if verbose:
            print('\nFound PTU port at: ' + ptu_port)
        return ptu_port
    else:
        raise Exception('No PTU port found.')


def deg2pos(res, val):
    """Returns a value in positions (PTU steps), taking as input the resolution (in arcsec/pos) and
    the value to convert, expressed in degrees."""
    return int(round(val / (res * 60 ** -2)))


def deg2pos_tuple(res, tup):
    """Returns a tuple with elements expressed in positions (PTU steps), taking as input
    the resolution (in arcsec/pos) and the tuple to convert, with elements expressed in degrees (angular notation)."""
    return tuple(deg2pos(res, val) for val in tup)


def pos2deg(res, val):
    """Returns a value in degrees, taking as input the resolution (in arcsec/pos) and
    the value to convert, expressed in positions (PTU steps)."""
    return val * res * 60 ** -2


def pos2deg_tuple(res, tup):
    """Returns a tuple with elements expressed in degrees (angular notation), taking as input
    the resolution (in arcsec/pos) and the tuple to convert, with elements expressed in positions (PTU steps)."""
    return tuple(pos2deg(res, val) for val in tup)
