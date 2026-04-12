# -*- coding: utf-8 -*-
"""
OpenSignals Reader
------------------

This package provides an OpenSignalsReader class for easy import of OpenSignals (r)evolution file
containing signals acquired with BITalino (r)evolution with automatic data conversion
using the official BITalino transfer functions.

Visit http://biosignalsplux.com/ and http://bitalino.com for more information about OpenSignals

..

:copyright: (c) 2018 by Pedro Gomes
:license: BSD 3-clause, see LICENSE for more details.


Notes
-----
..  Transfer functions are compatible for conversion data of BITalino (r)evolution
    devices only; older versions of BITalino are not supported

Author
------
..  Pedro Gomes, pgomes92@gmail.com

Notes Up To Version 0.2.1
-------------------------
..  Versions 0.2.1 and earlier of this package were part of the master thesis
    "Development of an Open-Source Python Toolbox for Heart Rate Variability (HRV)".
..	Thesis Author: Pedro Gomes, Master Student, University of Applied Sciences Hamburg
..  1st Supervisor: Prof. Dr. Petra Margaritoff, University of Applied Sciences Hamburg
..  2nd Supervisor: Hugo Silva, PhD, Instituto de Telecomunicacoes, PLUX wireless biosignals S.A.

Last Update
-----------
04.14.2026

"""

# Compatibility
from __future__ import absolute_import

# Imports
import json
import warnings
import matplotlib.pyplot as plt
import numpy as np

# Package imports
try:
    from .transfer_functions import bit
except (ImportError, ValueError, SystemError):
    from transfer_functions import bit


class OpenSignalsReader:
    """This class reads OpenSignals files (metadata & sensor data) and converts sensor data into original units
    (if required) and includes easy functions to plot sensor signals and to print acquisition data to the console.

    Methods
    -------
    read_file():
        Reads OpenSignals (r)evolution file.
    info():
        Prints acquisition's metadata to the console.
    plot():
        Plots sensor signals (customizable: plots all signals, plots all signals in preferred order or only individual
        signals.

    """

    TIME_LABEL = "Time (s)"
    CHANNEL_TYPE_ERROR = (
        "Please provide only channel numbers (int) or channel labels (str)."
    )

    def __init__(self, os_file=None, show=False, raw=False):
        # Initialize all member variables (acquisition metadata)
        self._init_member_variables()

        # Read file if provided
        if os_file is not None:
            self.load_file(os_file)
            if show:
                self.plot(raw=raw)

    def _init_member_variables(self):
        """Initialize all member variables to their default values."""
        self.file = None
        self.file_name = None
        self.device = None
        self.name = None
        self.mac = None
        self.sensors = None
        self.sampling_resolution = {}
        self.sampling_rate = None
        self.transfer_functions = None
        self.channels = None
        self.labels = {}
        self.digital = {}
        self.info = {}
        self._converted_signals = {}
        self._raw_signals = {}
        self._sensor_channels = {}
        self._channels_sensors = {}
        self._resolutions = {}
        self._labels = []

    def load_file(self, os_file):
        """Load an OpenSignals file and parse its contents.

        This method separates file loading from initialization, allowing
        for cleaner code and easier testing.
        """
        self._read_file(os_file)

    def _read_file(self, os_file):
        """Reads OpenSignals file (metadata & sensor data) and converts sensor data into original units (if required).

        Notes
        -----
        .. 	Signals of type 'RAW', 'CUSTOM or 'UNKNOWN' will not be converted; if 'raw' is true )
        .. 	If 'raw' is True, no signals will be converted.

        Parameters
        ----------
        os_file : str, file object
            OpenSignals file.
        raw : bool, optional
            If True, sensor data will be converted to original units

        """
        # Check input
        if type(os_file) is str:
            os_file = open(os_file, "r")
        else:
            raise TypeError(
                "Incompatible input. Please specify file path or file object."
            )

        self.file = os_file
        self.file_name = os_file.name

        # Check if OpenSignal file is being used
        if "OpenSignals" in str(self.file.readline()):
            self._read_metadata()
            self._read_sensordata()
        else:
            raise TypeError(
                "Provided file does not seem to be an OpenSignals (r)evolution file."
            )

        # Close file
        self.file.close()

    def _read_metadata(self):
        """Reads metadata from OpenSignals file stored in a string with JSON dictionary."""
        # Load metadata
        data = json.loads(self.file.readline().replace("#", ""))
        data = data[list(data.keys())[0]]

        if data["device"] != "biosignalsplux":
            # Load data if the acquisition device has been a BITalino
            # Save metadata
            self.sensors = [str(x) for x in data["sensor"]]
            for i, sens in enumerate(self.sensors):
                if "BITREV" in sens:
                    self.sensors[i] = sens.replace("BITREV", "")
                elif "BIT" in sens:
                    self.sensors[i] = sens.replace("BIT", "")

            # Save metadata in info dictionary
            self.info = {}
            self.info.update({"device name": data["device name"]})
            self.info.update({"column": [str(x) for x in data["column"]]})
            self.info.update({"time": data["time"]})
            self.info.update({"date": data["date"]})
            self.info.update({"comments": data["comments"]})
            self.info.update({"mac": data["device connection"]})
            self.info.update({"channels": data["channels"]})
            self.info.update({"firmware": data["firmware version"]})
            self.info.update({"device": data["device"]})
            self.info.update({"sampling rate": data["sampling rate"]})
            self.info.update({"resolution": data["resolution"]})
            self.info.update({"label": data["label"]})

            # Save critical data
            self.device = self.info["device"]
            self.name = self.info["device name"]
            self.mac = self.info["mac"]
            self.sampling_rate = self.info["sampling rate"]
            self.channels = self.info["channels"]
            self._labels = [str(x) for x in self.info["label"]]

            if self.device == "bitalino" or self.device == "bitalino_rev":
                self.ranges = bit.ranges
                self.units = bit.units
                self.transfer_functions = bit.transfer_functions

        else:
            raise ValueError(
                "Expected acquisition data from 'BITalino' device, but received from 'biosignalsplux'."
                "'biosignalsplux' is not supported at the moment."
            )

    def _read_sensordata(self):
        """Reads sensor data and calls required functions to convert into original units if required and possible.

        Notes
        -----
        ..	Raw sensor data from unsupported sensors (i.e. sensors not listed in the dictionaries in 'bitalino_tf')
            will not be converted. Raw sensor data will be returned.

        """
        data = np.loadtxt(self.file)

        # Get valid resolutions (6, 8, 10, 12, or 16 bit)
        self._resolutions = [
            int(x) for x in self.info["resolution"] if int(x) in [6, 8, 10, 12, 16]
        ]

        # Process sensors (columns are read right-to-left)
        processed_sensors = self._process_sensors(data)

        # Update sensor names
        self.sensors = processed_sensors

        # Prepare time vector
        self._time_vector()

        # Update channel label mappings
        self._update_channel_mappings()

    def _process_sensors(self, data):
        """Process sensor data and return list of processed sensor names."""
        num_sensors = len(self.sensors)
        processed_sensors = []
        existing_names = set()

        # Process sensors in reverse order (data columns are right-to-left)
        for idx, sensor in enumerate(reversed(self.sensors)):
            # Handle duplicate sensor names
            unique_name = self._get_unique_sensor_name(sensor, existing_names)
            existing_names.add(unique_name)

            # Calculate data column index (last column is -1, second-to-last is -2, etc.)
            col_idx = -(idx + 1)

            # Get corresponding resolution
            res_idx = (
                idx if idx < len(self._resolutions) else len(self._resolutions) - 1
            )
            resolution = self._resolutions[res_idx]

            # Store sensor data
            self.sampling_resolution[unique_name] = resolution
            self._raw_signals[unique_name] = data[:, col_idx]
            self._converted_signals[unique_name] = self._convert_data(
                unique_name, data[:, col_idx]
            )

            processed_sensors.append(unique_name)

        # Reverse to match original sensor order
        return list(reversed(processed_sensors))

    def _get_unique_sensor_name(self, sensor, existing_names):
        """Generate a unique sensor name by appending a number if needed."""
        if sensor not in existing_names:
            return sensor

        suffix = 1
        new_name = f"{sensor}{suffix}"
        while new_name in existing_names:
            suffix += 1
            new_name = f"{sensor}{suffix}"
        return new_name

    def _update_channel_mappings(self):
        """Update channel label mappings after sensors are processed."""
        for idx, sensor in enumerate(self.sensors):
            if idx < len(self._labels):
                self.labels[sensor] = self._labels[idx]
            if idx < len(self.channels):
                self._sensor_channels[self.channels[idx]] = sensor
                self._channels_sensors[sensor] = self.channels[idx]

    def _convert_data(self, sensor, samples):
        """Calls sensor specific transfer functions to convert raw sensor signals to their original units.

        Note
        ----
        Signals of type 'RAW', 'CUSTOM or 'UNKNOWN' will returned without conversion.

        Parameters
        ----------
        sensor : str
            OpenSignals sensor key (e.g. 'ECG' for Electrocardiography sensor).
        samples : array
            Raw sensor data.

        Returns
        -------
        samples : array
            Raw or converted sensor data.

        """
        sensor = "".join([x for x in sensor if not x.isdigit()])
        if "RAW" in sensor:
            return samples
        elif "CUSTOM" in sensor:
            return samples
        elif sensor in self.transfer_functions.keys():
            output = self.transfer_functions[sensor](
                samples, self.sampling_resolution[sensor]
            )
            return output
        else:
            return samples

    def _time_vector(self):
        """Computes time vector."""
        size = np.size(list(self._converted_signals.values())[0])
        self.t = np.linspace(
            0, float(size) / self.sampling_rate, size, 1.0 / self.sampling_rate
        )

    def _get_signal_data(self, sensors, use_raw=False):
        """Internal method to get signal data (raw or converted) for given sensors.

        Parameters
        ----------
        sensors : str, int, list, array
            Sensor label(s) or channel(s).
        use_raw : bool
            If True, return raw signals; otherwise return converted signals.

        Returns
        -------
        signal : array, dict
            Signal data. Returns array for single sensor or dictionary for multiple.
        """
        signals_dict = self._raw_signals if use_raw else self._converted_signals

        if sensors is None:
            return signals_dict

        if isinstance(sensors, list) and len(sensors) > 1:
            if all(isinstance(x, int) for x in sensors):
                _na_sensors = [
                    x for x in sensors if x not in self._sensor_channels.keys()
                ]
                if _na_sensors:
                    raise ValueError(
                        f"Could not find channel(s) {_na_sensors} in available channels."
                    )
                keys = [self._sensor_channels[key] for key in sensors]
                return {key: signals_dict[key] for key in keys}
            elif all(isinstance(x, str) for x in sensors):
                _na_sensors = [x for x in sensors if x not in self.sensors]
                if _na_sensors:
                    raise ValueError(
                        f"Could not find {_na_sensors} in available sensor data."
                    )
                return {key: signals_dict[key] for key in sensors}
            else:
                raise TypeError(self.CHANNEL_TYPE_ERROR)

        if isinstance(sensors, list) and len(sensors) == 1:
            sensors = sensors[0]

        if isinstance(sensors, int):
            if sensors in self._sensor_channels:
                return signals_dict[self._sensor_channels[sensors]]
            raise ValueError(f"Could not find channel {sensors} in available channels.")

        if isinstance(sensors, str):
            if sensors in self.sensors:
                return signals_dict[sensors]
            raise ValueError(f"Could not find '{sensors}' in available sensor data.")

        raise TypeError(self.CHANNEL_TYPE_ERROR)

    def signal(self, sensors=None):
        """Returns converted signal data from selected sensor(s) using the respective label(s) or channel(s).

        Parameters
        ----------
        sensors : str, int, list, array
            Sensor label(s) or channel(s).

        Return
        ------
        signal : array, dict
            Converted sensor data of the selected label(s) or channel(s). Returns array for single sensor signal
            or dictionary when returning multiple signals.

        Raises
        ------
        ValueError
            If channel numbers do not exist (array input).
        ValueError
            If sensor labels do not exist (array input).
        TypeError
            If sensors contain a mix of sensors labels and channel numbers (array input).
        ValueError
            If channel number does not exist (int or str input).
        ValueError
            If sensor label does not exist (int or str input).
        """
        return self._get_signal_data(sensors, use_raw=False)

    def raw(self, sensors=None):
        """Returns raw digital signal data from selected sensor(s) using the respective label(s) or channel(s).

        Parameters
        ----------
        sensors : str, int, list, array
            Sensor label(s) or channel(s).

        Return
        ------
        signal : array, dict
            Raw sensor data of the selected label(s) or channel(s). Returns array for single sensor signal
            or dictionary when returning multiple signals.

        Raises
        ------
        ValueError
            If channel numbers do not exist (array input).
        ValueError
            If sensor labels do not exist (array input).
        TypeError
            If sensors contain a mix of sensors labels and channel numbers (array input).
        ValueError
            If channel number does not exist (int or str input).
        ValueError
            If sensor label does not exist (int or str input).
        """
        return self._get_signal_data(sensors, use_raw=True)

    def plot(self, sensors=None, raw=False, interval=None, show=True, figsize=None):
        """Plots sensor signals.

        Notes
        -----
        ..	Signals are plotted in provided order in array (you can define the order yourself).
        .. 	If no input is provided, all available sensor signals will be plotted.

        Parameters
        ----------
        sensors : str, array, optional
            Sensor keys of the signals to be plotted. If None, all available signals will be plotted.
        raw : bool, optional
            If true, plot raw sensor data, otherwise plot converted sensor signals.
        interval : list, array (2-elements)
            Visualization interval [x_min, x_max]
        show : bool, optional
            If True, shows the figure of the plotted signals; default: True
        figsize : 2-element array, optional
            Figsize as used in pyplot figures ([width, height]); default: None (will be adjusted dynamically)

        Raises
        ------
        ValueError
            If channel number does not exist.
        ValueError
            If sensor label does not exist.
        TypeError
            If sensors contain a mix of sensors labels and channel numbers.
        """
        sensors = self._normalize_sensors(sensors)

        if interval is None:
            interval = [0, self.t[-1]]
        elif interval[0] >= interval[1] or interval[0] < 0 or interval[1] < 0:
            interval = [0, self.t[-1]]
            warnings.warn("Invalid interval. Interval set to [0, max_signal_duration].")

        if len(sensors) > 1:
            self._plot_multiple_signals(sensors, raw, interval, figsize, show)
        else:
            self._plot_single_signal(sensors[0], raw, interval, figsize, show)

    def _normalize_sensors(self, sensors):
        """Normalize sensor input to a list of sensor labels."""
        if sensors is None:
            return self.sensors if len(self.sensors) > 1 else [self.sensors[0]]

        if isinstance(sensors, list) and len(sensors) > 1:
            if all(isinstance(x, int) for x in sensors):
                _na_sensors = [
                    x for x in sensors if x not in self._sensor_channels.keys()
                ]
                if _na_sensors:
                    raise ValueError(
                        f"Could not find channel(s) {_na_sensors} in available channels."
                    )
                return [self._sensor_channels[key] for key in sensors]
            elif all(isinstance(x, str) for x in sensors):
                _na_sensors = [x for x in sensors if x not in self.sensors]
                if _na_sensors:
                    raise ValueError(
                        f"Could not find {_na_sensors} in available sensor data."
                    )
                return [x for x in sensors if x in self.sensors]
            else:
                raise TypeError(self.CHANNEL_TYPE_ERROR)
        else:
            if isinstance(sensors, list):
                sensors = sensors[0]

            if isinstance(sensors, int):
                if sensors in self._sensor_channels.keys():
                    return [self._sensor_channels[sensors]]
                else:
                    raise ValueError(
                        f"Could not find channel {sensors} in available channels."
                    )

            if isinstance(sensors, str):
                if sensors in self.sensors:
                    return [sensors]
                else:
                    raise ValueError(
                        f"Could not find '{sensors}' in available sensor data."
                    )

            raise TypeError(self.CHANNEL_TYPE_ERROR)

    def _plot_multiple_signals(self, sensors, raw, interval, figsize, show):
        """Plot multiple sensor signals."""
        if figsize is None:
            figsize = (12, 6)

        n_plots = len(sensors)

        if n_plots in [1, 2, 3]:
            rows, columns = n_plots, 1
        elif n_plots == 4:
            rows, columns = 2, 2
        elif n_plots in [5, 6]:
            rows, columns = 3, 2
        else:
            columns = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / columns))

        fig, axs = plt.subplots(nrows=rows, ncols=columns, sharex=True, figsize=figsize)
        fig.suptitle(
            f"Sensor Signals\n(OpenSignals (r)evolution file: {self.file_name})"
        )

        if rows == 1 and columns == 1:
            axs = np.array([[axs]])
        elif rows == 1:
            axs = axs.reshape(1, -1)
        elif columns == 1:
            axs = axs.reshape(-1, 1)

        for idx, sens in enumerate(sensors):
            row = idx // columns
            col = idx % columns

            if idx >= n_plots:
                axs[row][col].set_visible(False)
                continue

            self._plot_signal(axs[row][col], sens, raw, interval)

        for col in range(columns):
            axs[rows - 1][col].set_xlabel(self.TIME_LABEL)

        plt.tight_layout()

        if show:
            plt.show()

    def _plot_single_signal(self, sensor, raw, interval, figsize, show):
        """Plot a single sensor signal."""
        if figsize is None:
            figsize = (12, 4)

        channel = self._channels_sensors[sensor]
        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"Sensor Signals - OpenSignals (r)evolution file: {self.file_name}"
        )
        ax = fig.add_subplot(111)

        self._plot_signal(ax, sensor, raw, interval)
        ax.set_xlabel(self.TIME_LABEL)

        if show:
            plt.show()

    def _plot_signal(self, ax, sensor, raw, interval):
        """Plot a single signal on the given axis."""
        if raw:
            signal = self._raw_signals[sensor]
            units = "RAW"
            ranges = [0, 2 ** self.sampling_resolution[sensor]]
        else:
            key = self._get_key(sensor)
            signal = self._converted_signals[sensor]
            units = self.units[key]
            ranges = self.ranges[key]

        ax.plot(self.t, signal)
        ax.axis([interval[0], interval[1], ranges[0], ranges[1]])
        channel = self._channels_sensors[sensor]
        ax.set_ylabel(f"CH{channel} - {sensor} ({units})")
        ax.grid()

    def _get_key(self, sensor):
        """ """
        key = "".join([x for x in sensor if not x.isdigit()])
        if key not in self.transfer_functions.keys():
            return "RAW"
        else:
            return key


if __name__ == "__main__":
    # Create OpenSignals object and read file with automatic signal conversion and plot results
    acq = OpenSignalsReader("./files/SampleECG.txt")

    # Plot all signals
    acq.plot()

    # Plot all raw signals
    acq.plot(raw=True)

    # Plot ECG signal using channel number (ECG Channel on BITalino (r)evolution boards = 2)
    acq.plot(2)

    # Plot ECG signal using channel label
    acq.plot("ECG")

    # Access signal using channel number
    acq.signal(2)

    # Access signal using channel label
    acq.signal("ECG")

    # Access raw signal using channel number (ECG Channel on BITalino (r)evolution boards = 2)
    acq.raw(2)

    # Access raw signal using channel label
    acq.raw("ECG")
