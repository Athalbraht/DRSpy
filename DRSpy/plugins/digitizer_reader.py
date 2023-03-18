from dataclasses import dataclass, field
from itertools import cycle

import numpy as np

# import scipy.signal
import uproot


class RawDataContainer:
    """The class is used to read in the raw data from the root file into \
    python arrays with the help of the uproot package.

    It is embedded in the DigitizerEventData class and is not intended
    to be used separately.
    """

    def __init__(self, file: str):
        """TODO."""
        # print('Reading File...')
        self.file = file
        self.file_content = self.read_file()

        self.channels = self.get_channel_list()
        self.waveforms = np.array(self.get_waveforms())
        self.timestamps = np.array(self.get_timestamps())

    def read_file(self):
        """TODO."""
        return uproot.open(f"{self.file}:tree")

    def get_channel_list(self) -> list[int]:
        """Extract specific channel from branch name."""
        channels = [
            int(channel.split("_")[0].split("channel")[1])
            for channel in self.file_content.keys()
            if "waveforms" in channel
        ]
        channels = list(set(channels))
        return channels

    def get_waveforms(self) -> list[list[float]]:
        """TODO."""
        waveforms = [
            self.file_content[f"channel{channel}_waveforms"].array(library="np")
            for channel in self.channels
        ]
        return waveforms

    def get_timestamps(self) -> list[int]:
        """TODO."""
        timestamps = self.file_content["timestamp"].array(library="np")
        return timestamps


class Preprocessor:
    """The class preprocesses the raw waveforms before turning them into \
    DigitizerEvents.

    It is embedded in the DigitizerEventData class and is not intended
    to be used separately.
    """

    def __init__(self, RawDataContainer: RawDataContainer):
        """TODO."""
        self.RawDataContainer = RawDataContainer

    def subtract_baseline(self):
        """Calculate the baseline of eac waveform as the mean of the first 50 \
        samples of the waveform and subtracts it from the original waveform."""
        for i, channel in enumerate(self.RawDataContainer.channels):
            baselines = np.mean(
                self.RawDataContainer.waveforms[i][:, :50], axis=1, keepdims=True
            )
            self.RawDataContainer.waveforms[i] -= baselines
        return self

    def turn_into_volts(self):
        """Turn the y axis of the waveform into volts (12 bit digitizer and 1V \
        range --> divide Digitizer counts by 4096)."""
        for i, channel in enumerate(self.RawDataContainer.channels):
            self.RawDataContainer.waveforms[i] /= 4096
        return self

    def correct_timestamps(self) -> list[int]:
        """TODO."""
        # Find the indices where the timestamps reset
        reset_indices = (
            np.where(
                self.RawDataContainer.timestamps[:-1]
                > self.RawDataContainer.timestamps[1:]
            )[0]
            + 1
        )

        # Increment the timestamps by a one clock cycle every time they reset
        increment = 2**30
        for i, reset_index in enumerate(reset_indices):
            self.RawDataContainer.timestamps[reset_index:] += increment * (i + 1)
        return self

    def preprocess_raw_data(self):
        """Chains all the preprocessing methods and returns the preprocesed \
        RawDataContainer."""
        self.subtract_baseline().turn_into_volts().correct_timestamps()
        return self.RawDataContainer

    @classmethod
    def preprocess(cls, RawDataContainer):
        """Classmethod to create an instance of Preprocessor and apply the \
        preprocessing onto the RawDataContainer."""
        # print('Preprocessing...')
        preprocessor = cls(RawDataContainer)
        preprocessor.preprocess_raw_data()
        return preprocessor.RawDataContainer


# @dataclass(slots=True)
@dataclass()
class DigitizerEvent:
    """Class representing an event, characterised by its waveform and \
    timestamp.

    Contains method to return the amplitude of the waveform (simply the
    minimum samplepoint). It is embedded in the DigitizerEventData class
    and is not intended to be used separately.
    """

    waveform: list[float]
    timestamp: int
    amplitude: float = field(init=False)

    def __post_init__(self):
        """TODO."""
        self.amplitude = np.amin(self.waveform) * -1


class DigitizerEventData:
    """Class that creates the Digitizer events from a root file.

    The events are stored in a dictionary where the key stands for the
    channel number.
    """

    def __init__(self, data):
        """TODO."""
        self.data = data

    def create_events_dict(self):
        """Create the event dictionary."""
        # print('Creating event dict...')
        timestamps = cycle(self.data.timestamps)
        events_dict = {}

        for channel, waveforms in zip(self.data.channels, self.data.waveforms):
            events_dict[channel] = [
                (DigitizerEvent(waveform=waveform, timestamp=next(timestamps)))
                for waveform in waveforms
            ]
        return events_dict

    @classmethod
    def create_from_file(cls, file):
        """Classmethod with which the event dictionary is created from the root \
        file.

        This is the method wich should be used by the user to load the
        root data into python (see README)
        """
        raw_data = RawDataContainer(file)
        preprocessed_data = Preprocessor.preprocess(raw_data)
        return cls(preprocessed_data).create_events_dict()
