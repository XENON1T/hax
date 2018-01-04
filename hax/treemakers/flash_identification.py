import numpy as np
import hax
import pytz


class FlashIdentification(hax.minitrees.TreeMaker):
    """
    Identification of flashes during a dataset. Therefore it will extract trigger data checking for typical
    flashing properties. It will provide additional information for each event:
    - flashing_PMT
    - flashing_time (timestamp)
    - flashing_width (in seconds)
    - inside_flash: Events that are within the flash and should be cut in any case
    - nearest_flash: to set a time window around a flash which can be simply adjusted by a cut (in nseconds)
    """
    __version__ = '0.1'
    pax_version_independent = True

    def get_data(self, dataset, event_list=None):
        # Use 'all' pulses
        trigger_data = hax.trigger_data.get_trigger_data(dataset, select_data_types='count_of_all_pulses')
        self.transposed = trigger_data.T

        # Get BUSY_ON (channel id 255)
        self.BUSY_data = self.transposed[255]

        time_window = []
        window_biggest = []

        # Check in BUSY_ON channel if there where an increase "typical" for flashes and get the time-window in seconds
        for t in range(0, len(self.BUSY_data)):
            if self.BUSY_data[t] > 20:
                time_window.append(t)
            else:
                if len(time_window) <= 1:   # Ignore spikes
                    time_window = []
                else:
                    if len(time_window) > len(window_biggest):
                        window_biggest = time_window
                        time_window = []
                    else:
                        time_window = []
        self.flash_time_BUSY = np.nan
        self.flash_width = np.nan
        self.flash_time_highest_trig = np.nan
        self.flashing_PMT = np.nan
        self.flash_amplitude = np.nan
        self.flash_time_BUSY_first = np.nan
        # the thresholds are chosen to only select real flashes
        # get the time information from BUSY-channel
        if len(window_biggest) > 3:
            self.flash_time_BUSY = window_biggest[-1]
            self.flash_width = len(window_biggest)

            # check if there was also a "large" increase in one of the PMT channels
            self.flash_time_highest_trig = int(np.argmax(trigger_data)/len(self.transposed))

            # check if the two incidence match (roughly) in time
            if abs(self.flash_time_highest_trig - self.flash_time_highest_trig) < 200:
                self.flashing_PMT = np.argmax(trigger_data[self.flash_time_highest_trig])
                self.flash_amplitude = int(np.max(self.transposed[self.flashing_PMT]))

                # all flashes found so far a well beyond this threshold
                if self.flash_amplitude < 20000:
                    self.flashing_PMT = np.nan
                    self.flash_amplitude = np.nan
            else:
                self.flash_time_BUSY = np.nan
                self.flash_width = np.nan
                self.flash_time_highest_trig = np.nan
        # We need the run start time to find the time in run later
        self.run_start = hax.runs.get_run_info(dataset, 'start').replace(tzinfo=pytz.utc).timestamp()

        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)

    def extract_data(self, event):
        ret = {"inside_flash": False}
        ret["nearest_flash"] = np.nan
        ret["flashing_PMT"] = self.flashing_PMT
        ret["flashing_time"] = np.nan
        ret["flashing_width"] = np.nan

        if ~np.isnan(self.flashing_PMT):
            ret["flashing_time"] = int(self.run_start)/1e9 + self.flash_time_highest_trig
            ret["flashing_width"] = self.flash_width

            time_in_run_ns = int(event.start_time) - int(self.run_start)

            if time_in_run_ns in range(int((self.flash_time_highest_trig - self.flash_width)*1e9),
                                       int(self.flash_time_highest_trig*1e9)):
                ret["inside_flash"] = True

            if not ret["inside_flash"]:
                ret["nearest_flash"] = time_in_run_ns - int(self.flash_time_highest_trig*1e9)

        return ret
