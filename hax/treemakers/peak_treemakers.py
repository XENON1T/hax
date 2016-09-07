class PeakExtractor(MultipleRowExtractor):
    """Documentation goes here
    """
    __version__ = '0.0.2'
    # Default branch selection is EVERYTHING in peaks, overwrite for speed increase
    # Don't forget to include branches used in cuts
    extra_branches = ['peaks.*']
    peak_fields = ['area']
    event_cut_list = []
    peak_cut_list = []
    event_cut_string = 'True'
    peak_cut_string = 'True'
    stop_after = np.inf

    # Hacks for want of string support :'(
    peaktypes = dict(lone_hit=0, s1=1, s2=2, unknown=3)
    detectors = dict(tpc=0, veto=1, sum_wv=2, busy_on=3, busy_off=4)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_cut_string = self.build_cut_string(self.event_cut_list, 'event')
        self.peak_cut_string = self.build_cut_string(self.peak_cut_list, 'peak')
          
    
    def build_cut_string(self, cut_list, obj):
        '''
        Build a string of cuts that can be applied using eval() function.
        '''
        # If no cut is specified, always pass cut
        if len(cut_list) == 0:
            return 'True'
        # Check if user entered range_50p_area, since this won't work
        cut_list = [cut.replace('range_50p_area','range_area__decile[5]') for cut in cut_list]
        cut_string = '('
        for cut in cut_list[:-1]:
            cut_string += obj + '.' + cut + ') & ('
        cut_string += obj + '.' + cut_list[-1] + ')'
        return cut_string
    
    

    def extract_data(self, event):
        if event.event_number > self.stop_after:
            raise hax.paxroot.StopEventLoop()
        
        peak_data = []
        # Check if event passes cut
        if eval(self.build_cut_string(self.event_cut_list, 'event')):
            # Loop over peaks and check if peak passes cut
            for peak in event.peaks:
                if eval(self.peak_cut_string):
                    # Loop over properties and add them to _current_peak one by one
                    _current_peak = {}
                    for field in self.peak_fields:
                        # Deal with special cases
                        if field == 'range_50p_area':
                            _x = list(peak.range_area_decile)[5]
                        elif field in ('x', 'y'):
                            # In case of x and y need to get position from reconstructed_positions
                            for rp in peak.reconstructed_positions:
                                if rp.algorithm == 'PosRecTopPatternFit':
                                    _x = getattr(rp, field)
                                    break
                            else:
                                _x = float('nan')
                            # Change field name!
                            field = field + '_peak'
                        elif field == 'type':
                            _x = self.peaktypes.get(peak.type, -1)
                        elif field == 'detector':
                            _x = self.detectors.get(peak.detector, -1)
                        else:
                            _x = getattr(peak, field)  
                            
                        
                        _current_peak[field] = _x
                    # All properties added, now finish this peak
                    # The event number is necessary to join to event properties
                    _current_peak['event_number'] = event.event_number
                    peak_data.append(_current_peak)

            return peak_data
        else:
            # If event does not pass cut return empty list
            return []