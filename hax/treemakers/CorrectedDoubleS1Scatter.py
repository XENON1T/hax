# Get pax values
import pax
from pax import configuration
pax_config = configuration.load_configuration('XENON1T')
# setup hax for Midway
import hax
from hax import runs
from hax.minitrees import TreeMaker
from pax.InterpolatingMap import InterpolatingMap
import pax.utils
import numpy as np
from scipy.interpolate import interp1d

class CorrectedDoubleS1Scatter(TreeMaker):
    """Applies high level corrections which are used in DoubleS1Scatter analyses.
    ##############################################################################
    Be carefull, this treemaker was developed for Kr83m analysis. It will probably need modifications for other analysis.
    The search for double scatter events: made by Ted Berger
        double decays, afterpulses, and anything else that gets in our way
        if you have any questions contact Ted Berger (berget2@rpi.edu)
        
    The search proceeds as follows:
      * interaction[0] (int_0) provides s1_0 and s2_0
      * find additional interaction (int_1) to provide s1_1 and s2_1
      
      - loop through interactions (interactions store s1/s2 pairs in
      descending size order, s2s on fast loop)
      Choice A) select first interaction with s1 != s1_0 AND s2 != s2_0
      Choice B) if Choice A doesn't exist, select first interaction with s1 != s1_0 AND s2 == s2_0
      Choice C) if Choice A and B don't exist ignore, this isn't a double scatter event
      
      * int_0 and int_1 ordered by s1.center_time to int_a and int_b (int_a has s1 that happened first)
      
      The output provides the following variables attributed to specific peaks 
      (s1_a, s2_a, s1_b, s2_b), as well as specific interactions (int_a, int_b).
    
    ### Peak Output (for PEAK in [s1_a, s2_a, s1_b, s2_b]):
     - PEAK: The uncorrected area in pe of the peak
     - PEAK_center_time: The center_time in ns of the peak

    ### Interaction Output (for INT in [int_a, int_b]):
    - INT_x_pax: The x-position of this interaction (primary algorithm chosen by pax, currently TopPatternFit)
    - INT_y_pax: The y-position of this interaction
    - INT_z_pax: the z-position of this interaction 
    
    - INT_z_observed: The z-position of this interaction without correction
    - INT_drift_time : The drift time of the interaction
    
    - INT_x_nn : The x-position of this interaction with NeutralNetwork Analysis
    - INT_y_nn : The y-position of this interaction with NeutralNetwork Analysis 
    - INT_r_nn : The r-position of this interaction with NeutralNetwork Analysis  

    ### Corrected Signal Output (for INT in [int_a, int_b]):
    # Data-driven 3D position correction 
    - INT_r_correction_3d_nn : correction value for r position using NN
    - INT_r_3d_nn : The corrected interaction r coordinate using NN 
    - INT_x_3d_nn : The corrected interaction x coordinate using NN 
    - INT_y_3d_nn : The corrected interaction y coordinate using NN 
    - INT_z_correction_3d_nn : correction value for z position using NN
    - INT_z_3d_nn : The corrected interaction z coordinate using NN 
    
    # LCE correction on S1 using NN FDC xyz-corrected position
    - s1_INT_xyz_correction_nn_fdc_3d : correction value for s1 signals using the INT_nn position :
    
    /!\ Two way of doing things for the S1_b signal : 
      - either used the int_a position to correct s1_b signal (since S1_a and S1_b are closed in time) : by default
      - either used the int_b position to correct s1_b signal (but most of the time the s2_b signal (and thus z position) is badly reconstructed )
    
    - cS1_a and cS1_b : the corrected s1_a signal using int_a_3d_nn corrected position
    - cS1_b_int_b: the corrected s1_b signal using int_b_3d_nn corrected position
    
    ### DoubleScatter Specific Output:
    - ds_s1_b_n_distinct_channels: number of PMTs contributing to s1_b distinct from the PMTs that contributed to s1_a
    - ds_s1_dt : delay time between s1_a_center_time and s1_b_center_time
    - ds_second_s2: 1 if selected interactions have distinct s2s
    """
    __version__ = '1.0'
    extra_branches = ['peaks.n_contributing_channels',
                      'peaks.center_time',
                      'peaks.s2_saturation_correction',
                      'interactions.s2_lifetime_correction',
                      'peaks.area_fraction_top',
                      'peaks.area',
                      'peaks.reconstructed_positions*',
                      'interactions.x',
                      'interactions.y',
                      'interactions.z',
                      'interactions.r_correction',
                      'interactions.z_correction',
                      'interactions.drift_time',
                      'start_time',
                      'peaks.hits*',
                      'interactions.s1_pattern_fit'
                     ]
    extra_metadata = hax.config['corrections_definitions']
    # Electron Lifetime: hopefully doc was pulled in hax.init.
    # Otherwise get it here at significantly higher DB cost
    try:
        elife_correction_doc = runs.corrections_docs['hax_electron_lifetime']
        extra_metadata['electron_lifetime_version'] = elife_correction_doc['version']
        elife_interpolation = interp1d(elife_correction_doc['times'],
                                       elife_correction_doc['electron_lifetimes'])
    except Exception as e:
        elife_interpolation = None
        print("No electron lifetime document found. Continuing without.")
        print(e)
        
    # Load Correction Map
    loaded_xy_map_name = None
    loaded_3d_fdc_map_name = None
    loaded_lce_map_nn_fdc_3d_name = None
    fdc_3d_map = None
    lce_map_nn_fdc_3d = None
    xy_map = None
    
    def get_correction(self, correction_name):
        """Return the file to use for a correction"""
        if ('corrections_definitions' not in hax.config) \
                or (correction_name not in hax.config['corrections_definitions']):
            return None

        for entry in hax.config['corrections_definitions'][correction_name]:
            if 'run_min' not in entry or self.run_number < entry['run_min']:
                continue
            if 'run_max' not in entry or self.run_number <= entry['run_max']:
                if 'correction' in entry:
                    return entry['correction']
        return None

    def load_map(self, name, loaded_map, loaded_name):
        wanted_map_name = self.get_correction(name)
        if loaded_name != wanted_map_name:
            map_path = pax.utils.data_file_name(wanted_map_name)
            return InterpolatingMap(map_path), wanted_map_name
        else:
            return loaded_map, loaded_name

    def extract_data(self, event):
        result = dict()

        # If there are no interactions cannot do anything
        if not len(event.interactions):
            return result

        # shortcuts for pax classes
        peaks = event.peaks
        interactions = event.interactions
        
        ######### Select Interactions for DoubleScatter Event ########
        # assume one scatter is interactions[0]
        int_0 = 0
        s1_0 = interactions[int_0].s1
        s2_0 = interactions[int_0].s2
        
        # find another scatter
        otherInts = [0,0]  
        for i, interaction in enumerate(interactions):
            if (interaction.s1 != s1_0 and interaction.s2 == s2_0 and otherInts[0] == 0):
                otherInts[0] = i
            elif (interaction.s1 != s1_0 and interaction.s2 != s2_0 and otherInts[1] == 0):
                otherInts[1] = i
    
        # Distinction b/w single and double s2 scatters
        # Cut events without second s1
        if otherInts[1] != 0:
            s1_1 = interactions[otherInts[1]].s1
            s2_1 = interactions[otherInts[1]].s2
            int_1 = otherInts[1]
            ds_second_s2 = 1
        elif otherInts[0] != 0:
            s1_1 = interactions[otherInts[0]].s1
            s2_1 = interactions[otherInts[0]].s2
            int_1 = otherInts[0]
            ds_second_s2 = 0
        else: return dict()
                                
        # order s1s/interactions by time
        if peaks[s1_0].center_time <= peaks[s1_1].center_time:
            s1_a = s1_0
            s1_b = s1_1
            s2_a = s2_0
            s2_b = s2_1
            int_a = int_0
            int_b = int_1
        else:
            s1_a = s1_1
            s1_b = s1_0
            s2_a = s2_1
            s2_b = s2_0
            int_a = int_1
            int_b = int_0
            
        # Additional s1s and s2s removed! see v0.1.0
        result['s1_a']=peaks[s1_a].area
        result['s1_a_center_time'] = peaks[s1_a].center_time
        result['s1_a_area_fraction_top']=peaks[s1_a].area_fraction_top
        
        result['s2_a']=peaks[s2_a].area
        result['s2_a_center_time'] = peaks[s2_a].center_time
        result['s2_a_bottom']=(1.0-peaks[s2_a].area_fraction_top)*peaks[s2_a].area
        result['s2_a_area_fraction_top']=peaks[s2_a].area_fraction_top

        result['s1_b']=peaks[s1_b].area
        result['s1_b_center_time'] = peaks[s1_b].center_time
        result['s1_b_area_fraction_top']=peaks[s1_b].area_fraction_top
        
        result['s2_b']=peaks[s2_b].area
        result['s2_b_center_time'] = peaks[s2_b].center_time
        result['s2_b_bottom']=(1.0-peaks[s2_b].area_fraction_top)*peaks[s2_b].area
        result['s2_b_area_fraction_top']=peaks[s2_b].area_fraction_top
        
        result['ds_second_s2']=ds_second_s2
        
        # Drift Time 
        result['int_a_drift_time']=result['s2_a_center_time']- result['s1_a_center_time']
        result['int_b_drift_time']=result['s2_b_center_time']- result['s1_b_center_time']

        # Pax position (TpF)
        result['int_a_x_pax']=interactions[int_a].x
        result['int_a_y_pax']=interactions[int_a].y
        result['int_a_z_pax']=interactions[int_a].z

        result['int_b_x_pax']=interactions[int_b].x
        result['int_b_y_pax']=interactions[int_b].y
        result['int_b_z_pax']=interactions[int_b].z
        ######### Compute DoubleScatter Specific Variables #########
        # Select largest hits on each channel in s10 and s11 peaks
        s1_a_hitChannels = []
        s1_a_hitAreas = []
        for hit in peaks[s1_a].hits:
            if hit.is_rejected: continue
            if hit.channel not in s1_a_hitChannels:
                s1_a_hitChannels.append(hit.channel)
                s1_a_hitAreas.append(hit.area)
            else:
                hitChannel_i = s1_a_hitChannels.index(hit.channel)
                if hit.area > s1_a_hitAreas[hitChannel_i]:
                    s1_a_hitAreas[hitChannel_i] = hit.area
        s1_b_hitChannels = []
        s1_b_hitAreas = []
        
        for hit in peaks[s1_b].hits:
            if hit.is_rejected: continue
            if hit.channel not in s1_b_hitChannels:
                s1_b_hitChannels.append(hit.channel)
                s1_b_hitAreas.append(hit.area)
            else:
                hitChannel_i = s1_b_hitChannels.index(hit.channel)
                if hit.area > s1_b_hitAreas[hitChannel_i]:
                    s1_b_hitAreas[hitChannel_i] = hit.area
                    
        # count largest-hit channels in s1_b distinct from s1_a
        ds_s1_b_n_distinct_channels = 0
        for i, channel in enumerate(s1_b_hitChannels):
            if channel not in s1_a_hitChannels:
                ds_s1_b_n_distinct_channels += 1             
        result['ds_s1_b_n_distinct_channels'] = ds_s1_b_n_distinct_channels 
        result['ds_s1_dt'] = peaks[s1_b].center_time - peaks[s1_a].center_time
   
        ######### Correction Map ##############
        # Check that the correct S2 map is loaded and change if not
        self.xy_map, self.loaded_xy_map_name = self.load_map("s2_xy_map",self.xy_map, self.loaded_xy_map_name)
        # Load the 3D data driven FDC map
        self.fdc_3d_map, self.loaded_3d_fdc_map_name = self.load_map("fdc_3d",
                                                                     self.fdc_3d_map,
                                                                     self.loaded_3d_fdc_map_name)
        # Load the LCE map for NN 3D FDC
        self.lce_map_nn_fdc_3d, self.loaded_lce_map_nn_fdc_3d_name = self.load_map("s1_lce_map_nn_fdc_3d",
                                                                                   self.lce_map_nn_fdc_3d,
                                                                                   self.loaded_lce_map_nn_fdc_3d_name)

        # Need the observed ('uncorrected') position.
        # pax Interaction positions are corrected so lookup the
        # uncorrected positions in the ReconstructedPosition objects
        for rp in peaks[s2_a].reconstructed_positions:
            if rp.algorithm == 'PosRecNeuralNet':
                result['int_a_x_nn'] = rp.x
                result['int_a_y_nn'] = rp.y
                result['int_a_r_nn'] = np.sqrt(rp.x ** 2 + rp.y ** 2)
                int_a_x_observed=rp.x
                int_a_y_observed=rp.y
        for rp in peaks[s2_b].reconstructed_positions:
            if rp.algorithm == 'PosRecNeuralNet':
                result['int_b_x_nn'] = rp.x
                result['int_b_y_nn'] = rp.y
                result['int_b_r_nn'] = np.sqrt(rp.x ** 2 + rp.y ** 2)

        int_a_z = interactions[int_a].z - interactions[int_a].z_correction
        result['int_a_z_observed'] = int_a_z
        int_b_z = interactions[int_b].z - interactions[int_b].z_correction
        result['int_b_z_observed'] = int_b_z
    
        # Correct S2_a. No correction for S2_b because, S2_b is mostly backgroud events, or S2_b ==S2_a
        result['s2_a_xy_correction_tot'] = (1.0 /
                                          self.xy_map.get_value(int_a_x_observed, int_a_y_observed))
        result['s2_a_xy_correction_top'] = (1.0 /
                                          self.xy_map.get_value(int_a_x_observed, int_a_y_observed, map_name='map_top'))
        result['s2_a_xy_correction_bottom'] = (1.0 /
                                             self.xy_map.get_value(int_a_x_observed, int_a_y_observed, map_name='map_bottom'))
      
        # include electron lifetime correction
        if self.mc_data:
            wanted_electron_lifetime = self.get_correction("mc_electron_lifetime_liquid")
            result['s2_lifetime_correction'] = np.exp((result['int_a_drift_time']/1e3) / wanted_electron_lifetime)

        elif self.elife_interpolation is not None:
            # Ugh, numpy time types...
            ts = ((self.run_start - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's'))
            result['ts'] = ts
            self.electron_lifetime = self.elife_interpolation(ts)
            result['s2_lifetime_correction'] = np.exp((result['int_a_drift_time']/1e3)/self.electron_lifetime)
        else:
            result['s2_lifetime_correction'] = 1.
        # Combine all the s2 corrections for S2_a
        s2_a_correction = (result['s2_lifetime_correction'] *result['s2_a_xy_correction_tot'])
        s2_a_top_correction = (result['s2_lifetime_correction'] * result['s2_a_xy_correction_top'])
        s2_a_bottom_correction = (result['s2_lifetime_correction'] * result['s2_a_xy_correction_bottom'])
        
        result['cs2_a'] = peaks[s2_a].area * s2_a_correction
        result['cs2_a_top'] = peaks[s2_a].area * peaks[s2_a].area_fraction_top * s2_a_top_correction
        result['cs2_a_bottom'] = peaks[s2_a].area * (1.0 -  peaks[s2_a].area_fraction_top) * s2_a_bottom_correction
        
        # FDC: Apply the (new) 3D data driven FDC, using NN positions
        # Int_a Position
        algo='nn'
        result['int_a_r_correction_3d_' + algo] = self.fdc_3d_map.get_value(result['int_a_x_' + algo],
                                                                          result['int_a_y_' + algo],
                                                                          int_a_z)
        result['int_a_r_3d_' + algo] = result['int_a_r_' + algo] + result['int_a_r_correction_3d_' + algo]
        
        result['int_a_x_3d_' + algo] =\
                result['int_a_x_' + algo] * (result['int_a_r_3d_' + algo] / result['int_a_r_' + algo])
        
        result['int_a_y_3d_' + algo] =\
                result['int_a_y_' + algo] * (result['int_a_r_3d_' + algo] / result['int_a_r_' + algo])
        
        if abs(int_a_z) > abs(result['int_a_r_correction_3d_' + algo]):
            result['int_a_z_3d_' + algo] = -np.sqrt(int_a_z ** 2 - result['int_a_r_correction_3d_' + algo] ** 2)
        else:
            result['int_a_z_3d_' + algo] = int_a_z
        
        result['int_a_z_correction_3d_' + algo] = result['int_a_z_3d_' + algo] - int_a_z
        
        
        # Int_b Position
        result['int_b_r_correction_3d_' + algo] = self.fdc_3d_map.get_value(result['int_b_x_' + algo],
                                                                          result['int_b_y_' + algo],
                                                                          int_b_z)
        result['int_b_r_3d_' + algo] = result['int_b_r_' + algo] + result['int_b_r_correction_3d_' + algo]
        
        result['int_b_x_3d_' + algo] =\
            result['int_b_x_' + algo] * (result['int_b_r_3d_' + algo] / result['int_b_r_' + algo])
        
        result['int_b_y_3d_' + algo] =\
            result['int_b_y_' + algo] * (result['int_b_r_3d_' + algo] / result['int_b_r_' + algo])

        if abs(int_b_z) > abs(result['int_b_r_correction_3d_' + algo]):
            result['int_b_z_3d_' + algo] = -np.sqrt(int_b_z ** 2 - result['int_b_r_correction_3d_' + algo] ** 2)
        else:
            result['int_b_z_3d_' + algo] = int_b_z

        result['int_b_z_correction_3d_' + algo] = result['int_b_z_3d_' + algo] - int_b_z
                 
        # Apply LCE (light collection efficiency correction to s1)
        
        result['s1_int_a_xyz_correction_nn_fdc_3d'] = 1 / self.lce_map_nn_fdc_3d.get_value(result['int_a_x_3d_nn'],
                                                                                     result['int_a_y_3d_nn'],
                                                                                     result['int_a_z_3d_nn'])
        
        result['cs1_a'] = peaks[s1_a].area * result['s1_int_a_xyz_correction_nn_fdc_3d']
        result['cs1_b'] = peaks[s1_b].area * result['s1_int_a_xyz_correction_nn_fdc_3d']

        result['s1_int_b_xyz_correction_nn_fdc_3d'] = 1 / self.lce_map_nn_fdc_3d.get_value(result['int_b_x_3d_nn'],
                                                                                     result['int_b_y_3d_nn'],
                                                                                     result['int_b_z_3d_nn'])
        
        result['cs1_b_int_b'] = peaks[s1_b].area * result['s1_int_b_xyz_correction_nn_fdc_3d']
        
        return result
