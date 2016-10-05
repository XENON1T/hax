import hax
import numpy as np

class DoubleScatter(hax.minitrees.TreeMaker):
    
###############################################################################
    
    """
    The search for double scatter events:
        double decays, afterpulses, and anything else that gets in our way
        if you have any questions contact Ted Berger (berget2@rpi.edu)
        
    The search proceeds as follows:
      * interaction[0] (int_0) provides s1_0 and s2_0
      * find additional interaction (int_1) to provide s1_1 and s2_1
        - loop through interactions (interactions store s1/s2 pairs in 
          descending size order, s2s on fast loop)
        Choice A) select first interaction with s1 != s1_0 AND s2 != s2_0
        Choice B) if Choice A doesn't exist, select first interaction with 
                  s1 != s1_0 AND s2 == s2_0
        Choice C) if Choice A and B don't exist ignore, this isn't a double 
                  scatter event
      * int_0 and int_1 ordered by s1.center_time to int_a and int_b 
        (int_a has s1 that happened first)
      
    The output provides the following variables attributed to specific peaks 
    (s1_a, s2_a, s1_b, s2_b), as well as specific interactions (int_a, int_b).
    
    Peak Output (for PEAK in [s1_a, s2_a, s1_b, s2_b, s1_2, s2_2]):
     - PEAK: The uncorrected area in pe of the peak
     - PEAK_area_fraction_top: The fraction of uncorrected area in the peak 
       seen by the top array
     - PEAK_center_time: The center_time in ns of the peak
     - PEAK_n_contributing_channels: The number of PMTs contributing to the 
       peak
     - PEAK_range_50p_area: The width of the peak (ns), duration of region that 
       contains 50% of the area of the peak
    
    Interaction Output (for INT in [int_a, int_b]):
     - INT_x: The x-position of this interaction (primary algorithm chosen by 
       pax, currently TopPatternFit)
     - INT_y: The y-position of this interaction
     - INT_z: The y-position of this interaction
     - INT_s1_area_correction: The multiplicative s1 area correction of this 
       interaction
     - INT_s2_area_correction: The multiplicative s2 area correction of this 
       interaction
     - INT_drift_time: The drift time in ns (pax units) of this interaction
     - INT_s1_pattern_fit: The s1 pattern fit (-log liklihood) of this 
       interaction
       
    DoubleScatter Specific Output:
     - ds_s1_b_n_distinct_channels: number of PMTs contributing to s1_b 
       distinct from the PMTs that contributed to s1_a
     - ds_second_s2: True if selected interactions have distinct s2s   

    """
    
    __version__ = '0.1.0'
    extra_branches = ['peaks.n_contributing_channels',
                      'peaks.center_time',
                      'peaks.area_fraction_top',
                      'peaks.range_area_decile*',
                      'interactions.s1_pattern_fit',
                      'peaks.hits*']
    
    def extract_data(self, event):
        event_data = dict()
        
        # If there are no interactions at all, we can't extract anything...
        if not len(event.interactions):
            return dict()
        
        # shortcuts for pax classes
        peaks = event.peaks
        interactions = event.interactions
        
####### Select Interactions for DoubleScatter Event #######
        
        # assume one scatter is interactions[0]
        int_0 = 0
        s1_0 = interactions[int_0].s1
        s2_0 = interactions[int_0].s2
        
        # find another scatter
        otherInts = [0,0]  
        for i, interaction in enumerate(interactions):
            if (interaction.s1 != s1_0 and interaction.s2 == s2_0 
               and otherInts[0] == 0):
                otherInts[0] = i
            elif (interaction.s1 != s1_0 and interaction.s2 != s2_0 
                 and otherInts[1] == 0):
                otherInts[1] = i
    
        # Distinction b/w single and double s2 scatters
        # Cut events without second s1
        if otherInts[1] != 0:
            s1_1 = interactions[otherInts[1]].s1
            s2_1 = interactions[otherInts[1]].s2
            int_1 = otherInts[1]
            ds_second_s2 = True
        elif otherInts[0] != 0:
            s1_1 = interactions[otherInts[0]].s1
            s2_1 = interactions[otherInts[0]].s2
            int_1 = otherInts[0]
            ds_second_s2 = False
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
            
        # Find additional s1s and s2s
        #  conventions fall apart, because we're no longer looking in 
        #  interactions its possible that s1_2/s2_2 is larger than
        #  s1_0/s2_0
        s1_2 = -1
        for s1 in event.s1s:
            if s1 not in [s1_a, s1_b]:
                s1_2 = s1
                break
                
        if s1_2 == -1:
            s1_2_area = 0
            s1_2_area_fraction_top = 0
            s1_2_center_time = 0
            s1_2_n_contributing_channels = 0
            s1_2_range_50p_area = 0
        else:
            s1_2_area = peaks[s1_2].area
            s1_2_area_fraction_top = peaks[s1_2].area_fraction_top
            s1_2_center_time = peaks[s1_2].center_time
            s1_2_n_contributing_channels = peaks[s1_2].n_contributing_channels
            s1_2_range_50p_area = peaks[s1_2].range_area_decile[5]
            
        s2_2 = -1
        for s2 in event.s2s:
            if s2 not in [s2_a, s2_b]:
                s2_2 = s2
                break
                
        if s2_2 == -1:
            s2_2_area = 0
            s2_2_area_fraction_top = 0
            s2_2_center_time = 0
            s2_2_n_contributing_channels = 0
            s2_2_range_50p_area = 0
        else:
            s2_2_area = peaks[s2_2].area
            s2_2_area_fraction_top = peaks[s2_2].area_fraction_top
            s2_2_center_time = peaks[s2_2].center_time
            s2_2_n_contributing_channels = peaks[s2_2].n_contributing_channels
            s2_2_range_50p_area = peaks[s2_2].range_area_decile[5]
            
####### Coompute DoubleScatter Specific Variables #######
            
        # Select largest hits on each channel in s10 and s11 peaks
        s1_a_hitChannels = []
        s1_a_hitAreas = []
        s1_a_hitTimes = []
        
        for hit in peaks[s1_a].hits:
            if hit.is_rejected: continue
            if hit.channel not in s1_a_hitChannels:
                s1_a_hitChannels.append(hit.channel)
                s1_a_hitAreas.append(hit.area)
                s1_a_hitTimes.append(hit.center)
            else:
                hitChannel_i = s1_a_hitChannels.index(hit.channel)
                if hit.area > s1_a_hitAreas[hitChannel_i]:
                    s1_a_hitAreas[hitChannel_i] = hit.area
                    s1_a_hitTimes[hitChannel_i] = hit.center
                    
        s1_b_hitChannels = []
        s1_b_hitAreas = []
        s1_b_hitTimes = []
        
        for hit in peaks[s1_b].hits:
            if hit.is_rejected: continue
            if hit.channel not in s1_b_hitChannels:
                s1_b_hitChannels.append(hit.channel)
                s1_b_hitAreas.append(hit.area)
                s1_b_hitTimes.append(hit.center)
            else:
                hitChannel_i = s1_b_hitChannels.index(hit.channel)
                if hit.area > s1_b_hitAreas[hitChannel_i]:
                    s1_b_hitAreas[hitChannel_i] = hit.area
                    s1_b_hitTimes[hitChannel_i] = hit.center
                    
        # count largest-hit channels in s1_b distinct from s1_a
        ds_s1_b_n_distinct_channels = 0
        for i, channel in enumerate(s1_b_hitChannels):
            if channel not in s1_a_hitChannels: 
                ds_s1_b_n_distinct_channels += 1        
                
####### Grab Data for Output #######
    
        event_data.update(dict( # Peak Data
                                s1_a = peaks[s1_a].area,
                                s1_a_area_fraction_top = peaks[s1_a].area_fraction_top,
                                s1_a_center_time = peaks[s1_a].center_time,
                                s1_a_n_contributing_channels = peaks[s1_a].n_contributing_channels,
                                s1_a_range_50p_area = peaks[s1_a].range_area_decile[5],
                                
                                s2_a = peaks[s2_a].area,
                                s2_a_area_fraction_top = peaks[s2_a].area_fraction_top,
                                s2_a_center_time = peaks[s2_a].center_time,
                                s2_a_n_contributing_channels = peaks[s2_a].n_contributing_channels,
                                s2_a_range_50p_area = peaks[s2_a].range_area_decile[5],
                               
                                s1_b = peaks[s1_b].area,
                                s1_b_area_fraction_top = peaks[s1_b].area_fraction_top,
                                s1_b_center_time = peaks[s1_b].center_time,
                                s1_b_n_contributing_channels = peaks[s1_b].n_contributing_channels,
                                s1_b_range_50p_area = peaks[s1_b].range_area_decile[5],
                                
                                s2_b = peaks[s2_b].area,
                                s2_b_area_fraction_top = peaks[s2_b].area_fraction_top,
                                s2_b_center_time = peaks[s2_b].center_time,
                                s2_b_n_contributing_channels = peaks[s2_b].n_contributing_channels,
                                s2_b_range_50p_area = peaks[s2_b].range_area_decile[5],
                
                                s1_2 = s1_2_area,
                                s1_2_area_fraction_top = s1_2_area_fraction_top,
                                s1_2_center_time = s1_2_center_time,
                                s1_2_n_contributing_channels = s1_2_n_contributing_channels,
                                s1_2_range_50p_area = s1_2_range_50p_area,
                
                                s2_2 = s2_2_area,
                                s2_2_area_fraction_top = s2_2_area_fraction_top,
                                s2_2_center_time = s2_2_center_time,
                                s2_2_n_contributing_channels = s2_2_n_contributing_channels,
                                s2_2_range_50p_area = s2_2_range_50p_area,
                
                                # Interaction Data                               
                                int_a_x = interactions[int_a].x,
                                int_a_y = interactions[int_a].y,
                                int_a_z = interactions[int_a].z,
                                int_a_s1_area_correction = interactions[int_a].s1_area_correction,
                                int_a_s2_area_correction = interactions[int_a].s2_area_correction,
                                int_a_drift_time = interactions[int_a].drift_time,
                                int_a_s1_pattern_fit = interactions[int_a].s1_pattern_fit,
                
                                int_b_x = interactions[int_b].x,
                                int_b_y = interactions[int_b].y,
                                int_b_z = interactions[int_b].z,
                                int_b_s1_area_correction = interactions[int_b].s1_area_correction,
                                int_b_s2_area_correction = interactions[int_b].s2_area_correction,
                                int_b_drift_time = interactions[int_b].drift_time,
                                int_b_s1_pattern_fit = interactions[int_b].s1_pattern_fit,
                            
                                # DoubleScatter Specific Data
                                ds_s1_b_n_distinct_channels = ds_s1_b_n_distinct_channels,
                                ds_second_s2 = ds_second_s2 ))
              
        return event_data        
