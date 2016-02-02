
#include "TFile.h"
#include "TTree.h"
#include "TObject.h"
#include "TString.h"
#include <vector>






#ifndef INTERACTION
#define INTERACTION
 
class Interaction : public TObject {

public:
	Float_t  drift_time;
	Int_t  s1;
	Float_t  s1_area_correction;
	Float_t  s1_pattern_fit;
	Int_t  s2;
	Float_t  s2_area_correction;
	Float_t  x;
	TString  xy_posrec_algorithm;
	Float_t  xy_posrec_goodness_of_fit;
	Float_t  xy_posrec_ndf;
	Float_t  y;
	Float_t  z;

    ClassDef(Interaction, 412);
};

#endif







#ifndef HIT
#define HIT
 
class Hit : public TObject {

public:
	Float_t  area;
	Float_t  center;
	Int_t  channel;
	Int_t  found_in_pulse;
	Float_t  height;
	Int_t  index_of_maximum;
	Bool_t  is_rejected;
	Int_t  left;
	Int_t  n_saturated;
	Float_t  noise_sigma;
	Int_t  right;
	Float_t  sum_absolute_deviation;

    ClassDef(Hit, 412);
};

#endif





#ifndef RECONSTRUCTEDPOSITION
#define RECONSTRUCTEDPOSITION
 
class ReconstructedPosition : public TObject {

public:
	TString  algorithm;
	Float_t  goodness_of_fit;
	Float_t  ndf;
	Float_t  x;
	Float_t  y;
	Float_t  z;

    ClassDef(ReconstructedPosition, 412);
};

#endif



#ifndef PEAK
#define PEAK
 
class Peak : public TObject {

public:
	Float_t  area;
	Float_t  area_fraction_top;
	Float_t  area_midpoint;
	Double_t  area_per_channel[243];
	Float_t  birthing_split_fraction;
	Float_t  birthing_split_goodness;
	Float_t  bottom_hitpattern_spread;
	Float_t  center_time;
	TString  detector;
	Float_t  height;
	Float_t  hit_time_mean;
	Float_t  hit_time_std;
	std::vector <Hit>  hits;
	Float_t  hits_fraction_top;
	Short_t  hits_per_channel[243];
	Int_t  index_of_maximum;
	Float_t  interior_split_fraction;
	Float_t  interior_split_goodness;
	Int_t  left;
	Int_t  lone_hit_channel;
	Float_t  mean_amplitude_to_noise;
	Int_t  n_contributing_channels;
	Int_t  n_contributing_channels_top;
	Int_t  n_hits;
	Int_t  n_noise_pulses;
	Int_t  n_saturated_channels;
	Short_t  n_saturated_per_channel[243];
	Int_t  n_saturated_samples;
	Double_t  range_area_decile[11];
	std::vector <ReconstructedPosition>  reconstructed_positions;
	Int_t  right;
	Float_t  sum_waveform[251];
	Float_t  sum_waveform_top[251];
	Float_t  top_hitpattern_spread;
	TString  type;

    ClassDef(Peak, 412);
};

#endif



#ifndef EVENT
#define EVENT
 
class Event : public TObject {

public:
	TString  dataset_name;
	Long64_t  event_number;
	std::vector <Interaction>  interactions;
	Bool_t  is_channel_suspicious[243];
	Short_t  lone_hits_per_channel[243];
	Short_t  lone_hits_per_channel_before[243];
	Int_t  n_channels;
	Short_t  n_hits_rejected[243];
	Short_t  noise_pulses_in[243];
	std::vector <Peak>  peaks;
	Int_t  sample_duration;
	Long64_t  start_time;
	Long64_t  stop_time;
	std::vector <Int_t> s1s;
	std::vector <Int_t> s2s;

    ClassDef(Event, 412);
};

#endif

