2.4.0
-----
* Corrections - v2.0, CorrectedDoubleS1Scatter - v2.0:
  * Updated e-lifetime v0.6 (`ref <https://xe1t-wiki.lngs.infn.it/doku.php?id=greene:electron_lifetime_update_180206>`_)
  * Update 3D FDC map for 4th period (#210)
* LoneSignals - v0.2
  * Lone s2 xy correction and corrected AFT (#209)
* New SE treemaker (#203, #207)
* Stricter blinding cut (#206)
* Unblind between s2 = [150, 200] (#208)
* TensorFlow bypass (#205)
  * Temporarily for Midway1 and MC

2.3.3
-----
* PositionReconstruction (1.1)
  * Add s2_pattern_fit for NN and bugfix s1_pattern_fit (#202)

2.3.2
-----
* Extended - v0.0.7
  * Add largest other S1 drift time variable (#201)
  * Add s1 and s2 maximum hit channel information (#200)

2.3.1
-----
* Fix calculation of S1 AFT below 10 pe #198

2.3.0
-----
* Corrections - v1.8, CorrectedDoubleS1Scatter - v1.1:
   * Updated e-lifetime v0.5 (`ref <https://xe1t-wiki.lngs.infn.it/doku.php?id=greene:electron_lifetime_update_180110>`_)
   * New cs1 from field corrected LCE map (#195)
   * Update time-dependent 3D FDC run bins for SR1 (#197)
* LoneSignals - v0.1, LoneSignalsPreS1 - v0.2:
   * More variables for AC (LoneSignals) analysis (#196)
* Use pax refactored S1 AFT probability calculation (#193)

2.2.3
-----
* More variables for AC (LoneSignals) analysis (#192)

2.2.2
-----
* Fix typo in variable init (8d33196)
* Bugfix conditional when loading maps (#191) 

2.2.0
-----
* New reconstruction and dependent variables: S1 AFT, S1 pattern (#174)
* More S1 width variables (#190)
* Corrections files input refactor (#187)
* Add functionality to specify event_list with haxer (#182)
* More descriptive arrays error (#171)

2.1.1
-----
* Update version for e-lifetime update (#172)

2.1.0
-----
* Latest corrections in double S1 scatters (#170) 
* PMT flash identification treemaker (#164) 

2.0.0
-----
* Better Unblinding and Unblinded Max-radius for wall events (#161, #168, #169)
* Small bug fix (#166, #162)
* Double Scatter TreeMaker (#163)

1.9.0
-----
* Time Dependent 3D FDC for SR1 (#158)
* Default Correction of S1 based on NN with 3D FDC (#156)
* New S2 correction map for SR1 (#154)
* Extra information added in Extended minitree for Kr83 idetification (#157)
* Safer unblind (#153)  

1.8.0
-----
* Update Correction treemaker (#151)
* 3D data driven FDC (#147)
* Treemaker for AC analysis (#149)
* A few bug fix (#152, #145, #148)


1.7.0
-----
* Add new, alternative 3D data driven FDC correction (#143)
* Updated Correction treemaker for MC compatibility (#144)
* Added corrections definitions to minitree metadata (#141)
* Fix S2 map version to 2.1 (#140)
* Skip strings in acquisition monitor files (#142)
* Various bug fixes (#134, #138, #137)


1.6.2
-----
* Fix MC minitree generation bug related to rundb (#136)


1.6.1
-----
* Fix second place blinding logic applied (#131)


1.6.0
-----
* Extended Minitrees (#118, #121, #122, #123, #126)
* Correction Minitree (#119)
* Blinding SR0 again (#130)
* TailCut Treemaker updated (#127)
* Previous event Basics Info (#114)
* Proximity minitree for MV analysis (#125)


1.5.0
-----
* Correction treemaker (#109)
* Automatic blinding instead of blinding taggs. (#100, #111)
* Style issues fixed, cleaner coding, add missing branches for full chain simulation (#107, #113)
* run Tags print out messages (#108, #110)
* S1 AFT cut (#102, #112)
* Add LNGS server (#105)


1.4.4
-----
* unblind multiple scatter


1.4.1
-----

* Fix release tags.

1.4.0
-----
* Update blinding (#74)
* Update slow control HTTP API interface (#64)
* Handle multiple raw data locations (#68)
* Access metadata (trigger&aqm) from special folders
* Small updates/fixes


1.3.0
-----
* Blinding (#54, #61)
* Tag-based run selection helper (#62)
* TotalProperties bugfix (#58)
* Proximity minitrees, acquisition monitor pulses access (#55)
* Init hax without runs db access (#27)
* Multi-run queries for get_run_info (#41)
* Flexible policy for patch releases (#59, #49)
* Fix for XENON100 data access
* Fix pickle minitree format


1.2.0
-----
* Loading of partial minitrees / extending existing minitree dataframes #51
* Miscellaneous convenience functions #50


1.1.1
-----
* cache_file option to minitrees, TotalPeakProperties treemaker #40


1.1.0
-----
* Out-of-core treemaking and preselections (#37)
* Double scatter treemaker (#36)
* Cut helpers and history tracking (#35)
* Option to load only minitrees which exist (#38) and don't make any minitree files (8cbe2ce2f)


1.0.0
-----
* MultipleRowExtractor: 0 to many rows- per event minitrees  (#33)
* PeakExtractor: convenient interface for peak-per-row minitrees (#33)
* Array-field support for minitrees (#32)
* Pickle as alternative minitree caching backend (#32)
* All minitrees get event_number and run_number (so we can always merge them)
* Minitree metadata includes hax version; option to require minimum hax version from minitrees.


0.4.2
-----
* Add trigger data support
* Fundamentals treemaker (loaded automatically), improvements to Basics and LargestPeakProperties treemaker


0.4.1
-----

* Slow control tweaks


0.4.0
-----

* Slow control variables
* Checks for different pax versions (#30)


0.3.4
-----

* (x,y) positions (#29)


0.3.3
-----

* Redo arbitrary database queries in run DB, but lots of small Makefile issues with release.


0.3.2
-----

* Allow arbitrary run database queries in update_dataset
* Minitrees: enable opening of minitrees without write permission
* `haxer --daemon` mode to watch for data and create minitrees


0.3.1
-----

- Get metadata from pax root file (e.g. version, any setting used for processing) with hax.paxroot.get_metadata(run_id)
- Minitrees:

  - Fix duplicate columns (#7)
  - Configurable output folder (#25)
  - Run number added to basics treemaker for XENON1T 
