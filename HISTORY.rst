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
