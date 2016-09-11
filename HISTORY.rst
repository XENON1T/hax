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
