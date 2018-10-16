hax - Handy Analysis tools for XENON
====================================

.. image:: https://readthedocs.org/projects/hax/badge/?version=latest
         :target: http://hax.readthedocs.org/en/latest/?badge=latest
         :alt: Documentation Status    
.. image:: https://zenodo.org/badge/50927571.svg
   :target: https://zenodo.org/badge/latestdoi/50927571

Source code: `https://github.com/XENON1T/hax`

Documentation: `Usage tutorials
<https://github.com/XENON1T/hax/tree/master/examples>`_
and `Auto-generated function reference 
<http://hax.readthedocs.org/en/latest/>`_.

Authors: Jelle Aalbers and Chris Tunnell


Tools for common analysis tasks on pax processed data, such as:

* Create pandas DataFrames with reduced data ('minitrees') out of pax ROOT files;
* Find datasets of a particular source and type in the XENON1T runs database;
* Apply event selections, either from [lax](https://github.com/XENON1T/lax) or your own custom ones;
* Load metadata from the slow control and trigger monitor databases.


Usage
=====
Please see the tutorials at `https://github.com/XENON1T/hax/tree/master/examples`. You can also look at code linked to wiki notes of analyses you like, or ask others in XENON for code samples.

Installation
============
You probably want to run this library in an analysis facility (Xecluster/Stockholm/Chicago), in which case hax has already been setup for you. However, if you do want to install hax locally:

* Pax, pyROOT and root_numpy are prerequisites: follow the usual pax installation procedure, then do `conda install root_numpy`.
* Setup as usual by running 'python setup.py develop' or, if you want to hide the code, 'python setup.py install'.

