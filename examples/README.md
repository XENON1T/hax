Hax tutorials and examples
==========================

This directory contains tutorials and examples for using hax:
  * `01_introduction`: Basic dataset and event selection
  * `02_getting_serious`: More details, e.g. selecting many datasets and using cuts from lax
  * `03_advanced_features`: Custom minitrees, slow control interface, etc.

The `old` directory contains a few tutorials of advanced features that aren't documented elsewhere. They looked like they still ran, but are not really maintained as much.

Installation
------------
The tutorial notebooks will only run on midway. There are three ways to run notebooks there:

  * [Midway batch jobs](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:analysis:tutorial:jupyter_notebooks#using_jupyter_notebooks_remotely_midway_recommended_for_most_analyses)
  * [XENON1T jupyterhub](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:analysis:beginnersguide#the_midway_jupyterhub)
  * Midway login node. This is not meant for heavy-duty analyses, and IT people will kill your processes if they take up too much CPU or memory. However, it can be quite nice in the European morning when the US is still asleep...
    * Pick a magic port number that nobody is using. I'll use 12345 in the instructions below, but probably someone is using that already.
    * Setup the environment (one-time only)
      * Login to one of the midway nodes (e.g. ssh username@midway2-login1.rcc.uchicago.edu)
      * Add this to your ~/.bashrc file: `export PATH="/project/lgrandi/anaconda3/bin:$PATH"`
    * Setup the jupyter server (one-time only, or rather, once every time you get killed or the server gets rebooted)
      * screen -S my_notebook
      * source activate pax_head
      * jupyter notebook --no-browser --port=12345
      * CTRL+A (detatches the screen
      * exit (logs out)
    * Connect to the jupyter server (every time)
      * ssh -L localhost:12345:localhost:12345 username@midway2-login1.rcc.uchicago.edu
      

