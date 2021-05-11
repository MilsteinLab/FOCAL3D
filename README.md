# FOCAL3D
Clustering Algorithm for single-molecular localization microscopy data

Requirements: Python 3 as is when installed with Anaconda 

There is a change in lines 4 and 5 for the file titled “binned_statistic_64bit.py”  since “scipy._lib.six” is apparently no longer being supported by python. The required changes are:


4> from six import callable

5> from six.moves import xrange
