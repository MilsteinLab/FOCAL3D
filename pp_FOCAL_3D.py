"""
Daniel Djayakarsana
2018-05-17

pp_FOCAL_3D.py does post processing on the output from main_FOCAL_3D.py.
Gets statistics/additional information on clusters that would take up
extra time or memory. 

These methods cannot be easily integrated into the main file without
disturbing the flow.
"""
import numpy as np
from scipy.spatial import ConvexHull

def conv_hull_vol(loc_coords,clus_idx):
    """
    Uses convex hull to find volume of each cluster. Ignores any clusters
    labeled with idx 0 or -1 since we assume this is noise.

    loc_coords, should be for 3D coordinates (nxd) array
    clus_idx, should be (nx1) array

    outputs convex hull volume
    """
    # makes sure loc_coords and clus_idx are the same size
    if loc_coords.shape[0] != clus_idx.shape[0]:
        raise ValueError('loc_coords number of rows is different from clus_idx')
    
    # runs convex hull on each cluster and returns the volume in a python list
    ch_vol = []
    # loops over each cluster
    for i in range(1,np.amax(clus_idx)+1):
        i_clus = loc_coords[clus_idx==i]
        try:
            i_hull = ConvexHull(i_clus)
        except:
            # not enough points in cluster to do convex hull
            ch_vol.append(None)
            continue

        ch_vol.append(i_hull.volume)

    return ch_vol
