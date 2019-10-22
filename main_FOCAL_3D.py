"""
Daniel Djayakarsana
2017-11-07
main_FOCAL_3D.py generalizes FOCAL to 3D, also works for 2D

Ensure the localization file is drift corrected and the first three columns
are (x,y,z)[nm]. Other columns aren't needed for FOCAL, but will be kept.
Last column will be cluster number with -1 indicating noise.

Python 3.x must be used because True and False are automatically treated
as 1 and 0. To use this in 2.x, must manually change True and False to 1/0
scipy >= 0.17.0

Python should be 64bit as this is memory intensive.
del self."variable" to save memory
"""
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage import label, find_objects
from scipy.ndimage import sum as ndsum
from scipy import stats
import binned_statistic_64bit

class densityMap:
    """
    densityMap class is the main class to perform FOCAL, a clustering
    algorithm
    """
    def __init__(self,loc_table,res_sr,min_L,bool_3D,min_C):
        # loc_table format (x,y,*z) followed by other cols not needed for FOCAL
        # add a column of 0 for cluster labeling
        self.loc_table = np.column_stack((loc_table,np.zeros(loc_table.shape[0])))
        self.loc_x = loc_table[:,0]
        self.loc_y = loc_table[:,1]

        # super resolution pixel size
        self.res_sr = res_sr

        # density threshold
        self.min_L = min_L
        self.min_C = min_C

        # checks if user specifies 2D or 3D loc_table
        if bool_3D:
            self.loc_z = loc_table[:,2]
            self.z_min = np.amin(self.loc_z)
            self.z_max = np.amax(self.loc_z)
            self.z_edges = np.arange(self.z_min,self.z_max+self.res_sr,self.res_sr)
        else:
            self.loc_z = None

        # gets min and max value for each dimension
        self.x_min = np.amin(self.loc_x)
        self.x_max = np.amax(self.loc_x)
        self.x_edges = np.arange(self.x_min,self.x_max+self.res_sr,self.res_sr)
        self.y_min = np.amin(self.loc_y)
        self.y_max = np.amax(self.loc_y)
        self.y_edges = np.arange(self.y_min,self.y_max+self.res_sr,self.res_sr)
       
    def gen_pix_sr(self):
        # NOTE: run this first
        # pixelizes localization according to SR size
        # can optimize to use linearized bin number
        
        # sets up array of pixels
        # SR_idx gives respective bin numbers for later use
        if self.loc_z is not None:
            # 3D case
            self.SR_bins, _, self.SR_idx = binned_statistic_64bit.binned_statistic_dd(
                    np.column_stack((self.loc_x,self.loc_y,self.loc_z)),None,
                    'count',bins=(self.x_edges,self.y_edges,self.z_edges),
                    expand_binnumbers=True)
        else:
            # 2D case
            self.SR_bins, _, _, self.SR_idx = stats.binned_statistic_2d(
                    self.loc_x,self.loc_y,None,'count',bins=(self.x_edges,
                        self.y_edges),expand_binnumbers=True)

    def gen_den_map(self):
        # NOTE: run gen_pix_sr before this method
        # 1. takes the SR pixelized image then creates a density map

        # bins 3x3(2D)x3(3D) surrounding pixels if the center pixel is nonzero,
        # if zero, it stays zero 
        if self.loc_z is not None:
            # 3D case
            self.kernel = np.ones((3,3,3))
        else:
            # 2D case
            self.kernel = np.ones((3,3))
            
        self.den_map = convolve(self.SR_bins,self.kernel,mode='constant',
                cval=0.0)
        self.den_map[self.SR_bins==0] = 0

        # no longer need SR_bins
        del self.SR_bins

    def gen_core_pix(self):
        # NOTE: run gen_den_map before this method
        # 1. labels cores (higher than min_L)
        # 2. labels borders (neighbour connectivity core points)

        # labels core above min_L (bool)
        self.den_th = self.den_map>=self.min_L

        # connectivity for kernel
        if self.loc_z is not None:
            # 3D case
            # 6 connected
            self.kern_conn = np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],
                [1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]])
        else:
            # 2D case
            # 4 connected
            self.kern_conn = np.array([[0,1,0],[1,1,1],[0,1,0]])

        # label border points around core based on connectivity
        # 1. get all possible border points around core as bool
        # 2. get all nonzero density pixels as bool
        # 3. get the intersection of 1&2 to label border points
        self.den_th = convolve(self.den_th,self.kern_conn,mode='constant')
        self.den_map = self.den_map>0
        self.den_th = self.den_th & self.den_map

        # no longer needs den_map
        del self.den_map

    def process_clusters(self):
        # NOTE: run gen_core_pix before this method
        # 1. Find all clusters
        # 2. Throw away clusters that are too small
        # 3. Throw away clusters that are (n-1)D
        # for loops here are never bigger than the number of clusters
        # does NOT go over each localization.

        # finds all the clusters
        # 4 connected
        #s = [[1,1,1],[1,1,1],[1,1,1]]
        self.clus_label, self.clus_nb = label(self.den_th)
#        del self.den_th # no longer needed
        self.clus_slice = find_objects(self.clus_label)

    def process_clusters1(self):
        # throw away cluster sizes smaller than min_C and relabel
        # gets the size of each cluster
        clus_sz = ndsum(self.den_th,self.clus_label,index=np.arange(1,
            self.clus_nb+1))
        clus_rej = np.where(clus_sz < self.min_C)[0]
        for i in [self.clus_slice[x] for x in clus_rej]:
            self.den_th[i] = False
        self.clus_label, self.clus_nb = label(self.den_th)

        # cluster sizes in pixels
        self.clus_sz = ndsum(self.den_th,self.clus_label,index=np.arange(
            1,self.clus_nb+1)).astype(int)

        self.keep_clus = range(-1,self.clus_nb+1)

#old method changed 2018-04-10
#    def process_clusters1(self):
#        # loops over each cluster
#        counter = 0 # keep track of deleted clusters
#        for idx,i_clus in enumerate(self.clus_slice, start=1):
#            clus_rm = False
#            # throw away clusters that are smaller than min_C
#            if np.count_nonzero(self.clus_label[i_clus]) < self.min_C:
#                self.clus_label[i_clus] = -1
#                clus_rm = True
#                
#            # throw away clusters that are (n-1)D
#            if [j for j in self.clus_label[i_clus].shape if j == 1]:
#                self.clus_label[i_clus] = -1
#                clus_rm = True
#
#            # renumerates clusters
#            if clus_rm:
#                self.clus_label[self.clus_label >= idx - counter] -= 1
#                counter += 1
#
#        self.clus_nb -= counter
#        self.keep_clus = range(-1,self.clus_nb+1)

    def process_clusters2(self):
        # loop to renumerate clusters to compensate for prior deletion
        # currently this is the CPU bottleneck
#        for i,j in enumerate(self.keep_clus,start=1):
#            self.clus_label[self.clus_label == j] = i

        pass
        # range of clusters (for plotting)

    def process_clusters3(self):
        # sets all 0 to -1 to indicate noise
        del self.den_th # no longer needed
        self.clus_label[self.clus_label == 0] = -1
            
    def label_loc(self):
        # NOTE: run process_cluster before this method
        # labels the localization table according to cluster number
        # if not in a cluster, label is -1 for noise
        
        # Check if each localization belongs in a cluster and labels
        # it in the last column
        self.loc_table[:,self.loc_table.shape[1]-1] = self.clus_label[tuple(
            self.SR_idx-1)]

        # old implementation
#        for idx,i in enumerate(self.SR_idx.T):
#            if self.clus_label[tuple(i-1)]:
#                self.loc_table[idx,self.loc_table.shape[1]-1] = self.clus[tuple(i-1)]
