"""
Daniel Djayakarsana
2018-04-02

parameter_FC_rnd.py tries different parameters for 3D DBSCAN (scipy
implementation), FOCAL tree and convolution implementation

Current Implementation:
    2018-02-27
    No noise considerations, only does for speed and number of clusters detected
    vs real number of clusters (some clusters may overlap in simulation)
    Use constant density of clusters so clustering parameters don't change

    2018-03-26
    Add noise considerations
    We also test the effects of different parameters

    2018-04-11
    No more focal_tree
    for H2B data from 3D Voronoi

    2018-08-09
    Adjusted to run on simulated data
    Also optional check for accuracy, sensitivity and specificity
    NOTE: treat -1 as noise when doing the optional check
"""
import numpy as np
import main_FOCAL_3D as F3D
#import sim3DLocRndVarySize_BlinkingDyes as s3D #blinking dyes, localizations uniformly scattered in cluster
#import sim3DLocRndVarySize_BlinkingDyes_GaussianScatter as s3D #blinking dyes, localizations scattered about dye centre
#import sim3DLocRndVarySize_Feb_2019_2 as s3D
from sklearn.cluster import DBSCAN
#import psutil
import os
#import itertools
import time
import matplotlib.pyplot as plt
from PyQt5.QtCore import pyqtSignal, QObject
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import sum as ndsum
from scipy.spatial.distance import pdist,cdist
from tqdm import tqdm_gui
#from itertools import combinations
#import tkinter as tk
#from tkinter import filedialog

#class Signal(QObject):
#    
#    update = pyqtSignal(int) 
#    method_called = pyqtSignal(bool)
#    
#update_signal = Signal()
#method_signal = Signal()
#progress_max = None

def FOCAL_r_para(coords,min_L,min_C,res_SR,save_tab=False,gnd_tr=None):
    """
    tests FOCAL raster parameters
    """
    #print(coords)
#    print("Method is being called by {}".format(os.getpid()))
#    global count 
#    if count == 1:
#        raise ZeroDivisionError
    #print("Method called with parameter, min_L: {}, min_C: {}, res_SR: {}".format(min_L, min_C, res_SR))
#    path = "C:\\Users\\Admin\\Downloads\\FOCALDATA"
#    os.chdir(path)
    start_time = time.time()
    if coords.shape[1] == 2:
        focal_r = F3D.densityMap(coords,res_SR,min_L,False,min_C) 
    if coords.shape[1] == 3:
        focal_r = F3D.densityMap(coords,res_SR,min_L,True,min_C) 
    focal_r.gen_pix_sr()
    focal_r.gen_den_map()
    focal_r.gen_core_pix()
    focal_r.process_clusters()
    focal_r.process_clusters1()
#    vol_clus = ndsum(focal_r.den_th,focal_r.clus_label,index=np.arange(
#        1,focal_r.clus_nb+1))
    focal_r.process_clusters3()
    time_taken = time.time() - start_time

    # save localization table if desired
    # and plot
    if save_tab:
        focal_r.label_loc()
        np.savetxt('focalLocTabSR{}minL{}minC{}.txt'.format(res_SR,min_L,min_C),
                focal_r.loc_table)
        #focal_plot(focal_r,min_L,min_C,res_SR) #uncomment if you want to see plotting of clusters found


       #DOES NOT SEEM TO WORK-CHECK 
    # check for accuracy, sensitivity (tpr) and specificity (spc) - CHANGE THIS TO USE JACCARD INDEX
    if gnd_tr is not None:
        foc_clus = focal_r.loc_table[:,-1]!=-1
        
        gnd_tr = gnd_tr!=-1 
        
        
        tp = np.count_nonzero(foc_clus&gnd_tr)
        tn = np.count_nonzero(~foc_clus&~gnd_tr)
        p = np.count_nonzero(gnd_tr)
        n = gnd_tr.shape[0]-p
        acc = (tp+tn)/gnd_tr.shape[0]
        tpr = tp/p
        # ensure there's no divide by zero error
        if n == 0:
            spc = None
        else:
           spc = tn/n
        return time_taken, focal_r.clus_nb, acc, tpr, spc
    
    #print("The number of clusters is {}".format(focal_r.clus_nb))
    return time_taken, focal_r.clus_nb#, vol_clus




def DBSCAN_para(coords,d_eps,d_minPts,save_tab = False, gnd_tr=None):
    """
    tests DBSCAN parameters
    """
    start_time = time.time()
    db = DBSCAN(eps=d_eps,min_samples=d_minPts).fit(coords)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    time_taken = time.time() - start_time
    if save_tab:
        
        loc_table = np.column_stack((coords,np.zeros(coords.shape[0])))
        loc_table[:,loc_table.shape[1]-1] = labels
        np.set_printoptions(suppress = True)
#        np.savetxt('DBscan_LocTab_dEps{}minPts{}.txt'.format(d_eps,d_minPts),
#                loc_table)
        
        
    # check for accuracy, sensitivity (tpr) and specificity (spc)
    if gnd_tr is not None:
        lab_clus = labels!=-1
        gnd_tr = gnd_tr!=-1
        tp = np.count_nonzero(lab_clus&gnd_tr)
        tn = np.count_nonzero(~lab_clus&~gnd_tr)
        p = np.count_nonzero(gnd_tr)
        n = gnd_tr.shape[0]-p
        acc = (tp+tn)/gnd_tr.shape[0]
        tpr = tp/p
        # ensure there's no divide by zero error
        if n == 0:
            spc = None
        else:
            spc = tn/n
        return time_taken, n_clusters_, acc, tpr, spc
    
    return time_taken, n_clusters_

#def rnd_uni_minL(nb_loc,x_max,y_max,z_max):
#    """
#    Generates random localization in a cubic volume with a uniform
#    distribution
#    This is used to find a minL value
#    """
#    rnd_loc = np.random.uniform((0,0,0),(x_max,y_max,z_max),(nb_loc,3))
#
#    return min_L

def focal_plot(focal_r,min_L,min_C,res_SR):
    # plots focal for debugging
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    
    # black is noise
    coords = focal_r.loc_table[:,0:3]
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0,1,len(focal_r.keep_clus))]
    for k, col in zip(focal_r.keep_clus, colors):
        if k == -1:
            # Black is noise
            col = [0,0,0,1]
    
        xyz = coords[focal_r.loc_table[:,focal_r.loc_table.shape[1]-1] == k]
        ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],'o',markerfacecolor=tuple(col),
                markeredgecolor='k', markersize = 6)
    
    plt.title('FOCAL - Number of Clusters: %d' %focal_r.clus_nb)
    plt.savefig('focalLocTabSR{}minL{}minC{}.png'.format(res_SR,min_L,min_C))
    plt.show()
    plt.close() #remove if you want to see plots



def scan_FOCAL(l_start, l_end, l_step, c_start, c_end, c_step, sr_start, sr_end, sr_step, file_pth):    
    loc_table = np.loadtxt(file_pth)


    x_max = np.max(loc_table[:,0])
    y_max = np.max(loc_table[:,1]) 
    if loc_table.shape[1] == 2:
        min_L_rnd = np.random.uniform((0,0),(x_max,y_max),(loc_table.shape[0],2))

    if loc_table.shape[1] == 3:
        z_max = np.max(loc_table[:,2])
        min_L_rnd = np.random.uniform((0,0,0),(x_max,y_max,z_max),(
        loc_table.shape[0],3))
    
    min_l_lst = [] #A keeps track of the min_L that lead to zero clusters. 
    min_l_grid = [] #Is a list that tracks the min L of a specific grid size and resets after each grid size. 
    grid_lst = [size for size in range(sr_start, sr_end, sr_step)]
    min_c_lst = [c for c in range(c_start, c_end, c_step)]

    # loop over FOCAL parameters
    # minL is now determined by a certain method
    min_L_start = l_start #User defined
    min_L_end = l_end #User defined 
    min_L_step = l_step #User defined 
    min_L_range = np.arange(min_L_start,min_L_end,min_L_step)
    min_C_start = c_start #*******SHOULD BE USER DEFINED*********************
    min_C_end = c_end #*******SHOULD BE USER DEFINED*********************
    min_C_step = c_step #User defined
    min_C_range = np.arange(min_C_start,min_C_end,min_C_step)
    SR_start = sr_start #*******SHOULD BE USER DEFINED*********************
    SR_end = sr_end #*******SHOULD BE USER DEFINED*********************
    SR_step = sr_step #User defined
    SR_range = np.arange(SR_start,SR_end,SR_step)


    f_time = np.zeros((min_C_range.size,SR_range.size)) #time taken for FOCAL to run
    f_clus = np.zeros((min_C_range.size,SR_range.size)) # number of clusters found by FOCAL
    f_acc = np.zeros((min_C_range.size,SR_range.size))
    f_tpr = np.zeros((min_C_range.size,SR_range.size))
    f_spc = np.zeros((min_C_range.size,SR_range.size))
    opt_min_L = np.zeros((min_C_range.size,SR_range.size))

    plt_num = 1;
    for idx, i in enumerate(SR_range):
        #print(i)
        for idy, j in enumerate(min_C_range):

            num_clus_minL = (-1)*np.ones((min_L_range.size,1))
            minL_used = (-1)*np.ones((min_L_range.size,1))
            
            for idz, k in enumerate(min_L_range):
                min_L_found = False
                try:
                    # find optimal minL through uniform localizations
                    
                    _,min_L_clus = FOCAL_r_para(min_L_rnd,k,j,i)
                    minL_used[idz] = k
                    num_clus_minL[idz] = min_L_clus
                    plt_num = plt_num + 1
                    # breaks out of outer loop once min L is found
                    #bool_save = False #TEMPORARY
                    if min_L_clus == 0:
                        
                        min_L_found = True
                        min_l_grid.append(k)

                        bool_save = True #*****THIS SHOULD BE USER DEFINED FOR SCANNING PARAMETERS
#                        fig_minL = plt.figure()
#                        ax_minL = fig_minL.add_subplot(111)
                        minL_used = minL_used[minL_used!=-1]
                        num_clus_minL = num_clus_minL[num_clus_minL !=-1]
#                        ax_minL.plot(minL_used,num_clus_minL)
#                        ax_minL.set_title("Choosing Optmized minL: {}, for minC {} and SR {} ".format(k,j,i))
#                        ax_minL.set_xlabel("minL")
#                        ax_minL.set_ylabel("Nb of Clusters")
                        
#                        plt.savefig('Choose_minL{}minC{}SRP{}.png'.format(k,j,i))
#                        plt.close(fig_minL)
                        #not saving FOCAL table for SR = 40
                        f_time[idy,idx], f_clus[idy,idx] = FOCAL_r_para(loc_table,k,j,i,bool_save,None)
                        #f_time[idy,idx], f_clus[idy,idx], f_acc[idy,idx], f_tpr[idy,idx], f_spc[idy,idx] = FOCAL_r_para(loc_table,k,j,i,bool_save,loc_tab.gnd_truth)
                            
                            
                        

                        opt_min_L[idy,idx] = k
                        #f_time[idy,idx], f_clus[idy,idx], f_acc[idy,idx], f_tpr[idy,idx], f_spc[idy,idx] = FOCAL_r_para(loc_tab.coords,k,j,i,bool_save,loc_tab.gnd_tr)
                        
                        #f_time[idy,idx], f_clus[idy,idx], f_acc[idy,idx], f_tpr[idy,idx], f_spc[idy,idx] = FOCAL_r_para(loc_table,k,j,i,bool_save,None)
                        print("FOCAL success at minL = {}. minC = {}. SR = {}".format(
                            k, j, i))
                        bool_save = False
                        break
                except Exception as e:
                    print(e)
                    print("FOCAL error at minL = {}. minC = {}. SR = {}".format(
                        k, j, i))
                    continue
                finally:
                    if min_L_found:
                        break
        unique_lst = []
        for elem in min_l_grid:
            if elem not in unique_lst:
                unique_lst.append(elem)
        min_l_lst.append(unique_lst) #min_l is a list of list 

        min_l_grid = []
    

    
        # gui: SHOULD KEEP THIS VARIABLES IN MEMORY IN CASE USER WANTS TO SEE PLOTS
        np.save('min_L_range.npy',min_L_range)
        np.save('min_C_range.npy',min_C_range)
        np.save('SR_range.npy',SR_range)
    #    np.save('de_range.npy',de_range)
    #    np.save('dm_range.npy',dm_range)
        np.save('f_clus.npy',f_clus) #f_clus is an array with len(minC) x len(SRgrid) elements
#        plt.plot(f_clus)
    #    np.save('d_clus.npy',d_clus)
        np.save('f_time.npy',f_time)
        if 'f_acc' in locals():
            np.save('f_acc.npy',f_acc)
            np.save('f_tpr.npy',f_tpr)
            np.save('f_spc.npy',f_spc)
        np.save('opt_min_L.npy',opt_min_L) #opt_min_L is an array with len(minC) x len(SRgrid) elements
    
#        plt.plot(opt_min_L)
    print("Min_C plots loading...")
    num_minL = 0
    for lst in min_l_lst:
        for minL in lst:
            num_minL += 1
    progress_bar_max = num_minL*len(min_c_lst)
#    progress_max = num_minL*len(min_c_lst)
    #plot_c_data = plot_min_C(loc_table, min_l_lst, grid_lst, min_c_lst)
    return loc_table, min_l_lst, grid_lst, min_c_lst, progress_bar_max 
    



def plot_min_C(l_table, minL_lst, grid_size_lst, c_range_lst):
    data_lst = []
    min_c_lst = []
    num_cluster_lst = []
    dic = {}
    index = 0 
    for grid_size in grid_size_lst:
        dic[grid_size] = minL_lst[index]
        index += 1

    for grid_size, min_l_lst in dic.items():
        for min_l in min_l_lst:
            for min_c in c_range_lst:
                _, num_cluster = FOCAL_r_para(l_table, min_l, min_c, grid_size)
                min_c_lst.append(min_c)
                num_cluster_lst.append(num_cluster)
            data_lst.append((grid_size, min_l, min_c_lst, num_cluster_lst))
            min_c_lst, num_cluster_lst = [], []
    return data_lst 


def scan_DB(eps_start, eps_end, eps_step, pts_start, pts_end, pts_step, file_path):

    # this is for using data already given/generated
#    # retrieve data from 3d voronoi paper
#    loc_table = np.genfromtxt("H2B.csv",delimiter=',')
#    # delete 3 column this is data from 3D voronoi paper
#    loc_table = loc_table[:,3:]
#    
#    # get the localization table : format x | y | z |

#    root = tk.Tk()
#    root.withdraw()
    print("Get Localization Table")
    #file_path = filedialog.askopenfilename()
    loc_table = np.loadtxt(file_path)


    x_max = np.max(loc_table[:,0])
    y_max = np.max(loc_table[:,1])
    if loc_table.shape[1] == 2:
        start_main = time.time()
        minPts_rnd = np.random.uniform((0,0),(x_max,y_max),(
        loc_table.shape[0],2))
    if loc_table.shape[1] == 3:
        z_max = np.max(loc_table[:,2])
        start_main = time.time()
        # generate randomly spaced localizations for minL optimization - CHANGE TO MAKE DYES BLINK
        #min_L_rnd = rand_blinking_dyes(x_max,y_max,z_max,len(loc_tab.coords))
        minPts_rnd = np.random.uniform((0,0,0),(x_max,y_max,z_max),(
        loc_table.shape[0],3))
    
    # loop over DBSCAN parameters
    
    d_eps_start = eps_start #********************start at localization precision maybe??*********************************
    d_eps_end = eps_end
    d_eps_step = eps_step
    d_eps_range = np.arange(d_eps_start,d_eps_end,d_eps_step)
    
    minPts_start = pts_start #*******SHOULD BE USER DEFINED*********************
    minPts_end = pts_end #*******SHOULD BE USER DEFINED*********************
    minPts_step = pts_step
    minPts_range = np.arange(minPts_start,minPts_end,minPts_step)
    
    success_lst_d_eps = []
    success_lst_min_pts = []
    
    f_time = np.zeros((d_eps_range.size,1)) #time taken for FOCAL to run
    f_clus = np.zeros((d_eps_range.size,1)) # number of clusters found by FOCAL
    f_acc = np.zeros((d_eps_range.size,1))
    f_tpr = np.zeros((d_eps_range.size,1))
    f_spc = np.zeros((d_eps_range.size,1))
    opt_minPts = np.zeros((d_eps_range.size,1))
    print(np.shape(f_clus))
    
    
    for idx, i in enumerate(d_eps_range):
        #print(i)
        # minPts is determined by taking the same amount of random
            # localizations and fitting until no clusters are found
#                f_time[idx,idy,idz], f_clus[idx,idy,idz], f_vol[idx,
#                         idy,idz] = FOCAL_r_para(coords,i,j,k)
            # this loops finds minPts
        print("Looking for minPts....")
        num_clus_minPts = (-1)*np.ones((minPts_range.size,1))
        minPts_used = (-1)*np.ones((minPts_range.size,1))
        for idy, j in enumerate(minPts_range):
        
            minPts_found = False
            try:
                _, minPts_clus = DBSCAN_para(minPts_rnd,i,j)
                minPts_used[idy] = j
                num_clus_minPts[idy] = minPts_clus
                # breaks out of outer loop once min L is found
                if minPts_clus == 0:
                    minPts_found = True
                    bool_save = True #*****THIS SHOULD BE USER DEFINED FOR SCANNING PARAMETERS
                    
                    success_lst_d_eps.append(i)
                    success_lst_min_pts.append(j)
#                    fig_minPts = plt.figure()
#                    ax_minPts = fig_minPts.add_subplot(111)
                    minPts_used = minPts_used[minPts_used !=-1]
                    num_clus_minPts = num_clus_minPts[num_clus_minPts !=-1]
#                    ax_minPts.plot(minPts_used,num_clus_minPts)
#                    ax_minPts.set_title("Choosing Optimized minPts: {}, for d_eps {}".format(j,i))
#                    ax_minPts.set_xlabel("minPts")
#                    ax_minPts.set_ylabel("Nb of Clusters")
#                    plt.savefig('Choose_minPts{}d_eps{}.png'.format(j,i))
#                    plt.close(fig_minPts)
                    
                    f_time[idx], f_clus[idx] = DBSCAN_para(loc_table,i,j,bool_save, None)
                    print("number of clusters found:")
                    print(f_clus[idx])
                    opt_minPts[idx] = j 
                    print("DBSCAN success at minPts = {}. d_eps = {}".format(j,i))
                    bool_save = False
                    break
                    
            
            except Exception as e:
                print(e)
                print("DBSCAN error at minPts = {}. d_esp = {}.".format(
                    j, i))
                continue
            finally:
                if minPts_found:
                    break
        





    end_main = time.time() - start_main
    print("--- {}s ---".format(end_main))

    # gui: SHOULD KEEP THIS VARIABLES IN MEMORY IN CASE USER WANTS TO SEE PLOTS
    np.save('minPts_range.npy',minPts_range)
    np.save('d_eps_range.npy',d_eps_range)
    

    np.save('f_clus.npy',f_clus) #f_clus is an array with len(minC) x len(SRgrid) elements
#    plt.plot(f_clus)
#    np.save('d_clus.npy',d_clus)
    np.save('f_time.npy',f_time)
    if 'f_acc' in locals():
        np.save('f_acc.npy',f_acc)
        np.save('f_tpr.npy',f_tpr)
        np.save('f_spc.npy',f_spc)
    np.save('opt_minPts.npy',opt_minPts) #opt_min_L is an array with len(minC) x len(SRgrid) elements

#    plt.plot(opt_minPts)
    
    print("D_eps plot loading")
    d_eps_lst, num_clus_lst = plot_d_eps(loc_table, success_lst_d_eps, success_lst_min_pts)
    return d_eps_lst, num_clus_lst

    

def plot_d_eps(coords, eps_lst, min_pts_lst):
    num_cluster_lst = []
    for index in range(len(eps_lst)):
        _, num_cluster = DBSCAN_para(coords, eps_lst[index], min_pts_lst[index])
        num_cluster_lst.append(num_cluster)
    return eps_lst, num_cluster_lst
        
    
    


    
        
#if __name__ == "__main__":
    
    #scan_DB(10, 125, 5, 1, 251, 1)
    #scan_FOCAL(1,351,1,1,51,1,40,45,5) #For focal we need "FOCAL_Python_Code" file
