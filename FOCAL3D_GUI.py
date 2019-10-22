# -*- coding: utf-8 -*-
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from pyqtgraph.Qt import QtGui

from PyQt5.QtWidgets import (QAction, QMenu, QWidget, QSpinBox, QTreeWidgetItem, 
QHBoxLayout, QPlainTextEdit, QLineEdit, QLabel, QGridLayout, QPushButton, 
QDialog, QComboBox, QTreeWidget, QVBoxLayout, QProgressBar)
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import numpy as np
import time
import main_FOCAL_3D as f3d
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from Scan_Parameters_FOCAL_DBSCAN import *
from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem
import pyqtgraph
import os

  
 
class MyApp(QtGui.QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
         
        self.focal_window = None
        self.dbscan_window = None
        self.tutorial_window = None
        self.about_window = None
        self.point_size_window = None
        self.color_window = None
        self.plot_window = None 
         
        self.m1_lst = []
        self.clus_flag = False
        self.is_3d = None
        self.is_new_file = None #True if new file and False if load File 
        
        self.focal_param = []
        self.dbscan_param = []
                 
        self.initUI()
 
    def initUI(self):
 
 
 
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setGeometry(50,50,1600,800)
        self.setWindowTitle('FOCAL3D')
         
        layout_widget = QtGui.QWidget(self)
        self.setCentralWidget(layout_widget)
         
         
        self.grid = QtGui.QGridLayout()
        layout_widget.setLayout(self.grid)
 
        self.graph_fr = gl.GLViewWidget()
        self.grid.addWidget(self.graph_fr,0,0)
         
        self.create_actions()
        self.create_shortcut()
        self.create_menu()
        self.connect_actions()
 
 
    def create_actions(self):
        #File Menu Actions
         
        self.new_action = QAction('&New', self)
        self.load_action = QAction("&Load", self)
        self.load_plot_action = QAction("&Load Plot", self)
         
        #Cluster Menu Actions
         
        self.focal_action = QAction('&FOCAL', self)
        self.dbscan_action = QAction('&DBSCAN', self)
         
        #Image Tool Menu Actions 
         
        self.on_action = QAction("ON", self)
        self.off_action_1 = QAction("OFF", self)
        self.color_action = QAction("Color", self)
         
        self.all_action = QAction("All", self)
        self.off_action_2 = QAction("Off", self)
        self.cluster_action = QAction("Clustered", self)
        self.point_size_action = QAction("Point Size", self)
         
        #Help Menu Actions
         
        self.tutorial_action = QAction("Tutorial", self)
        self.about_action = QAction("About", self)
         
    def create_shortcut(self):
        self.new_action.setShortcut('Ctrl+N')
        self.load_action.setShortcut("Ctrl+L")
        self.load_plot_action.setShortcut("Ctrl+P")
        self.focal_action.setShortcut('Ctrl+R')
        self.dbscan_action.setShortcut('Ctrl+D')
         
    def create_menu(self):
        menu_bar = self.menuBar()
         
        #Adding Menu's to the menu bar
        file_menu = menu_bar.addMenu("&File")
        cluster_menu = menu_bar.addMenu('&Cluster')
        image_menu = menu_bar.addMenu("&Image Tools")
        help_menu = menu_bar.addMenu("&Help")
 
 
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.load_action)
        file_menu.addAction(self.load_plot_action)
         
         
        cluster_menu.addAction(self.focal_action)
        cluster_menu.addAction(self.dbscan_action)
 
         
       
        cluster_boundary = QMenu("Cluster Boundaries", self)
        localization = QMenu("Localization", self)
         
        image_menu.addMenu(cluster_boundary)
        image_menu.addMenu(localization)
         
         
        cluster_boundary.addAction(self.on_action)
        cluster_boundary.addAction(self.off_action_1)
        cluster_boundary.addAction(self.color_action)
         
        localization.addAction(self.all_action)
        localization.addAction(self.cluster_action)
        localization.addAction(self.off_action_2)
        localization.addAction(self.point_size_action)
         
         
        help_menu.addAction(self.tutorial_action)  
        help_menu.addAction(self.about_action)
         
    def connect_actions(self):
        #File Menu Actions 
        self.new_action.triggered.connect(self.new)
        self.load_action.triggered.connect(self.load)
        self.load_plot_action.triggered.connect(self.load_plot)
         
        #Cluster Menu Actions 
        self.focal_action.triggered.connect(self.launch_focal)
        self.dbscan_action.triggered.connect(self.launch_dbscan)
        self.color_action.triggered.connect(self.launch_color)
         
        #Image tool Menu Actions 
        self.on_action.triggered.connect(self.clus_on)
        self.off_action_1.triggered.connect(self.clus_off)
         
        self.all_action.triggered.connect(self.local_all)
        self.cluster_action.triggered.connect(self.local_clus)
        self.off_action_2.triggered.connect(self.local_off)
        self.point_size_action.triggered.connect(self.launch_point_size)
         
        #Help Menu Actions 
        self.tutorial_action.triggered.connect(self.launch_tutorial)
        self.about_action.triggered.connect(self.launch_about)
         
    def is_3D(self):
        '''Determines whether the localization table has 2D data or 
        3D data. If it has 3D data it sets the attribute self.is_3D to
        True, otherwise it sets it to False. This should work on both new and
        loaded files
        '''
        if self.is_new_file and self.loc_table.shape[1] == 2:
            self.is_3d = False
        if self.is_new_file and self.loc_table.shape[1] == 3:
#            print("This statement was called")
            self.is_3d = True
        if not self.is_new_file and self.loc_table.shape[1] == 3:
            self.is_3d = False
        if not self.is_new_file and self.loc_table.shape[1] == 4:
            self.is_3d = True
     
    def check_format(self):
        '''If a user has a localization table it checks whether the format of 
        it is correct or not. This should work for both new and loaded files.
        Returns True if the file has the correct format otherwise it opens 
        a dialog box telling the user that the format is incorrect. 
        '''
        if np.isnan(np.sum(self.loc_table)) == True:
            return False
        if self.is_new_file and self.loc_table.shape[1] == 2:
            if self.loc_table.dtype == "float64":
                return True
        if self.is_new_file and self.loc_table.shape[1] == 3:
            if self.loc_table.dtype == "float64":
                return True
        if not self.is_new_file and self.loc_table.shape[1] == 3:
            correct_format = self.check_load()
            if correct_format == True:
                return True
        if not self.is_new_file and self.loc_table.shape[1] == 4:
            correct_format = self.check_load()
            if correct_format == True:
                return True
        return False
         
 
 
    def check_load(self):
        '''Checks if the format of a loaded file is correct
        '''
        if self.loc_table.dtype == "float64":
            cluster_num_arr = self.loc_table[:,-1]
            if np.min(cluster_num_arr) >= -1:
                int_arr = cluster_num_arr.astype("int")
                if np.all((cluster_num_arr - int_arr) == 0):
                    return True
        return False
 
    def switch(self):
        '''Switches the display from either 2D to 3D or 3D to 2D. 
        '''
        if self.is_3d == False:
            try:
                self.grid.removeWidget(self.graph_fr)
                self.graph_fr.setParent(None)
            except:
                pass
            self.plt_widget = PlotWidget()
            self.grid.addWidget(self.plt_widget, 0,0)
            self.plt_widget.getPlotItem().hideAxis("bottom")
            self.plt_widget.getPlotItem().hideAxis("left")
        if self.is_3d == True:
            try:
                self.grid.removeWidget(self.plt_widget)
                self.plt_widget.setParent(None)
            except:
                pass
            self.graph_fr = gl.GLViewWidget()
            self.grid.addWidget(self.graph_fr,0,0)
             
    def show_points(self):
        '''Adds the the points to the display in the main window for both 
        2D and 3D
        '''
        self.plt_table = self.loc_table.copy() #Makes a deep copy of self.loc_table
        if self.is_3d == False:
            self.points = ScatterPlotItem(pos = self.plt_table)
            self.points.setBrush(color = (255,255,255))
            self.points.setPen(color = (0,0,0))
            self.plt_widget.addItem(self.points)
#            time.sleep(20)
        if self.is_3d == True:
            x_coords = self.plt_table[:,0]
            y_coords = self.plt_table[:,1]
            z_coords = self.plt_table[:,2]
             
            self.mid_x = np.median(x_coords)
            self.mid_y = np.median(y_coords)
            self.mid_z = np.median(z_coords)
            mid_coords = np.ptp(self.plt_table,axis=0) + np.amin(self.plt_table,axis=0)  
            self.plt_table -= mid_coords 
            self.sp = gl.GLScatterPlotItem(pos=self.plt_table,color=(1.0,1.0,1.0,1.0) 
                    ,size=1.0,pxMode=False)
            self.sp.translate(self.mid_x,self.mid_y,self.mid_z)
#            self.sp.rotate(90, 1, 0, 0)
#            self.sp.rotate(90, 0, 1, 0)
#            self.sp.translate(-mid_x, -mid_)
            self.graph_fr.addItem(self.sp) 
            self.graph_fr.update() 
             
    def new(self):
         
        self.is_new_file = True
#        try:
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open File',
                '/',"Localization File (*.txt)")
        if fname == ('',''):
            return 
#        except:
#            print("cat")
#            return None 
#        try:
        self.loc_table = np.genfromtxt(fname[0],delimiter=' ')            
#        except ValueError: #Example is if file has different number of columns 
#            QtGui.QMessageBox.about(self, "Error", "File format is incorrect")
#            return
        self.file_path = fname[0]
        self.dir_path = self.file_path[:self.file_path.rfind("/")] #Use this when determing where to save
        correct_format = self.check_format()
        self.is_3D()
        if correct_format == True:
            self.switch()
            self.show_points()
            QtGui.QMessageBox.about(self, "File", "Localization Table Loaded")
        else:
            QtGui.QMessageBox.about(self, "Error", "File format is incorrect")
         
 
    def load(self):
        self.is_new_file = False
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open File',
                    '/',"Localization File (*.txt)")
        if fname == ('',''):
            return 
        self.loc_table = np.genfromtxt(fname[0],delimiter=' ', skip_header = 1)
        self.check_format()
        self.is_3D()
        self.switch()
        if self.is_3d == True: #Haven't accounted for the fact that DBSCAN starts at 0
            self.coords = self.loc_table[:,0:4]
            self.points = self.loc_table[:, 0:3]
 
            self.plt_table = self.points.copy()                     #Makes a deep copy of self.coords
            
            x_coords = self.points[:,0]
            y_coords = self.points[:,1]
            z_coords = self.points[:,2]
             
            self.mid_x = np.median(x_coords)
            self.mid_y = np.median(y_coords)
            self.mid_z = np.median(z_coords)
            
            mid_coords = np.ptp(self.plt_table,axis=0) + np.amin(self.plt_table,axis=0)  
            self.plt_table -= mid_coords # centers the plot
            self.sp = gl.GLScatterPlotItem(pos=self.plt_table,color=(1.0,1.0,1.0,1.0) 
                    ,size=1.0,pxMode=False) #This code works well with an actual mouse not a laptop trackpad
            self.sp.translate(self.mid_x, self.mid_y, self.mid_z)
            self.graph_fr.addItem(self.sp)
            arr = self.loc_table[:,3]
            clustered = arr[arr != -1]
            min_cluster = np.amin(clustered)
            num_cluster = np.amax(self.loc_table[:,3])
            self.cluster_lst = []
            self.mesh_lst = []
            for num in range(int(min_cluster), int(num_cluster) + 1):
 
                i_clus = self.plt_table[self.loc_table[:,3]==num]
                self.cluster_lst.append(i_clus)
                try:
                    i_hull = ConvexHull(i_clus)
                except:
                    continue
                i_simplices = i_clus[i_hull.simplices]
 
                m1 = gl.GLMeshItem(vertexes=i_simplices,shader="edgeHilight")
                m1.setColor((1.0,0.416,0,1.0))
                m1.translate(self.mid_x,self.mid_y,self.mid_z)
                self.mesh_lst.append(m1)
                self.graph_fr.addItem(m1)
#            self.local_clus()
#            self.local_all()
        if self.is_3d == False:
            self.coord_2d = self.loc_table[:,0:2]
#            self.plt_table_2d = self.coord_2d.copy()
            self.points = ScatterPlotItem(pos = self.coord_2d)
            self.points.setBrush(color = (255,255,255))
            self.points.setPen(color = (0,0,0))
            self.plt_widget.addItem(self.points)
             
            cluster_arr = self.loc_table[:,0:3][self.loc_table[:,2] != -1]
            start = np.amin(cluster_arr[:,2])
            nb_clus = np.amax(cluster_arr[:,2])
            self.clus_lst = []
            self.line_lst = [] #Remembers the boundaries 
            for i in range(int(start),int(nb_clus)+1):
                i_clus = self.loc_table[:,0:2][self.loc_table[:,2] == i]
                self.clus_lst.append(i_clus)
                try:
                    i_hull = ConvexHull(i_clus)
                except:
                    continue
                i_vertices = i_hull.vertices
                x_coord = []
                y_coord = []
                for index in i_vertices:
                    x_coord.append(i_clus[index][0])
                    y_coord.append(i_clus[index][1])
                x_coord.append(x_coord[0])
                y_coord.append(y_coord[0])
                self.lines = PlotCurveItem()
                self.lines.setData(x = np.array(x_coord), y = np.array(y_coord))
                self.lines.setPen(width = 5, color = (255,140,0))
                self.lines.setBrush(color = (255,140,0))
                self.line_lst.append(self.lines)
                self.plt_widget.addItem(self.lines)
                     
    def get_header(self, file_name):
        '''Gets the header of the csv file not including the header of the first column. 
        np.genfromtext is getting rid of the header and need that information for 
        loading the plots for focal
        '''
        file = open(file_name)
        header1 = file.readline()
        header2 = file.readline() 
        return [header1.strip(), header2.strip()]
    
    
    
    
    def load_plot(self):
        '''Should be able to detect whether a correct csv file has been opened. 
        
        '''
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Load plot',
                    '/',"Plot File (*.csv)")
        if fname == ('',''):
            return 
        header = self.get_header(fname[0])
        self.plot_data = np.loadtxt(fname[0], delimiter=',')
        if header[0] == "# DBSCAN PLOT DATA":
            epsilon = self.plot_data[:,1]
            num_cluster = self.plot_data[:,2]
            self.plot_window = Plot_d_eps([list(epsilon), list(num_cluster)])
        if header[0] == "# FOCAL PLOT DATA": #Should check whether a valid num_clus csv file 
            min_c_range = self.plot_data[:,0]
            lst = header[1].split(",")
            grid_size_lst = []
            for grid_size in lst[1:]:
                grid_size_lst.append(grid_size.strip())
            grid_size_range = np.array(grid_size_lst)
            num_clus_arr = self.plot_data[:,1:]
            self.plot_window = PlotMinC([min_c_range, grid_size_range, num_clus_arr])

        
        
 
    def launch_point_size(self):
        self.point_size_window = PointSizeWindow() 
         
    def launch_about(self):
        self.about_window = AboutWindow() 
 
    def launch_tutorial(self):
        self.tutorial_window = TutorialWindow() 
         
    def launch_focal(self):
        self.focal_window = FocalWindow()
         
    def launch_dbscan(self):
        self.dbscan_window = DbscanWindow() 
        
    def launch_color(self):
        self.color_window = ColorWindow()
 
         
    def clear_plot(self):
#        print("Within clear plot")
        if self.is_3d == True:
#            print("Within if statement")
            self.graph_fr.items = []
            self.graph_fr.update()
        if self.is_3d == False:
            self.plt_widget.getPlotItem().clear()
#            for item in self.plt_widget.getPlotItem().items:
#                print(item)
#                self.plt_widget.removeItem(item)
#            self.plt_widget.getPlotItem().items = []
#            self.plt_widget.update()
#            print("Within the clear plot")
#            time.sleep(20)
#            print("Leaving the clear plot")
            
            
            
    def clus_on(self):
        if self.is_3d == True:
            for item in self.graph_fr.items:
                if type(item) == gl.items.GLMeshItem.GLMeshItem:
                    item.setVisible(True)
        if self.is_3d == False:
            for item in self.plt_widget.getPlotItem().items:
                if type(item) == pyqtgraph.graphicsItems.PlotCurveItem.PlotCurveItem:
                    item.setPen(width = 5, color = (255,140,0))
                    item.setBrush(color = (255,140,0))
             
        #self.plt_ch(self.arg1, self.arg2, self.arg3, self.arg4)
#        for elem in self.m1_lst:
#            if elem not in self.gr.items:
#                self.gr.addItem(elem)
 
    def clus_off(self):
        if self.is_3d == True:            
            for item in self.graph_fr.items:
                if type(item) == gl.items.GLMeshItem.GLMeshItem:
                    item.hide()
        if self.is_3d == False:
            for item in self.plt_widget.getPlotItem().items:
                if type(item) == pyqtgraph.graphicsItems.PlotCurveItem.PlotCurveItem:
                    item.setPen((0,0,0))
                    item.setBrush((0,0,0))
 
                 
    def local_all(self):
        if self.is_3d == True:
            for item in self.graph_fr.items:
                if type(item) == gl.items.GLScatterPlotItem.GLScatterPlotItem:
                    item.setVisible(True)
        if self.is_3d == False:
            self.plt_widget.getPlotItem().items[0].setBrush(color = (255,255,255))
            self.plt_widget.getPlotItem().items[0].setPen(color = (0,0,0))
             
 
    def local_clus(self):
        if self.is_3d == True:
            if self.clus_flag == False:
                index = 1
                array = self.cluster_lst[0]
                while index < len(self.cluster_lst):
                    array = np.concatenate((array, self.cluster_lst[index]))
                    index += 1
                 
                self.clus_Item = gl.GLScatterPlotItem(pos=array,color=(1.0,1.0,1.0,1.0) 
                       ,size=1.0,pxMode=False)
                self.clus_Item.translate(self.mid_x, self.mid_y, self.mid_z)
                self.graph_fr.addItem(self.clus_Item)
                for item in self.graph_fr.items:
                    if (item != self.clus_Item) and (type(item) == gl.items.GLScatterPlotItem.GLScatterPlotItem):
                        item.hide()
                self.clus_flag = True
            else:
                for item in self.graph_fr.items:
                    if (item != self.clus_Item) and (type(item) == gl.items.GLScatterPlotItem.GLScatterPlotItem):
                        item.hide()
                    if item == self.clus_Item:
                        item.setVisible(True)
        if self.is_3d == False:
            try:
                memory = []
                cluster_array = np.vstack(tuple(self.clus_lst))
                self.cluster_point = ScatterPlotItem(pos = cluster_array)
                if self.cluster_point not in self.plt_widget.getPlotItem().items:
                     
                    self.cluster_point.setBrush(color = (255,255,255))
                    self.cluster_point.setPen(color = (0,0,0))
                    self.plt_widget.getPlotItem().items[0].setBrush(None)
                    self.plt_widget.getPlotItem().items[0].setPen(None)
                    memory.append(self.plt_widget.getPlotItem().items[0])
                    self.plt_widget.getPlotItem().clear()
                    self.plt_widget.addItem(memory[0])
                    self.plt_widget.addItem(self.cluster_point)
                    for line in self.line_lst:
                        self.plt_widget.addItem(line)
                    
                    
#                    boundary_lst = []
#                    for item in self.plt_widget.getPlotItem().items:
#                        if type(item) == pyqtgraph.graphicsItems.PlotCurveItem.PlotCurveItem:
#                            item.setPen(width = 5, color = (255,140,0))
#                            item.setBrush(color = (255,140,0))
#                            self.plt_widget.removeItem(item)
#                            boundary_lst.append(item)
#                    self.plt_widget.addItem(cluster_point)
#                    for item in boundary_lst:
#                        self.plt_widget.addItem(item)
                            
                else:
                    index = self.plt_widget.getPlotItem().items.index(self.cluster_point)
                    scatter_plot_item = self.plt_widget.getPlotItem().items[index]
                    if self.plt_widget.getPlotItem().items[0].opts["pen"] == None:
                        scatter_plot_item.setBrush(color = (255,255,255))
                        scatter_plot_item.setPen(color = (0,0,0))
                    else:
                        self.plt_widget.getPlotItem().items[0].setBrush(None)
                        self.plt_widget.getPlotItem().items[0].setPen(None)
                        scatter_plot_item.setBrush(color = (255,255,255))
                        scatter_plot_item.setPen(color = (0,0,0))

                         
            except AttributeError:
                print("need to run clustering algorithm before clicking submenu item")
                 
 
     
    def local_off(self):
        if self.is_3d == True:
            for item in self.graph_fr.items:
                if type(item) == gl.items.GLScatterPlotItem.GLScatterPlotItem:
                    item.hide()
        if self.is_3d == False:
            for item in self.plt_widget.getPlotItem().items:
                if type(item) == pyqtgraph.graphicsItems.ScatterPlotItem.ScatterPlotItem:
                    item.setBrush(None)
                    item.setPen(None)
 
 
             
    def run_FOCAL(self, grid_size, min_l, min_c, path = ""):
        try:
            self.loc_table
        except AttributeError:
            QtGui.QMessageBox.about(self, "Error", "No File Opened")
        else:
#            bool_3D = self.bool3d
#            self.focal_flag = True
#            if self.dbscan_flag == True:
#                print("Within the statement")
            self.clear_plot()
            self.show_points()
#            self.dbscan_flag = False
            res_sr = grid_size
            min_L = min_l
            min_C = min_c
            start = time.time()
            self.f3d_clus = f3d.densityMap(self.plt_table,res_sr,min_L,self.is_3d,min_C)
            self.f3d_clus.gen_pix_sr()
            self.f3d_clus.gen_den_map()
            self.f3d_clus.gen_core_pix()
            self.f3d_clus.process_clusters()
            self.f3d_clus.process_clusters1()
            self.f3d_clus.process_clusters3()
            self.f3d_clus.label_loc()
            if path != "":
                np.savetxt(str(path) +'/focalLocTabSR{}minL{}minC{}.txt'.format(res_sr,min_L,min_C),
                           self.f3d_clus.loc_table, fmt='%f',
                           header = "Min L: {}, Min C: {}, Grid Size: {}".format(min_L, min_C, res_sr))
            else:
                np.savetxt(str(self.dir_path) + '/focalLocTabSR{}minL{}minC{}.txt'.format(res_sr,min_L,min_C),
                           self.f3d_clus.loc_table, fmt='%f',
                           header = "Min L: {}, Min C: {}, Grid Size: {}".format(min_L, min_C, res_sr))
            end_time = time.time() - start
            QtGui.QMessageBox.about(self, "FOCAL", "FOCAL finished in {}s".format(end_time))
            if self.is_3d:
                self.plt_ch(self.graph_fr,self.f3d_clus.clus_nb,
                        self.f3d_clus.loc_table[:,-1],1)
            else:
                self.plt_ch_2d(self.plt_widget, self.f3d_clus.clus_nb,
                               self.f3d_clus.loc_table[:,-1], 1)
         
 
             
    def run_DBSCAN(self, epsilon, min_pts, path = ""):
        try:
            self.loc_table
        except AttributeError:
            QtGui.QMessageBox.about(self, "Error", "No File Opened")
        else:
#            self.dbscan_flag = True
#            if self.focal_flag == True:
            self.clear_plot()
            self.show_points()
#            self.focal_flag = False
            d_eps = epsilon
            d_minpts = min_pts
            start_time = time.time()
            self.db = DBSCAN(eps=d_eps,min_samples=d_minpts).fit(self.plt_table)
            table = self.loc_table.copy()
            cluster_table = np.column_stack((table,self.db.labels_))
            if path != "":
                np.savetxt(str(path) + '/DBSCANLocTabEps{}MinPts{}.txt'.format(d_eps, d_minpts),
                           cluster_table, fmt = "%f", 
                           header = "Epsilon: {}, Min Pts: {}".format(d_eps, d_minpts))
            else:
                np.savetxt(str(self.dir_path)+ '/DBSCANLocTabEps{}MinPts{}.txt'.format(d_eps, d_minpts),
                           cluster_table, fmt='%f',
                           header = "Epsilon: {}, Min Pts: {}".format(d_eps, d_minpts))
        end_time = time.time() - start_time
        QtGui.QMessageBox.about(self, "DBSCAN", "DBSCAN finished in {}s".format(end_time))
        if self.is_3d:
            self.plt_ch(self.graph_fr,len(set(self.db.labels_)),self.db.labels_,0)
        else:
            self.plt_ch_2d(self.plt_widget,len(set(self.db.labels_)),self.db.labels_,0)
 
 
    def plt_ch(self,gr_ch,nb_clus,clus_lb,start):
        self.cluster_lst = []
        self.mesh_lst = []
        for i in range(start,nb_clus+start):
            i_clus = self.plt_table[clus_lb==i]
            self.cluster_lst.append(i_clus)
            try:
                i_hull = ConvexHull(i_clus)
            except:
                continue
            i_simplices = i_clus[i_hull.simplices]
            m1 = gl.GLMeshItem(vertexes=i_simplices,shader="edgeHilight")
            m1.translate(self.mid_x, self.mid_y, self.mid_z)
            m1.setColor((1.0,0.416,0,1.0))
            self.mesh_lst.append(m1)
            gr_ch.addItem(m1)
#        self.local_clus()
#        self.local_all()
#        self.clus_on()
 
         
    def plt_ch_2d(self, plt_wig, nb_clus, clus_label, start):
        
        self.clus_lst = [] 
        self.line_lst = []
        for i in range(start,nb_clus+start):
            i_clus = self.loc_table[clus_label == i]
            self.clus_lst.append(i_clus)
            try:
                i_hull = ConvexHull(i_clus)
            except:
                continue
            i_vertices = i_hull.vertices
            x_coord = []
            y_coord = []
            for index in i_vertices:
                x_coord.append(i_clus[index][0])
                y_coord.append(i_clus[index][1])
             
            x_coord.append(x_coord[0])
            y_coord.append(y_coord[0])
            self.lines = PlotCurveItem()
            self.lines.setData(x = np.array(x_coord), y = np.array(y_coord))
            self.lines.setPen(width = 5, color = (255,140,0))
            self.lines.setBrush(color = (255,140,0))
            self.line_lst.append(self.lines)
            self.plt_widget.addItem(self.lines)
 
class FocalWindow(QWidget):
     
    def __init__(self):
        super().__init__()
        self.label1 = None #Need this otherwise the text that the user inputs into the line won't be remembered
        self.label2 = None
        self.label3 = None
        self.focal_optimal = None
#        self.prev_param = [] 
        self.initUI()
 
         
    def initUI(self):
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setGeometry(60,60,339,147)
        self.setWindowTitle("FOCAL Parameters")
         
        grid = QGridLayout()
        self.setLayout(grid)
         
         
        label1 = QLabel("Grid Size", self)
        label2 = QLabel("Min L", self)
        label3 = QLabel("Min C", self)
        label4 = QLabel("Directory", self)
         
        global my_app
        self.comboBox = QComboBox(self)
        try:
            self.data = [my_app.dir_path, "Browse"] #drop down menu 
        except AttributeError:
            QtGui.QMessageBox.about(self, "Error", "No file has been opened")
            return
             
        self.comboBox.addItems(self.data)
         
        lineWidget = QLineEdit(self)
        lineWidget2 = QLineEdit(self)
        lineWidget3 = QLineEdit(self)
         
         
        self.label1 = lineWidget
        self.label2 = lineWidget2
        self.label3 = lineWidget3
        
        if my_app.focal_param != []:
            self.label1.setText(my_app.focal_param[0])
            self.label2.setText(my_app.focal_param[1])
            self.label3.setText(my_app.focal_param[2])
        grid.addWidget(label1, 0,0)
        grid.addWidget(lineWidget, 0,1)
        grid.addWidget(label2, 1,0)
        grid.addWidget(lineWidget2, 1,1)
        grid.addWidget(label3, 2,0)
        grid.addWidget(lineWidget3, 2,1)
        grid.addWidget(self.comboBox, 3,1)
         
        runButton = QPushButton("Run", self)
        scanButton = QPushButton("Scan", self)
         
        self.grid_info_btn = QPushButton(self)
        self.minl_info_btn = QPushButton(self)
        self.minc_info_btn = QPushButton(self)
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        
        
        self.grid_info_btn.setIcon(QtGui.QIcon(scriptDir + os.path.sep + "buttonicon.png"))
        self.grid_info_btn.setStyleSheet("background-color: rgb(255,255,255)")
         
        self.minl_info_btn.setIcon(QtGui.QIcon(scriptDir + os.path.sep + "buttonicon.png"))
        self.minl_info_btn.setStyleSheet("background-color: rgb(255,255,255)")
         
        self.minc_info_btn.setIcon(QtGui.QIcon(scriptDir + os.path.sep + "buttonicon.png"))
        self.minc_info_btn.setStyleSheet("background-color: rgb(255,255,255)")
 
         
        grid.addWidget(label4, 3, 0)
        grid.addWidget(runButton, 4, 1)
        grid.addWidget(scanButton, 4, 0)
         
        grid.addWidget(self.grid_info_btn, 0, 2)
        grid.addWidget(self.minl_info_btn, 1, 2)
        grid.addWidget(self.minc_info_btn, 2, 2)
         
        runButton.clicked.connect(self.runFOCALBUTTON)
        scanButton.clicked.connect(self.scanParam)
 
        self.comboBox.activated[str].connect(self.location)
        
        self.grid_info_btn.clicked.connect(self.create_box)
        self.minl_info_btn.clicked.connect(self.create_box)
        self.minc_info_btn.clicked.connect(self.create_box)
        
        self.show()
         
    def runFOCALBUTTON(self):
        global my_app
        try:
            if int(self.label1.text()) > 0 and int(self.label2.text()) > 0 and int(self.label3.text()) > 0:
                my_app.focal_param = []
                my_app.focal_param.append(self.label1.text())
                my_app.focal_param.append(self.label2.text())
                my_app.focal_param.append(self.label3.text())
                my_app.run_FOCAL(int(self.label1.text()), int(self.label2.text()), 
                                 int(self.label3.text()), self.comboBox.currentText())
            else: #Negative integers won't be caught by exception
                QtGui.QMessageBox.about(self, "Error", "Please enter positive integers for your parameters")
        except ValueError:
            QtGui.QMessageBox.about(self, "Error", "Please enter positive integers for your parameters")
             
    def scanParam(self):
        self.focal_optimal = FocalOptimal()
         
         
    def location(self, text):
        if text == "Browse":
            file = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
            #print(file)
            if file == "":
                self.comboBox.setCurrentText(self.data[0])
                return
            if file not in self.data:
                self.data.insert(0, file)
                self.comboBox.clear()
                self.comboBox.addItems(self.data)
                self.comboBox.setCurrentText(file)
            if file in self.data:
                self.comboBox.setCurrentText(file)
                
    def create_box(self):
        button = self.sender()
        if button == self.grid_info_btn:
            txt = ("The grid size depends on the size of your clusters.\n Picking too small of a cluster is memory (RAM) intensive"
            " and FOCAL3D takes more time.\n Piciking too large of a cluster size will result in poorer performance (missed clusters, or "
            "combining nearby clusters into a single cluster.\n\n The optimal grid size tends to be less than or about the cluster size"
            " which can be heuristically determined, for example with a Ripley's K function analysis.\n\n For more details, please refer to the tutorial"
            "under the \" Help \" menu option."
                   )
            self.text_box = TextBox(txt)
        if button == self.minl_info_btn:
            txt = ("A minL is assigned to every grid size and minC that was scanned. "
                   "A csv file is saved which has the min_L corresponding to given "
                   "grid size and min_c values.\n\n For more details on how minL is chosen "
                   "please refer to FOCAL3D: A 3D Clustering Algorithm for single-molecule localization microscopy.\nFor more details on the GUI, please refer to the tutorial"
                    "under the \" Help \" menu option."
                   )
            self.text_box = TextBox(txt)
        if button == self.minc_info_btn:
            txt = ("In order to pick a good min_C we need to scan for parameters. "
                   "After scanning the parameters a series of plots will be generated " 
                   "for different grid sizes.\n\n Look at the num_cluster vs min_C " 
                   "plot that corresponds to your grid size. The min_C value that you "
                   "pick should be close to the 'elbow' of the plot. The 'elbow' of the plot is "
                   "simply when the first point where the curve becomes flat. \n\n"
                   "For more details, please refer to the tutorial"
                    "under the \" Help \" menu option."
                   )
            self.text_box = TextBox(txt)
             
class DbscanWindow(QWidget):
     
    def __init__(self):
        super().__init__()
        self.epsilon_line = None #Need to set these values before calling .initUI otherwise we'll get error
        self.min_pts_line = None
        self.dbscan_optimal = None
        self.initUI()
 
         
    def initUI(self):
        
        
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        
        self.setGeometry(60, 60, 349, 151)
        self.setWindowTitle("DBSCAN Parameters")
         
        grid = QGridLayout()
        self.setLayout(grid)
         
        epsilon_label = QLabel("eps", self)
        min_pts_label = QLabel("Minimum Points", self)
        dir_label = QLabel("Directory", self)
         
        global my_app
        #self.save_loc_label = QLabel(my_app.dir_path, self)
         
        epsilon_line = QLineEdit(self)
        min_pts_line = QLineEdit(self)
         
        self.epsilon_line = epsilon_line
        self.min_pts_line = min_pts_line
        
        if my_app.dbscan_param != []:
            self.epsilon_line.setText(my_app.dbscan_param[0])
            self.min_pts_line.setText(my_app.dbscan_param[1])
         
        self.comboBox = QComboBox(self)
        try:
            self.data = [my_app.dir_path, "Browse"]
        except AttributeError:
            QtGui.QMessageBox.about(self, "Error", "No file has been opened")
            return
        self.comboBox.addItems(self.data)
         
        run_button = QPushButton("Run", self)
        scan_button = QPushButton("Scan", self)
         
        self.eps_info_btn = QPushButton(self)
        self.min_pts_info_btn = QPushButton(self)
        
        self.eps_info_btn.setIcon(QtGui.QIcon(scriptDir + os.path.sep +  "buttonicon.png"))
        self.eps_info_btn.setStyleSheet("background-color: rgb(255,255,255)")
         
 
        self.min_pts_info_btn.setIcon(QtGui.QIcon(scriptDir + os.path.sep +  "buttonicon.png"))
        self.min_pts_info_btn.setStyleSheet("background-color: rgb(255,255,255)")
         
         
        grid.addWidget(epsilon_label, 0,0)
        grid.addWidget(epsilon_line, 0,1)
        grid.addWidget(min_pts_label, 1,0)
        grid.addWidget(min_pts_line, 1,1)
        grid.addWidget(dir_label, 2,0)
        grid.addWidget(self.comboBox, 2,1)
        grid.addWidget(self.eps_info_btn, 0,2)
        grid.addWidget(self.min_pts_info_btn, 1,2)
        #grid.addWidget(save_button, 2,0)
        #grid.addWidget(self.save_loc_label, 2,1)
        grid.addWidget(run_button, 3,1)
        grid.addWidget(scan_button, 3,0)
         
        run_button.clicked.connect(self.runDBSCANBUTTON)
        scan_button.clicked.connect(self.scanParam)
        self.comboBox.activated[str].connect(self.location)
        #save_button.clicked.connect(self.saveLocation)
        
        self.eps_info_btn.clicked.connect(self.create_box)
        self.min_pts_info_btn.clicked.connect(self.create_box)
        self.show()
         
    def runDBSCANBUTTON(self):
        global my_app
        try:
            if int(self.epsilon_line.text()) > 0 and int(self.min_pts_line.text()) > 0:
                my_app.dbscan_param = []
                my_app.dbscan_param.append(self.epsilon_line.text())
                my_app.dbscan_param.append(self.min_pts_line.text())
                my_app.run_DBSCAN(int(self.epsilon_line.text()), int(self.min_pts_line.text()), self.comboBox.currentText())
            else: #Negative integers won't be caught by exception
                QtGui.QMessageBox.about(self, "Error", "Please enter positive integers for your parameters")
        except ValueError:
            QtGui.QMessageBox.about(self, "Error", "Please enter positive integers for your parameters")
             
    def scanParam(self):
        self.dbscan_optimal = DbscanOptimal()

        
    def create_box(self): 
        button = self.sender()
        if button == self.eps_info_btn:
            self.textbox = TextBox("The value for eps is usually correlated with the size of the cluster, particularly for spherical-like clusters"
            "The parameter can sometimes be chosen from a scan of parameters. \n\n For more details, please refer to the tutorial"
            "under the \" Help \" menu option.")
        if button == self.min_pts_info_btn:
            self.textbox = TextBox("A minPts value is assigned to every eps scanned. A csv file is saved which has the minPts for a particular eps.\n\n."
            "For more details, please refer to the tutorial"
            "under the \" Help \" menu option.")
         
#        file = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
#        self.save_loc_label.setText(file)
         
    def location(self, text):
        if text == "Browse":
            #print(self.comboBox.currentData())
            file = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
            if file == "":
                self.comboBox.setCurrentText(self.data[0])
                return
            if file not in self.data:
                self.data.insert(0, file)
    #            txt = self.data.pop()
    #            self.data.append(file)
    #            self.data.append(txt)
                self.comboBox.clear()
                self.comboBox.addItems(self.data)
                #self.comboBox.removeItem(0)
    #            self.comboBox.addItem(file)
    #            self.comboBox.addItem(text)
                self.comboBox.setCurrentText(file)
            if file in self.data:
                self.comboBox.setCurrentText(file)
#        print("The current text is {}".format(self.comboBox.currentText()))
     
class DbscanOptimal(QWidget):
     
    def __init__(self):
        super().__init__()
        self.eps_s = None
        self.eps_e = None
        self.eps_stp = None
        self.min_pt_s = None
        self.min_pt_e = None
        self.min_pt_stp = None
        self.plot_d_eps = None #If not initialized in constructor window will not appear 
        self.dbscan_pbar = None
        self.initUI()
     
    def initUI(self):
        
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setGeometry(60, 60, 350, 350)
        self.setWindowTitle(scriptDir + os.path.sep + "DBSCAN Optimal")
         
        grid = QGridLayout()
        self.setLayout(grid)
         
        label1 = QLabel("eps_start", self)
        label2 = QLabel("eps_end", self)
        label3 = QLabel("eps_step", self)
         
        label4 = QLabel("min point start", self)
        label5 = QLabel("min point end", self)
        label6 = QLabel("min point step", self)
         
        lineWidget1 = QLineEdit(self)
        lineWidget2 = QLineEdit(self)
        lineWidget3 = QLineEdit(self)
         
        lineWidget4 = QLineEdit(self)
        lineWidget5 = QLineEdit(self)
        lineWidget6 = QLineEdit(self)
         
        self.eps_s = lineWidget1
        self.eps_e = lineWidget2
        self.eps_stp = lineWidget3
        self.min_pt_s = lineWidget4
        self.min_pt_e = lineWidget5
        self.min_pt_stp = lineWidget6
         
        run_button = QPushButton("Run", self)
         
        grid.addWidget(label1, 0, 0)
        grid.addWidget(label2, 1, 0)
        grid.addWidget(label3, 2, 0)
        grid.addWidget(label4, 3, 0)
        grid.addWidget(label5, 4, 0)
        grid.addWidget(label6, 5, 0)
         
        grid.addWidget(lineWidget1, 0, 1)
        grid.addWidget(lineWidget2, 1, 1)
        grid.addWidget(lineWidget3, 2, 1)
        grid.addWidget(lineWidget4, 3, 1)
        grid.addWidget(lineWidget5, 4, 1)
        grid.addWidget(lineWidget6, 5, 1)
         
        grid.addWidget(run_button, 6, 1)
         
        run_button.clicked.connect(self.scan)
         
        self.show()
         
    def scan(self):
        try:
            global my_app 
            file_pth = my_app.file_path
            parm1 = int(self.eps_s.text())
            parm2 = int(self.eps_e.text())
            parm3 = int(self.eps_stp.text())
            parm4 = int(self.min_pt_s.text())
            parm5 = int(self.min_pt_e.text())
            parm6 = int(self.min_pt_stp.text())
            min_parm = min(parm1, parm2, parm3, parm4, parm5, parm6)
            if min_parm > 0:
                #ep_lst, clus_lst = scan_DB(parm1, parm2, parm3, parm4, parm5, parm6, file_pth)
                dbscan_args = [parm1, parm2, parm3, parm4, parm5, parm6, file_pth]
                del self.dbscan_pbar
                self.dbscan_pbar = DbscanProgressBar(dbscan_args)
            else:
                QtGui.QMessageBox.about(self, "Error", "Please enter positive integers for your parameters")
        except ValueError:
            if file_pth == None:
                QtGui.QMessageBox.about(self, "Error", "No File Open")
            else:
                QtGui.QMessageBox.about(self, "Error", "Please enter positive integers for your parameters")
                

class DbscanWorker(QObject):
     
    progressChanged = pyqtSignal(int)
    progressChanged2 = pyqtSignal(int)
    progressbarmax1 = pyqtSignal(int) 
    progressbarmax2 = pyqtSignal(int)
    give_me_the_list = pyqtSignal(list)
     
     
    def __init__(self, args):
        super().__init__()
        self.eps_start = args[0]
        self.eps_end = args[1]
        self.eps_step = args[2]
        self.min_pts_start = args[3]
        self.min_pts_end = args[4]
        self.min_pts_step = args[5]
        self.file_path = args[6]
      
    def doWork(self):
        loc_table = np.loadtxt(self.file_path)

        path_name = os.path.split(self.file_path)[0]
    
        x_max = np.max(loc_table[:,0])
        y_max = np.max(loc_table[:,1])
        if loc_table.shape[1] == 2:
#            start_main = time.time()
            minPts_rnd = np.random.uniform((0,0),(x_max,y_max),(
            loc_table.shape[0],2))
        if loc_table.shape[1] == 3:
            z_max = np.max(loc_table[:,2])
#            start_main = time.time()
            # generate randomly spaced localizations for minL optimization - CHANGE TO MAKE DYES BLINK
            #min_L_rnd = rand_blinking_dyes(x_max,y_max,z_max,len(loc_tab.coords))
            minPts_rnd = np.random.uniform((0,0,0),(x_max,y_max,z_max),(
            loc_table.shape[0],3))
        
        # loop over DBSCAN parameters
        
        d_eps_start = self.eps_start #********************start at localization precision maybe??*********************************
        d_eps_end = self.eps_end
        d_eps_step = self.eps_step
        d_eps_range = np.arange(d_eps_start,d_eps_end,d_eps_step)
        
        minPts_start = self.min_pts_start #*******SHOULD BE USER DEFINED*********************
        minPts_end = self.min_pts_end #*******SHOULD BE USER DEFINED*********************
        minPts_step = self.min_pts_step
        minPts_range = np.arange(minPts_start,minPts_end,minPts_step)
        

        
        success_lst_d_eps = []
        success_lst_min_pts = []
        
        f_time = np.zeros((d_eps_range.size,1)) #time taken for FOCAL to run
        f_clus = np.zeros((d_eps_range.size,1)) # number of clusters found by FOCAL
        f_acc = np.zeros((d_eps_range.size,1))
        f_tpr = np.zeros((d_eps_range.size,1))
        f_spc = np.zeros((d_eps_range.size,1))
        opt_minPts = np.zeros((d_eps_range.size,1))
        d_eps = np.zeros((d_eps_range.size, 1))
#        print(np.shape(f_clus))
        progress = 0
        progress_bar_1_max = int(d_eps_range.size)*int(minPts_range.size)
        self.progressbarmax1.emit(progress_bar_1_max)
        for idx, i in enumerate(d_eps_range): #I think the break statement is the reason why its not running all the way
#            print("This is the value of i:{}".format(i))
            # minPts is determined by taking the same amount of random
                # localizations and fitting until no clusters are found
    #                f_time[idx,idy,idz], f_clus[idx,idy,idz], f_vol[idx,
    #                         idy,idz] = FOCAL_r_para(coords,i,j,k)
                # this loops finds minPts
#            print("Looking for minPts....")
            num_clus_minPts = (-1)*np.ones((minPts_range.size,1))
            minPts_used = (-1)*np.ones((minPts_range.size,1))
#            print("This is the value of i:{} after stuff".format(i))
            min_pts_left = minPts_range.size
            for idy, j in enumerate(minPts_range):
                
#                print("This is the value of j:{}".format(j))
                progress += 1 
                min_pts_left = min_pts_left - 1
                #progress += 1 
                self.progressChanged.emit(progress)

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
#                        fig_minPts = plt.figure()
#                        ax_minPts = fig_minPts.add_subplot(111)
                        minPts_used = minPts_used[minPts_used !=-1]
                        num_clus_minPts = num_clus_minPts[num_clus_minPts !=-1]

                        
#                        ax_minPts.plot(minPts_used,num_clus_minPts)
#                        ax_minPts.set_title("Choosing Optimized minPts: {}, for d_eps {}".format(j,i))
#                        ax_minPts.set_xlabel("minPts")
#                        ax_minPts.set_ylabel("Nb of Clusters")
#                        plt.savefig('Choose_minPts{}d_eps{}.png'.format(j,i))
#                        plt.close(fig_minPts)
                        opt_minPts[idx] = j                         
                        f_time[idx], f_clus[idx] = DBSCAN_para(loc_table,i,j,bool_save, None)
                        d_eps[idx] = i
#                        print("number of clusters found:")
#                        print(f_clus[idx])
#                        print("The optimal minpts:{}".format(j))
#                        print("The number of points in the cluster is {}".format(f_clus[idx]))
                        
#                        print("DBSCAN success at minPts = {}. d_eps = {}".format(j,i))
                        bool_save = False
                        progress = progress + min_pts_left
                        self.progressChanged.emit(progress)
                        break
                        
                
                except Exception as e:
                    print(e)
                    print("DBSCAN error at minPts = {}. d_esp = {}.".format(
                        j, i))
                    continue
                finally:
                    if minPts_found:
                        break
        
        np.savetxt(str(path_name) + "DBSCAN_eps({},{},{})_minpts({},{},{}).csv".format(d_eps_start, d_eps_end, d_eps_step, minPts_start, minPts_end, minPts_step), 
                   np.hstack((opt_minPts, d_eps, f_clus)), "%d,%d,%d", 
                   header = "DBSCAN PLOT DATA\nMin Pts, eps, Num Cluster")
    
    
    
#        end_main = time.time() - start_main
#        print("--- {}s ---".format(end_main))
    
        # gui: SHOULD KEEP THIS VARIABLES IN MEMORY IN CASE USER WANTS TO SEE PLOTS
        np.save(str(path_name) + 'minPts_range.npy',minPts_range)
        np.save(str(path_name) + 'd_eps_range.npy',d_eps_range)
        
    
        np.save(str(path_name) +'f_clus.npy',f_clus) #f_clus is an array with len(minC) x len(SRgrid) elements
#        plt.plot(f_clus)
    #    np.save('d_clus.npy',d_clus)
        np.save(str(path_name) +'f_time.npy',f_time)
        if 'f_acc' in locals():
            np.save('f_acc.npy',f_acc)
            np.save('f_tpr.npy',f_tpr)
            np.save('f_spc.npy',f_spc)
        np.save(str(path_name) +'opt_minPts.npy',opt_minPts) #opt_min_L is an array with len(minC) x len(SRgrid) elements
    
#        plt.plot(opt_minPts)
        
 #       print("D_eps plot loading")
        self.plot_d_eps(loc_table, success_lst_d_eps, success_lst_min_pts)
#        d_eps_lst, num_clus_lst = self.plot_d_eps(loc_table, success_lst_d_eps, success_lst_min_pts)
#        print(f_clus)
#        print(opt_minPts)

    

    def plot_d_eps(self, coords, eps_lst, min_pts_lst):
        num_cluster_lst = []
        self.progressbarmax2.emit(int(len(eps_lst)-1))
        progress = 0 
        for index in range(len(eps_lst)):
            progress += 1 
            self.progressChanged2.emit(progress)
            _, num_cluster = DBSCAN_para(coords, eps_lst[index], min_pts_lst[index])
            num_cluster_lst.append(num_cluster)
        info_lst = [eps_lst, num_cluster_lst]
        self.give_me_the_list.emit(info_lst)
        #return eps_lst, num_cluster_lst
    
class DbscanProgressBar(QDialog):#(QWidget):
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.plot = None
        self.initUI()
     
    def initUI(self):
        self.setGeometry(60, 60, 502, 150)
        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)
        self.progress_bar = QProgressBar(self)
        self.progress_bar2 = QProgressBar(self)
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.label1 = QLabel("DBSCAN SCAN PROGRESS", self)
        #self.label2 = QLabel("PLOTTING PROGRESS", self)
        self.vbox.addWidget(self.label1)
        self.vbox.addWidget(self.progress_bar)
        #self.vbox.addWidget(self.label2)
        #self.vbox.addWidget(self.progress_bar2)
        self.thread = QThread(self)
        self.worker = DbscanWorker(self.data)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.doWork)
        self.thread.start()
        
        self.worker.progressChanged.connect(self.progress_bar.setValue)
        #self.worker.progressChanged2.connect(self.progress_bar2.setValue)
        self.worker.give_me_the_list.connect(self.get_data)
        self.worker.progressbarmax1.connect(self.progress_bar.setMaximum)
        #self.worker.progressbarmax2.connect(self.progress_bar2.setMaximum)
        self.show()
         
    def get_data(self, lst):
        self.plot = Plot_d_eps(lst)
        self.thread.terminate() #Trying to solve QThread destroyed while still running error, quick bandaid fix
        #Note GUI crashes if you close the progress bar midway and then run it again

class Plot_d_eps(QDialog):
     
    def __init__(self, lst):#eps_lst, num_clus_lst):
        super().__init__()
         
 
        # a figure instance to plot on
 #       print("D_eps plot loaded")
         
        self.eps_lst = lst[0]  #eps_lst
        self.num_clus_lst = lst[1] #num_clus_lst 
         
        self.figure = plt.figure() 
        self.canvas = FigureCanvas(self.figure)
 
 
 
        # set the layout
        layout = QGridLayout()
        layout.addWidget(self.canvas, 0, 1)
        self.setLayout(layout)
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setWindowTitle("FOCAL3D")
        ax = self.figure.add_subplot(111)
        ax.set_title("Num Cluster vs eps")
        ax.set_xlabel("eps")
        ax.set_ylabel("Nb of Clusters")
        ax.plot(self.eps_lst, self.num_clus_lst)
        self.canvas.draw()
        self.show()
                 
                 
class FocalOptimal(QWidget):
     
    def __init__(self):
        super().__init__()
        self.l_start, self.l_end, self.l_step = None, None, None
        self.c_start, self.c_end, self.c_step = None, None, None
        self.sr_start, self.sr_end, self.sr_step = None, None, None
        self.plot = None
        self.progress_bar = None
        self.focal_pbar = None
        self.focal_pbar2 = None
        self.initUI()
     
    def initUI(self):
        
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setGeometry(60, 60, 350, 350)
        self.setWindowTitle("FOCAL Optimal")
         
        grid = QGridLayout()
        self.setLayout(grid)
         
        label1 = QLabel("min L start", self)
        label2 = QLabel("min L end", self)
        label3 = QLabel("min L step", self)
         
        label4 = QLabel("min C start", self)
        label5 = QLabel("min C end", self)
        label6 = QLabel("min C step", self)
         
        label7 = QLabel("Grid Size start", self)
        label8 = QLabel("Grid Size end", self)
        label9 = QLabel("Grid Size step", self)
         
        lineWidget1 = QLineEdit(self)
        lineWidget2 = QLineEdit(self)
        lineWidget3 = QLineEdit(self)
         
        lineWidget4 = QLineEdit(self)
        lineWidget5 = QLineEdit(self)
        lineWidget6 = QLineEdit(self)
         
        lineWidget7 = QLineEdit(self)
        lineWidget8 = QLineEdit(self)
        lineWidget9 = QLineEdit(self)
         
        self.l_start, self.l_end, self.l_step = lineWidget1, lineWidget2, lineWidget3
        self.c_start, self.c_end, self.c_step = lineWidget4, lineWidget5, lineWidget6
        self.sr_start, self.sr_end, self.sr_step = lineWidget7, lineWidget8, lineWidget9
 
 
        run_button = QPushButton("Run", self)
         
        grid.addWidget(label1, 0, 0)
        grid.addWidget(label2, 1, 0)
        grid.addWidget(label3, 2, 0)
        grid.addWidget(label4, 3, 0)
        grid.addWidget(label5, 4, 0)
        grid.addWidget(label6, 5, 0)
        grid.addWidget(label7, 6, 0)
        grid.addWidget(label8, 7, 0)
        grid.addWidget(label9, 8, 0)
         
         
        grid.addWidget(lineWidget1, 0, 1)
        grid.addWidget(lineWidget2, 1, 1)
        grid.addWidget(lineWidget3, 2, 1)
        grid.addWidget(lineWidget4, 3, 1)
        grid.addWidget(lineWidget5, 4, 1)
        grid.addWidget(lineWidget6, 5, 1)
        grid.addWidget(lineWidget7, 6, 1)
        grid.addWidget(lineWidget8, 7, 1)
        grid.addWidget(lineWidget9, 8, 1)
         
        grid.addWidget(run_button, 9, 1)
         
        run_button.clicked.connect(self.scan)
                 
        self.show()
         
    def scan(self):
        try:
            global my_app 
            file_pth = my_app.file_path
            parm1 = int(self.l_start.text())
            parm2 = int(self.l_end.text())
            parm3 = int(self.l_step.text())
            parm4 = int(self.c_start.text())
            parm5 = int(self.c_end.text())
            parm6 = int(self.c_step.text())
            parm7 = int(self.sr_start.text())
            parm8 = int(self.sr_end.text())
            parm9 = int(self.sr_step.text())
            min_parm = min(parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9)
            if min_parm > 0:
#                global count 
                scan_focal_args = [parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, file_pth] #We will run the plot_min_C method within the class and update it accordingly
                
#                if count%2 == 0:
#                    if count > 0:
#                        self.focal_pbar2.thread.terminate()
                self.focal_pbar = FocalProgressBar(scan_focal_args)
#                if count%2 == 1:
#                    self.focal_pbar.thread.terminate()
#                    self.focal_pbar2 = FocalProgressBar(scan_focal_args)
#                count += 1
            else:
                QtGui.QMessageBox.about(self, "Error", "Please enter positive integers for your parameters")
        except ValueError:
            if file_pth == None:
                QtGui.QMessageBox.about(self, "Error", "No File Open")
            else:
                QtGui.QMessageBox.about(self, "Error", "Please enter positive integers for your parameters")
 
 
class FocalWorker(QObject):
     
    progressChanged = pyqtSignal(int)
#    progressChanged2 = pyqtSignal(int)
#    progressbarmax2 = pyqtSignal(int)
    give_me_the_list = pyqtSignal(list)
     
     
    def __init__(self, args):
        super().__init__()
        self.l_start = args[0]
        self.l_end = args[1]
        self.l_step = args[2]
        self.c_start = args[3]
        self.c_end = args[4]
        self.c_step = args[5]
        self.sr_start = args[6] 
        self.sr_end = args[7]
        self.sr_step = args[8]
        self.file_pth = args[9]
        self.plot= None
 
     
    def doWork(self):
        data_lst = []
        loc_table = np.loadtxt(self.file_pth)
        #path_name = os.path.dirname(self.file_pth)
        path_name = os.path.split(self.file_pth)[0]
        x_max = np.max(loc_table[:,0])
        y_max = np.max(loc_table[:,1]) 
        if loc_table.shape[1] == 2:
            min_L_rnd = np.random.uniform((0,0),(x_max,y_max),(loc_table.shape[0],2))
     
        if loc_table.shape[1] == 3:
            z_max = np.max(loc_table[:,2])
            min_L_rnd = np.random.uniform((0,0,0),(x_max,y_max,z_max),(
            loc_table.shape[0],3))
         
#        min_l_lst = [] #A keeps track of the min_L that lead to zero clusters. 
#        min_l_grid = [] #Is a list that tracks the min L of a specific grid size and resets after each grid size. 
##        grid_lst = np.arange()
#        grid_lst = [size for size in range(self.sr_start, self.sr_end, self.sr_step)]
#        min_c_lst = [c for c in range(self.c_start, self.c_end, self.c_step)]
     
        min_L_range = np.arange(self.l_start,self.l_end,self.l_step)
        min_C_range = np.arange(self.c_start,self.c_end,self.c_step)
        SR_range = np.arange(self.sr_start,self.sr_end,self.sr_step)
        
        data_lst.append(min_C_range)  #Need this information when generating the plot 
        data_lst.append(SR_range)     #Need this information when generating the plot
        
#        min_l_array = np.zeros((len(min_c_lst), len(grid_lst)))
        min_l_arr = np.zeros((min_C_range.size, SR_range.size)) #This array keeps track of the min_L successes for each grid_size and min_C
        num_clus_arr = np.zeros((min_C_range.size, SR_range.size)) #This array keeps track of the num clus for each grid size and min_C 
#        plt_num = 1;
        progress = 0
        for idx, i in enumerate(SR_range):
            for idy, j in enumerate(min_C_range):
                progress += 1
                self.progressChanged.emit(progress)
                num_clus_minL = (-1)*np.ones((min_L_range.size,1))
                minL_used = (-1)*np.ones((min_L_range.size,1))
                 
                for idz, k in enumerate(min_L_range):
                    min_L_found = False
                    try:
                        # find optimal minL through uniform localizations
                         
                        _,min_L_clus = FOCAL_r_para(min_L_rnd,k,j,i)
                        minL_used[idz] = k
                        num_clus_minL[idz] = min_L_clus
#                        plt_num = plt_num + 1
                        # breaks out of outer loop once min L is found
                        #bool_save = False #TEMPORARY
                        if min_L_clus == 0:
                             
                            min_L_found = True
#                            min_l_grid.append(k)
 
#                            fig_minL = plt.figure()
#                            ax_minL = fig_minL.add_subplot(111)
                            minL_used = minL_used[minL_used!=-1]
                            num_clus_minL = num_clus_minL[num_clus_minL !=-1]
                            min_l_arr[idy, idx] = k #This is the min_l that was successful. We are putting it in the array
                            
                            _, num_cluster = FOCAL_r_para(loc_table, k, j, i)
                            
                            num_clus_arr[idy, idx] = num_cluster
                            

                            
#                            ax_minL.plot(minL_used,num_clus_minL)
#                            ax_minL.set_title("Choosing Optmized minL: {}, for minC {} and SR {} ".format(k,j,i))
#                            ax_minL.set_xlabel("minL")
#                            ax_minL.set_ylabel("Nb of Clusters")
#                             
#                            plt.savefig('Choose_minL{}minC{}SRP{}.png'.format(k,j,i))
#                             
#                            plt.close(fig_minL)
                            break
                    except Exception as e:
                        print(e)
                        print("FOCAL error at minL = {}. minC = {}. SR = {}".format(
                            k, j, i))
                        continue
                    finally:
                        if min_L_found:
                            break
        data_lst.append(num_clus_arr)
        min_C_range_col = min_C_range.reshape((min_C_range.size, 1))
        title = "min_c_range,"
        for grid_size in SR_range:
            title = title + str(grid_size) + ","
        title = title[:-1] #Extra comma at the end 
        #os.chdir(str(path_name))
        format_str = (str("%d,"*int(SR_range.size + 1)))[:-1]
        np.savetxt("FOCAL_MINL_min_l({},{},{})_min_c({},{},{})_grid_size({},{},{}).csv".format(self.l_start,
                   self.l_end, self.l_step, self.c_start, self.c_end, self.c_step, 
                   self.sr_start, self.sr_end, self.sr_step),
                   np.hstack((min_C_range_col,min_l_arr)), format_str, 
                   header = "FOCAL MIN L DATA\n{}".format(title))
        np.savetxt("FOCAL_NUM_CLUS_min_l({},{},{})_min_c({},{},{})_grid_size({},{},{}).csv".format(self.l_start,
                   self.l_end, self.l_step, self.c_start, self.c_end, self.c_step, 
                   self.sr_start, self.sr_end, self.sr_step), 
                   np.hstack((min_C_range_col,num_clus_arr)), format_str, 
                   header = "FOCAL PLOT DATA\n{}".format(title))
        self.give_me_the_list.emit(data_lst)
        # gui: SHOULD KEEP THIS VARIABLES IN MEMORY IN CASE USER WANTS TO SEE PLOTS
        '''
        np.save('min_C_range.npy',min_C_range)
        np.save('Grid_size_range.npy',SR_range)
        
    
        np.save('FOCAL_num_clus.npy',num_clus_arr) #f_clus is an array with len(minC) x len(SRgrid) elements
#        plt.plot(f_clus)
    #    np.save('d_clus.npy',d_clus)
        #np.save('FOCAL_f_time.npy',f_time)
        
        np.save('opt_minL.npy',opt_minL) #opt_min_L is an array with len(minC) x len(SRgrid) elements
        '''

                  
class FocalProgressBar(QWidget):
     
    def __init__(self, data):
        super().__init__()
        self.data = data
        c_start = data[3]
        c_end = data[4]
        c_step = data[5]
        sr_start = data[6] 
        sr_end = data[7]
        sr_step = data[8]
        self.max = ((sr_end-sr_start)//(sr_step))*((c_end-c_start)//c_step)
        self.plot = None
         
        self.initUI()
     
    def initUI(self):
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setWindowTitle("FOCAL3D")
        self.setGeometry(60, 60, 502, 90)
        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(self.max)
#        self.progress_bar2 = QProgressBar(self)
        self.label1 = QLabel("FOCAL SCAN PROGRESS", self)
#        self.label2 = QLabel("PLOTTING PROGRESS", self)
        self.vbox.addWidget(self.label1)
        self.vbox.addWidget(self.progress_bar)
#        self.vbox.addWidget(self.label2)
#        self.vbox.addWidget(self.progress_bar2)
        self.thread = QThread(self)
        self.worker = FocalWorker(self.data)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.doWork)
        self.thread.start()
        self.worker.progressChanged.connect(self.progress_bar.setValue)
#        self.worker.progressChanged2.connect(self.progress_bar2.setValue)
        self.worker.give_me_the_list.connect(self.get_data)
#        self.worker.progressbarmax2.connect(self.progress_bar2.setMaximum)
        self.show()
         
    def get_data(self, lst):
        self.plot = PlotMinC(lst)
        self.thread.terminate()
        
#    def closeEvent(self, event):
        #print("I am a cat")
#        self.thread.wait()
#        self.thread.exit()

         
 
         
class PlotMinC(QDialog):
    def __init__(self, data_lst):
        super().__init__()
        self.count = 0
        # a figure instance to plot on
#        print("Min_C plots loaded")
        self.data_lst = data_lst 
        self.col = 0
          
        self.figure = plt.figure() 
 
        self.canvas = FigureCanvas(self.figure)
 
 
        self.next_btn = QPushButton("Next")
        self.back_btn = QPushButton('Back')
         
        self.next_btn.clicked.connect(self.next_button)
        self.back_btn.clicked.connect(self.back_button)
 
        # set the layout
        layout = QGridLayout()
        #layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 0, 1)
        layout.addWidget(self.back_btn, 1, 0)
        layout.addWidget(self.next_btn, 1, 2)
        self.setWindowTitle("FOCAL3D")
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setLayout(layout)
        self.plot()
        self.show()
         
    def next_button(self):
        if 0 <= self.col and self.col < self.data_lst[2].shape[1] - 1:
            self.col += 1
            self.plot()
     
    def back_button(self):
        if 0 < self.col and self.col < self.data_lst[2].shape[1]:
            self.col -= 1
            self.plot() 
         
     
    def plot(self):
        self.figure.clear()       
        ax = self.figure.add_subplot(111)
        num_clus_arr = self.data_lst[2]
        num_array = num_clus_arr[:,self.col]
        min_c_arr = self.data_lst[0]
        grid_size_arr = self.data_lst[1]
        
#        grid_size = self.data_lst[self.index][0]
#        min_L = self.data_lst[self.index][1]
        ax.set_title("Num Cluster vs Min C, for Grid Size:{} ".format(grid_size_arr[self.col]))
        ax.set_xlabel("minC")
        ax.set_ylabel("Nb of Clusters")
#        min_c = self.data_lst[self.index][2]
#        print(self.data_lst)
#        print(self.data_lst[self.index])
#        print(min_c)
#        num_cluster = (self.data_lst[self.index][3])[min_c, grid_size]
        ax.plot(min_c_arr, num_array)
        self.canvas.draw()
         
class TutorialWindow(QWidget):
     
    def __init__(self):
        super().__init__()
        self.initUI()
         
    def initUI(self):
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setGeometry(50,50,550,500)
        self.setWindowTitle('Tutorial')
        self.vbox = QHBoxLayout()
         
        self.root = QTreeWidget(self)
        self.root.setHeaderHidden(True)
         
        self.start = QTreeWidgetItem(self.root, ["Getting Started"])
        self.open = QTreeWidgetItem(self.start, ["Opening a New File"])
        self.load = QTreeWidgetItem(self.start, ["Loading a File"])
        self.load_plots = QTreeWidgetItem(self.start, ["Loading Plots"])
         
         
         
        self.cluster = QTreeWidgetItem(self.root, ["Running Clustering Algorithm"])
         
        self.focal = QTreeWidgetItem(self.cluster, ["FOCAL"])
        #self.algorithm1 = QTreeWidgetItem(self.focal, ["Description of FOCAL"])
        self.scan1 = QTreeWidgetItem(self.focal, ["Scanning for parameters"])
        self.param1 = QTreeWidgetItem(self.focal, ["Picking the Parameters"])
        
        
        
        
        
        self.min_l = QTreeWidgetItem(self.param1, ["Min L"])
        self.min_c = QTreeWidgetItem(self.param1, ["Min C"])
        self.grid_size = QTreeWidgetItem(self.param1, ["Grid Size"])
 
         
        self.dbscan = QTreeWidgetItem(self.cluster, ["DBSCAN"])
        #self.algorithm2 = QTreeWidgetItem(self.dbscan, ["Description of DBSCAN"])
        self.scan2 = QTreeWidgetItem(self.dbscan, ["Scanning for parameter"])
        self.param2 = QTreeWidgetItem(self.dbscan, ["Picking the Parameters"])
        
        self.eps = QTreeWidgetItem(self.param2, ["eps"])
        self.minpts = QTreeWidgetItem(self.param2, ["Min Pts"])
         
        self.image_tools = QTreeWidgetItem(self.root, ["Image tools"])
         
        self.clus_bound = QTreeWidgetItem(self.image_tools, ["Cluster Boundaries"])
        self.localizations = QTreeWidgetItem(self.image_tools, ["Localizations"])
 
        self.txt1 = QPlainTextEdit(self)
        self.txt1.setReadOnly(True)
 
        self.vbox.addWidget(self.root)
        self.vbox.addWidget(self.txt1)
         
        self.txt1.setStyleSheet("background-color: #F2F2F2")
         
        #self.root.itemClicked(self.open, 0).connect(self.test)
        self.root.itemClicked.connect(self.set_text)
        self.setLayout(self.vbox)
 
        self.show()
         
    def set_text(self): 
        if self.root.currentItem() == self.open:
            text = (" The format of your file needs to be a text file containing"
                    " either two or three columns corresponding to the x,y, and z coordinates"
                    " (respectively) of each localization. \n\n"
                    " You can open your file by"
                    " going to the \"File\" menu and selecting  \"New\". After "
                    " selecting your file,the data will be displayed in the display window."
                    )
            self.txt1.setPlainText(text)
        if self.root.currentItem() == self.load:
            text = ("For a given FOCAL3D or DBSCAN run on data that which has been opened up"
                    " FOCAL saves a text file with data from the clustering algorithm as"
                    "and additional column.\n\n"
                    " If you click \"Load\" under the \"File\" menu you can visualize"
                    " the data at anytime by clicking the textfile that was saved"
                    " when you initally ran FOCAL or DBSCAN."
                    )
            self.txt1.setPlainText(text)
            
            
        if self.root.currentItem() == self.load_plots:
            text = ("After running a scan, FOCAL saves the data to generate the plots."
                    "FOCAL gives the option to load plots made from a previous scan under the \"File\Load Plot \" menu."
                    )
            self.txt1.setPlainText(text)
            
        if self.root.currentItem() == self.cluster:
            text = ("In this GUI, the user can run the FOCAL or DBSCAN clustering algorithms.\n\n"
                    "Click on the dropdown menu arrow to explore the tutorial for each algorithm."
                    )
            self.txt1.setPlainText(text)
        if self.root.currentItem() == self.clus_bound:
            text = ("The cluster menu has two options for the cluster boundaries. The user can choose"
                    "to show the boundaries of the cluster by choosing the option \"ON\" under the \"Image Tools/Cluster Boundaries\" "
                    "option. \n\n By default, after an algorithm is run, FOCAL will display the cluster boundaries in orange.\n\n"
                    "The user can also choose to turn off the boundaries after running a clustering algorithm by "
                    "choosing the option \"OFF\" under the \"Image Tools/Cluster Boundaries\". \n\n"
                    "The user can also change the color of the boundaries under the \"Image Tools/Cluster Boundaries\Color\" option.\n\n"
                    "For more details, please refer to the tutorial guide \"FOCAL3D: A 3D Clustering Package for Single-Molecule Localization Microscopy - "
                    "A Tutorial Guide to the Graphical User Interface\"."
                    )
            self.txt1.setPlainText(text)
            
            
        if self.root.currentItem() == self.focal:
            text = ("FOCAL is a density-based clustering algorithm which requires three parameters."
                    " These parameters are termed minL, minC and the grid size. \n\n"
                    " The parameter minL is a measure of the minimum number of localization that need to be in"
                    " a voxel for the voxel to be considered a core point of the cluster.\n\n"
                    " minC is a threshold on the minimum number of connected voxels  to be considered a cluster.\n\n"
                    " The grid size determines the side length of each voxel.\n\n"
                    " Use the dropdown arrow to learn how to use the FOCAL GUI to scan for the correct parameters for FOCAL3D."
                    )
            self.txt1.setPlainText(text)
            
        if self.root.currentItem() == self.localizations:
            text = ("In the \"Image Tools/Localization\" menu the user has the option of which"
                    " points to display (All, Clustered, or none (Off)). The clustered points "
                    "correspond to any points which have been associated with a cluster."
                    "The user also has the option of choosing the point size under the \"Image Tools/Localization\" menu.\n\n"
                    "For more details, please refer to the tutorial guide \"FOCAL3D: A 3D Clustering Package for Single-Molecule Localization Microscopy - "
                    "A Tutorial Guide to the Graphical User Interface\"." 
                    )
            self.txt1.setPlainText(text)
            
        # if self.root.currentItem() == self.algorithm1:
        #     text = ("FOCAL is a density-based clustering algorithm which means"
        #             " that the clusters are considered to be regions in the data set"
        #             " with a high concentration of points. FOCAL takes in three"
        #             " parameters which are minL, minC and the grid size. The parameter"
        #             " minL is a measure of the minimum number of points that need to be in"
        #             " a cluster and minC is a threshold on the minimum number of voxels in a cluster."
        #             " Each voxel has a size length given by the grid size."
        #             )
        #     self.txt1.setPlainText(text)
            
            
        if self.root.currentItem() == self.dbscan :
            text = ("DBSCAN is a density-based clustering algorithm which requires two parameters."
                    " These parameters are termed \"minPts\", \"eps\". \n\n"
                    " The parameter \"minPts\" is a measure of the minimum number of localization that need to be within"
                    " a sphere with radius given by \"eps\".\n\n"
                    " The parameter \"eps\" then roughly corresponds to the radius of the clusters. \n\n"
                    " Use the dropdown arrow to learn how to use the FOCAL GUI to scan for the correct parameters for DBSCAN."
                    )
            self.txt1.setPlainText(text)
            
            
            # text = ("DBSCAN is a density-based clustering algorithm which means"
            #         " that the clusters are considered to be regions in the data set"
            #         " with a high concentration of points. DBSCAN takes in two"
            #         " parameters which are minPts and epsilon. We can think of"
            #         " minPts as the minimum number of points that need to be in"
            #         " a cluster and epsilon as the “size” of each cluster. It"
            #         " uses these two parameters to figure out what the clusters"
            #         " in the dataset should be."
            #         )
            # self.txt1.setPlainText(text)
        
        if self.root.currentItem() == self.min_l:
            text = ("After FOCAL finishes scanning for parameters, it will generate a comma-separated value (csv) file."
                    "Each entry in the csv file contains the value of minL corresponding to the given minC and grid size.\n\n"
                    "The entries are arranged as follows:\n\n"
                    "Each column corresponds to a fixed grid size. The first column corresponds to the first grid size used in the scan."
                    "The second column corresponds to the following grid size in the scan range (initial grid size + step size of scan), and so on.\n\n"
                    "Each row corresponds to a given minC. Each entry in the first row correspond to the value of minL at the first minC in the scan"
                    " for different grid sizes. The second column corresponds to values of minL found for the following minC in the scan (initial minC "
                    "+ step size in minC) at each grid size.\n\n"
                    "For more details, please refer to the tutorial guide \"FOCAL3D: A 3D Clustering Package for Single-Molecule Localization Microscopy - "
                    "A Tutorial Guide to the Graphical User Interface\"." 
                    )
            self.txt1.setPlainText(text)
        if self.root.currentItem() == self.min_c:
            text = ("Choosing minC is done from the series of plots generated by the scan.\n "
                    "After finding the first plot which displays the flattened region, a minC " 
                    "close to the \"elbow\" usually yields good clustering (The \"elbow\" of the plot is "
                   "simply the first point where the curve becomes flat). \n\n The data for the plots is "
                    "also outputted by the program if the user wants to generate the plots again.\n\n"
                    "For more details, please refer to the tutorial guide \"FOCAL3D: A 3D Clustering Package for Single-Molecule Localization Microscopy - "
                    "A Tutorial Guide to the Graphical User Interface\"." 
                    )
            self.txt1.setPlainText(text)
            
        if self.root.currentItem() == self.scan1:
            text = ("Under the \"Cluster\" menu item, choosing \"FOCAL\" will pop up a window where the user "
                    "can input parameters to run the algorithm with. We recommend using the \"Scan\" option"
                    "to let the program scan for parameters first.\n\n"
                    "The following guidelines can be used to select the range to have the program scan over: \n\n"
                    "minL: Localization density per voxel threshold. \n"
                    "Typical range: 1-1000 in steps of 1.\n\n"
                    "minC: Number of connected voxels threshold. \n"
                    "Typical range: 5-150 in steps of 5.\n\n"
                    "Grid size: side length of voxel to create grid.\n"
                    "Typical range: 25-65 in steps of 5 or 10.\n\n"
                    #"*The choice of grid size is often correlated with the effective size of the clusters."
                    #"If there is prior knowledge on roughly the size of the clusters, or can obtain an independent"
                    #" measure from a Ripley's K function analysis, this sometimes helps to narrow the range of grid sizes to scan over.\n\n"
                    "At the end of the scan, the GUI will generate a series of plots of the "
                    "number of detected clusters as a function of minC for each grid size."
                    "These guidelines are used as exploratory parameters to search for the characteristic flattening "
                    "of the number of detected clusters as a function of minC.\n\n"
                    "For guidelines on choosing the parameters from these curves, please refer to the \"Picking the Parameters\\FOCAL \" menu below."
                    
                    # "When choosing a range of parameters to scan over, minL should usually "
                    # "start at a value of 1. The upper limit of minL should be relatively high "
                    # "depending on the density of localizations in your data set. Choosing a value of 1000"
                    # " as an upper bound should be enough for most data sets. When choosing a range for minC,"
                    # " starting at minC = 5 will work depending on the grid size. When combined with a small grid size"
                    # " a low minC will result in over identifying clusters. When combined with a large grid size, a large minC"
                    # " will result in clusters overlapping. To start off a range of minC from 5 to 105 in steps of 4 should give"
                    # " an idea of the proper range of minC. For the grid size, steps of 5 or 10 in the grid size usually suffice."
                    )
            self.txt1.setPlainText(text)
            
        if self.root.currentItem() == self.scan2:
            
            text = ("Under the \"Cluster\" menu item, choosing \"DBSCAN\" will pop up a window where the user "
                    "can input parameters to run the algorithm with. We recommend using the \"Scan\" option"
                    "to let the program scan for parameters first.\n\n"
                    "The following guidelines can be used to select the range to have the program scan over: \n\n"
                    "minPts: Localization threshold within a radius \"eps\". \n"
                    "Typical range: 1-1000 in steps of 1.\n\n"
                    "eps: Radius of sphere defining the region to measure the number of localizations. \n"
                    "Typical range: 15-125 in steps of 10.\n\n"

                    "At the end of the scan, the GUI will generate a plot of the "
                    "number of detected clusters as a function of eps."
                    "These guidelines are used as exploratory parameters to search for the characteristic flattening "
                    "of the number of detected clusters as a function of eps.\n\n"
                    "For guidelines on choosing the parameters from these curves, please refer to the \"Picking the Parameters\\DBSCAN \" menu below.")
            self.txt1.setPlainText(text)
            
            # text = ("When choosing a range of parameters to scan over, minPts can start at "
            #         "a value of 1. The upper limit of minPts should be relatively high "
            #         "depending on the density of localizations in your data set. Choosing a value of 1000"
            #         " as an upper bound should be enough for most data sets. When choosing a range for epsilon,"
            #         " starting at 10 and increasing in steps of 5 until 80 should give a good indication of the correct range."
            #         " Keep in mind that the time DBSCAN takes increases quadratically with increasing epsilon."
            #         )
            
            
            
            
        if self.root.currentItem() == self.grid_size:
            text = ("The value for the grid size roughly corresponds to the first grid size where the curve in the plots genereated "
            "by the scan first displays a flattening.\n\n"
            "For more details, please refer to the tutorial guide \"FOCAL3D: A 3D Clustering Package for Single-Molecule Localization Microscopy - "
                    "A Tutorial Guide to the Graphical User Interface\"." 
            
                    )
            self.txt1.setPlainText(text)
        if self.root.currentItem() == self.eps:
            text = ("For some data sets, scanning for eps can give a region of "
            "less sensitivity similar to FOCAL3D. A value for epsilon based on this region "
            "can be used. Alternatively, if there is some prior estimate of the expected cluster "
            "sizes, this can be used as a starting point for epsilon.\n\n"
            "The data for the plot is also outputted by the program if the user wants to generate the plots again.\n\n"
            "For more details, please refer to the tutorial guide \"FOCAL3D: A 3D Clustering Package for Single-Molecule Localization Microscopy - "
                    "A Tutorial Guide to the Graphical User Interface\".")
            self.txt1.setPlainText(text)
        if self.root.currentItem() == self.minpts:
            text = ("After FOCAL finishes scanning for parameters for DBSCAN, it will generate a comma-separated value (csv) "
                    "file, The entries of the csv file are the minPts corresponding to each eps scanned.\n\n"
                    "For more details, please refer to the tutorial guide \"FOCAL3D: A 3D Clustering Package for Single-Molecule "
                    "Localization Microscopy - A Tutorial Guide to the Graphical User Interface \". "
                    )
            self.txt1.setPlainText(text)
            
             
             
class AboutWindow(QWidget):
     
    def __init__(self):
        super().__init__()
        self.initUI()
         
    def initUI(self):
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setGeometry(50,50,500,500)
        self.setWindowTitle('About')
        label = QLabel()
        label.setText('<a href="https://www.utm.utoronto.ca/milsteinlab/software/">Project Website</a>')
        label.setOpenExternalLinks(True)
         
 
        self.vbox = QVBoxLayout()
         
         
        self.txt1 = QPlainTextEdit(self)
        self.txt1.setReadOnly(True)
#        self.txt1.appendHtml("<p><b>About FOCAL</b></p> <p>FOCAL is a open-source program that I have been working on. </p> <p><b>Author and affiliation</b></p> <p>I don’t know who to give the credit to?</p> <p><b>How to cite FOCAL</b></p> <p>You’re going to have to ask Daniel how to cite the program </p>")
        text = '''
<p>
    <strong><font size = "7">About FOCAL3D</font></strong><br />
</p>
<p>
    Fast Optimized Clustering Algorithm for Localizations in 3D (FOCAL3D) 
    is an extremely efficient clustering algorithm for analyzing 
    single-molecule localization microscopy datasets.  Developed within 
    the Milstein Laboratory at the University of Toronto, the software 
    is provided as an open-source platform for the microscopy community. <br />

</p>
<p>
    The source code was developed by Daniel Nino and Daniel Djayakarsana, with the help 
    from Muhammad Kamal on the GUI. <br />
</p>
<p>
    <strong><font size = "5">How to cite FOCAL</font></strong><br />
</p>
<p>
    If you  find this code useful, and use FOCAL3D to analyse your data,
    please cite our paper:
    D. Nino, D. Djayakarsana and J. N. Milstein. 
    “FOCAL3D: A 3-dimensional clustering package for single-molecule 
    localization microscopy.” bioRxiv 777722; doi: https://doi.org/10.1101/777722<br />
</p>
<p>
    <strong><font size = "5">License</font></strong><br />
</p>
<p>
    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.<br />
</p>
<p>
    <strong><font size = "5">Report a Bug</font></strong><br />
</p>
<p>
    If you find a bug, please report it to us by emailing josh.milstein@utoronto.ca.
    
    
</p>
        '''
        self.txt1.appendHtml(text)
         
         
        self.txt1.setStyleSheet("background-color: #F2F2F2; border: 0;")
        self.vbox.addWidget(self.txt1)
        self.vbox.addWidget(label)
        self.setLayout(self.vbox)
 
 
        self.show()
         
 
 
class PointSizeWindow(QWidget):
     
    def __init__(self):
        super().__init__()
        self.initUI()
         
    def initUI(self):
        try:
            if type(my_app.loc_table) == np.ndarray: #Check whether the user has opened a localization table
                scriptDir = os.path.dirname(os.path.realpath(__file__))
                self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
                self.setGeometry(200,200,288,83)
                self.setWindowTitle('Point Size')
                 
                self.gridlayout = QGridLayout() 
                self.setLayout(self.gridlayout)
                 
        #        self.size = QLineEdit(self)
                self.size = QSpinBox(self)
                if my_app.is_3d == True:
                    self.size.setRange(1, 20)
                if my_app.is_3d == False:
                    self.size.setRange(5, 20)
                self.label = QLabel("Pixel Size", self)
                self.button = QPushButton("Update", self)
                 
                self.gridlayout.addWidget(self.label, 0,0)
                self.gridlayout.addWidget(self.size, 0,1)
                self.gridlayout.addWidget(self.button, 1,1)
                 
                self.button.clicked.connect(self.change_size)
                self.show()
        except:
            QtGui.QMessageBox.about(self, "Error", "Need to open a localization table")
            
         
    def change_size(self):
        #There will be problems when doing this on loaded file. Might also run into 
        #issues when changing size after clustering since we add new scatterplot item
        #Need to make sure it works. better to use is_3d_bool instead of size of loc_table
        self.new_size = int(self.size.value()) 
        if my_app.is_3d == False:
            my_app.points.setSize(self.new_size)
            try:
                my_app.cluster_point.setSize(self.new_size)
            except:
                pass
        if my_app.is_3d == True:
            my_app.sp.size = self.new_size
            try:
                my_app.clus_Item.size = self.new_size
            except:
                pass
            my_app.sp.update()

class ColorWindow(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setGeometry(300,300, 300, 150)
        self.setWindowTitle("Color")
        
        self.gridlayout = QGridLayout() 
        self.setLayout(self.gridlayout)
        
        label1 = QLabel("red", self)
        label2 = QLabel("blue", self)
        label3 = QLabel("green", self)
        
        self.red = QLineEdit(self)
        self.blue = QLineEdit(self)
        self.green = QLineEdit(self)
        
        self.gridlayout.addWidget(label1, 0, 0)
        self.gridlayout.addWidget(label2, 1, 0)
        self.gridlayout.addWidget(label3, 2, 0)
        
        self.gridlayout.addWidget(self.red, 0, 1)
        self.gridlayout.addWidget(self.blue, 1, 1)
        self.gridlayout.addWidget(self.green, 2, 1)
        
        update_btn = QPushButton("Update", self)
        
        self.gridlayout.addWidget(update_btn, 3, 1)
        
        update_btn.clicked.connect(self.change_color)
        
        
        self.show()
        
    def change_color(self):
        try:
            self.red_val = int(self.red.text())
            self.blue_val = int(self.blue.text())
            self.green_val = int(self.green.text())
            if self.check_valid() == True:
                if my_app.is_3d == False:
                    for line in my_app.line_lst:
                        line.setPen(width = 5, color = (self.red_val, self.blue_val, self.green_val))
                        line.setBrush(self.red_val, self.blue_val, self.green_val)
                if my_app.is_3d == True:
                    for mesh_item in my_app.mesh_lst:
                        mesh_item.setColor((self.red_val, self.blue_val, self.green_val, 1.0))
            else:
                raise Exception
        except:
            QtGui.QMessageBox.about(self, "Error", "The RGB value needs to be an integer between 0-255")


    def check_valid(self):
        if (0 <= self.red_val <= 255) and (0 <= self.blue_val <= 255) and (0 <= self.green_val <= 255):
            return True

            
        

class TextBox(QWidget):
    
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.initUI()
        
    def initUI(self):
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'iconFOCAL.png'))
        self.setWindowTitle("FOCAL3D")
        self.setGeometry(300,300,350,150)
        self.vbox = QVBoxLayout()
        self.textBox = QPlainTextEdit()
        self.vbox.addWidget(self.textBox)
        self.set_text(self.text)
        self.textBox.setReadOnly(True)
        self.textBox.setStyleSheet("background-color: #F2F2F2; border: 0;")
        self.setLayout(self.vbox)
        self.show()
        
    def set_text(self, text):
        self.textBox.setPlainText(text)
        

         
         
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    my_app = MyApp()
    my_app.show()
    sys.exit(app.exec_())