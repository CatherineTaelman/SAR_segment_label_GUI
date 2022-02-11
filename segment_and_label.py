# ---- This is <segment_and_label.py> ----

"""
Module implementing a segmentation algorithm (Gaussian mixture model), 
followed by a GUI that enables the user to label the obtained segments.

Adapted from crop_and_segment_and_label.py (this version does not give the possibility
                                            to make a crop in the original image)

@author: Catherine Taelman
"""

import matplotlib.pyplot as plt
import os
import sys
import pathlib
import numpy as np
from loguru import logger
import matplotlib.figure
from matplotlib.colors import ListedColormap
import tifffile

import TLib
from helpers_without_opencv import get_image_dimensions, segmentation, generate_label_mask

from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QListWidget, QListWidgetItem, QMessageBox, QPushButton,
        QVBoxLayout)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# GUI window, which inherits QDialog
class Window(QDialog):

    def __init__(self, parent=None, data = [], existing_bands = [], nr_classes = 1, labels = [], result_folder = '',
                 sub_step_fit_to = 1, labels_list = [], MaxC = 1, original_rows = 1, original_cols = 1,
                         result_path = ''):
        super(Window, self).__init__(parent)
        self.originalPalette = QApplication.palette()
        self.data = data
        self.existing_bands = existing_bands
        self.nr_classes = nr_classes
        self.labels = labels
        self.result_folder = result_folder
        self.sub_step_fit_to = sub_step_fit_to
        self.labels_list = labels_list
        self.MaxC = MaxC
        self.original_rows = original_rows
        self.original_cols = original_cols
        self.result_path = result_path
        
        # track when user changes combobox values
        global user_changes
        user_changes = False
    
        self.setWindowTitle('Segmentation results')
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        
        plt.ioff() #prohibit empty figure from displaying at this point
        self.figure = matplotlib.figure.Figure()
        
        # the Canvas Widget displays the 'figure', it takes the
        # 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # the Navigation widget takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.createBottomLeftGroupBox(self.labels_list)
        self.createBottomRightGroupBox(self.nr_classes, self.labels_list)
        self.plot(self.data, self.existing_bands, self.nr_classes, self.labels)
        
        self.savebtn = QPushButton('Save labels', self)
        self.savebtn.clicked.connect(self.saveLabels)
        
        self.refinebtn = QPushButton('Run segmentation on finer resolution', self)
        self.refinebtn.clicked.connect(self.finerSegmentation)
        
        global mainLayout
        mainLayout = QGridLayout()
        # LAYOUT OF WINDOW:
        # first row = toolbar
        # second row = plots
        # third row = list of labels + pop-up menus
        # fourth row = buttons
        
        mainLayout.addWidget(self.bottomLeftGroupBox, 2, 0) # add widget in row 2, col 0
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        mainLayout.setRowStretch(1, 2) # row 1 is stretched with factor 2
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1) 
        mainLayout.setColumnStretch(1, 2)
        
        mainLayout.addWidget(self.toolbar, 0,0)
        mainLayout.addWidget(self.canvas, 1, 0, 1, 2)
        mainLayout.addWidget(self.savebtn, 3, 1)
        mainLayout.addWidget(self.refinebtn, 3, 0)
        self.setLayout(mainLayout)
   
    
    def finerSegmentation(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Re-segmentation will reset all label values; plots will be re-drawn")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        self.sub_step_fit_to, self.nr_classes, self.labels = helperFunctions.reRunSegmentation(self.data, self.sub_step_fit_to, self.MaxC, self.result_folder)
        global user_changes
        user_changes = False
        print('New amount of classes detected: ', self.nr_classes)
        self.plot(self.data, self.existing_bands, self.nr_classes, self.labels)
        mainLayout.removeWidget(self.bottomRightGroupBox)
        self.bottomRightGroupBox.close()
        self.createBottomRightGroupBox(self.nr_classes, self.labels_list)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        
    def saveLabels(self):
        global label_mask
        label_mask = self.writeSettings(self.result_folder, self.result_path, self.original_rows, self.original_cols, self.labels)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Lavels have been saved to label mask")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        global user_changes
        user_changes = False
        
    def closeEvent(self, event):
        if user_changes:
            quit_msg = "Unsaved changes detected. Do you want to save the labels before closing the application?"
            reply = QMessageBox.question(self, 'Message', 
                         quit_msg, QMessageBox.Yes, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.saveLabels()
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()
        
        else:
            quit_msg = "Are you sure you want to close the application?"
            reply = QMessageBox.question(self, 'Message', 
                         quit_msg, QMessageBox.Yes, QMessageBox.No)

            if reply == QMessageBox.Yes:
                event.accept()
            elif reply == QMessageBox.No:
                event.ignore()
    
    def writeSettings(self, result_folder, result_path, original_rows, original_cols, labels):
        logger.debug("Writing user labels to txt file")
        label_dict = {'index': [], 'label': []}
        
        for i,box in enumerate(combobox_list):
            label_index = box.currentIndex()
            label = box.currentText()
            
            label_dict['index'].append(i+1) # this is the segment numbers
            label_dict['label'].append(label_index) # this is the label indices associated with each segment
            
            if label_index != 0:
                file = open(os.path.join(result_folder,"user_labels.txt"), "a+")
                file.write('segment ' + str(i+1) + ': class ' + str(label_index) + ' (' + label + ')' + '\n')
                file.close()
                
        label_mask = generate_label_mask(original_rows, original_cols, label_dict, labels, result_folder)
        
        return label_mask
    
    def createBottomLeftGroupBox(self, labels_list):
        self.bottomLeftGroupBox = QGroupBox("List of conventional labels")
        listWidget = QListWidget()
        for item in labels_list:
            QListWidgetItem(item, listWidget)
        layout = QVBoxLayout()
        layout.addWidget(listWidget)
        
        self.bottomLeftGroupBox.setLayout(layout)  
        
    def createBottomRightGroupBox(self, nr_classes, labels_list):
        self.bottomRightGroupBox = QGroupBox("Label segments. NOTE: only label a segment if the *whole* segment belongs to one class")
        layout = QGridLayout()
        
        # extract colors from TLib library, to match with figure of segments
        # convert the rgb floating points colors to QColor colours
        rgb_floating_point_colors = TLib.MatPlotLibColourTable(nr_classes)
        rgb_floating_point_colors = rgb_floating_point_colors[1:]
        
        Qcolor_list = []
        for color in rgb_floating_point_colors:
            Qcolor = QColor.fromRgbF(color[0],color[1],color[2], color[3])
            Qcolor_list.append(Qcolor)

        label_list = []
        global combobox_list
        combobox_list = []  
        
        #create a label and combobox for each segment
        for i in range(len(rgb_floating_point_colors)):
            label = QLabel(self)
            label.setAutoFillBackground(True)
            palette = QPalette()
            palette.setColor(QPalette.Window, Qcolor_list[i])
            label.setPalette(palette)
            label.setAlignment(Qt.AlignCenter)
            
            global labelsComboBox
            labelsComboBox = QComboBox(self)
            labelsComboBox.addItems(labels_list)
            
            # when user changes index of a combobox, set user_changes = True
            labelsComboBox.currentIndexChanged.connect(self.trackChanges)
            label.setBuddy(labelsComboBox)
            label_list.append(label)
            combobox_list.append(labelsComboBox)
        
        row = 0
        for label, comboBox in zip(label_list, combobox_list):
            col = 0
            layout.addWidget(label, row, col)
            col = 1
            layout.addWidget(comboBox, row, col)
            row += 1
            
        self.bottomRightGroupBox.setLayout(layout)
     
    def trackChanges(self):
        global user_changes
        user_changes = True
        
    def plot(self, data, existing_bands, nr_classes, labels):

        # clearing old figure
        self.figure.clf()
        # Create nx1 sub plots showing the different bands and segmention result
        ax1 = self.figure.add_subplot(1,3,1)
        ax1.imshow(data[:,:,0], cmap = 'gray')
        ax1.axis('off')
        title1 = existing_bands[0].strip('.tiff')
        ax1.set_title(title1)
        
        ax2 = self.figure.add_subplot(1,3,2, sharex=ax1, sharey=ax1)
        ax2.imshow(data[:,:,1], cmap = 'gray')
        ax2.axis('off')
        title2 = existing_bands[1].strip('.tiff')
        ax2.set_title(title2)
        
        ax3 = self.figure.add_subplot(1,3,3, sharex=ax1, sharey=ax1)
        colors = TLib.MatPlotLibColourTable(nr_classes)
        colors = colors[1:]
        custom_cmap = ListedColormap(colors)
        ax3.axis('off')
        ax3.set_title('segmentation')
        ax3.imshow(labels, cmap = custom_cmap, interpolation = 'nearest')
        
        # refresh canvas
        self.canvas.draw()
        
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #        

class helperFunctions():
    def reRunSegmentation(data, sub_step_fit_to, MaxC, result_folder):
        rows, cols, dims = data.shape
        data_flattened = np.reshape(data,((rows*cols),dims))
        sub_step_fit_to = sub_step_fit_to * 2
        model_object, labels = segmentation(data_flattened, rows, cols, sub_step_fit_to, MaxC, result_folder)
        nr_classes = model_object.number_of_classes
        return sub_step_fit_to, nr_classes, labels


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
def import_existing_bands(existing_bands, feat_folder):
    
    # read out data of each band and stack all bands into 3D data array
    for i in range(len(existing_bands)):
        input_image_name = existing_bands[i]
        input_image = tifffile.imread(os.path.join(feat_folder, input_image_name))
        
        if i == 0:
            data = input_image
            
        if i > 0:
            data = np.stack(((data, input_image)), axis=-1)
            
    return data
            
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #           
            
def segment(data, sub_step_fit_to, MaxC, result_folder):
    
    # flatten data to input into segmentation algorithm
    rows, cols, dims = data.shape
    data_flattened = np.reshape(data,((rows*cols),dims))
    
    # segmentation algorithm
    model_object, labels = segmentation(data_flattened, rows, cols, sub_step_fit_to, MaxC, result_folder)
    
    return labels, model_object

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
            
            
def gui(data, existing_bands, nr_classes, labels, result_folder, sub_step_fit_to,
        labels_list, MaxC, original_rows, original_cols, result_path):
    
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    
    app.setStyle('Fusion')
    app.setQuitOnLastWindowClosed(True)
    main_window = Window(data = data, existing_bands = existing_bands, nr_classes=nr_classes, labels = labels, result_folder = result_folder, sub_step_fit_to = sub_step_fit_to, labels_list=labels_list, MaxC = MaxC,
                         original_rows = original_rows, original_cols = original_cols, result_path = result_path)
    main_window.show()
    sys.exit(app.exec_())
    return

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def main(feat_folder, result_folder, goal = 'icetypes', loglevel = 'INFO'):
    plt.close('all')
        
    # remove default logger handler and add personal one
    logger.remove()
    logger.add(sys.stderr, level=loglevel)

    # -------------------------------------------------------------------------------------
    # define some variables
    sub_step_fit_to = 500 # start value for subsampling during segmentation
    MaxC = 10      # maximum number of classes for segmentation
    
    # convert folder strings to paths
    feat_folder     = pathlib.Path(feat_folder).expanduser().absolute()
    result_folder   = pathlib.Path(result_folder).expanduser().absolute()
    
    if not feat_folder.is_dir():
        logger.error('Cannot find feat_folder: {feat_folder}')
        raise NotADirectoryError('Cannot find feat_folder: {feat_folder}')
    
    if not result_folder.is_dir():
        result_folder.mkdir(parents=True, exist_ok=True)
        
    logger.debug(f'feat_folder:     {feat_folder}')
    logger.debug(f'result_folder:   {result_folder}')
    
    if not goal in ['icetypes', 'ice-water']:
        logger.error('-goal argument invalid, choose from: icetypes, ice-water')
        raise ValueError(
          'invalid value given for -goal argument'
        )
        sys.exit()
        
    if goal == 'icetypes':
        labels_file = 'icetypes.txt' 
    elif goal == 'ice-water':
        labels_file = 'ice_water.txt' 
     
    # define output file name and path
    scene = feat_folder.stem
    result_basename = scene + '_labelMask'
    result_path     = result_folder / f'{result_basename}.tiff'
    
    # logger
    logger.debug(f'result_path {result_path}')
    
    # check if outfile already exists
    if result_path.is_file():
      logger.info('Output files already exist, newly-labeled data will be added to already existing label mask file')

    # -------------------------------------------------------------------------------------

    # CHECK EXISTING FEATURES
        
    # get list of existing bands in feat_folder
    existing_bands = sorted(
      [f for f in os.listdir(feat_folder) if (f.endswith('.tif') or f.endswith('.tiff')or f.endswith('.img'))]
    )
    
    logger.info( f'Found following bands: {existing_bands}')
                
    # get Nx, Ny and Nz from first band
    Nx, Ny, Nz = get_image_dimensions(
      (feat_folder / f'{existing_bands[0]}').as_posix()
    )
    
    if Nz > 1 :
        logger.error('Dimension error: enter each band seperately')
        raise ValueError('Band has depth dimension larger than 1')
        
    # check that all existing features have same dimensions
    logger.info('Checking if dimensions of existing bands match')
    for f in existing_bands:
      Nx_current, Ny_current, Nz_current = get_image_dimensions(
        (feat_folder / f'{f}').as_posix()
      )
      if not Nx == Nx_current or not Ny == Ny_current or Nz_current > 1:
        logger.error('Image dimensions of existing bands do not match')
        raise ValueError('Image dimensions of existing bands do not match')

    with open(labels_file) as f:
        labels_list = f.read().splitlines()
    
    data = import_existing_bands(existing_bands, feat_folder)
    
    labels, model_object = segment(data, sub_step_fit_to, MaxC, result_folder)
    
    nr_classes = model_object.number_of_classes
    
    gui(data, existing_bands, nr_classes, labels, result_folder, sub_step_fit_to, labels_list, MaxC, Nx, Ny, result_path)                 

# # # UNCOMMENT FOLLOWING LINES TO RUN INSIDE SPYDER EDITOR    
# feat_folder = r"C:\Users\cta014\OneDrive - UiT Office 365\work\data\sentinel-1\S1\features\S1A_EW_GRDM_1SDH_20210210T044428_20210210T044533_036523_0449EE_0B02"
# result_folder = r"C:\Users\cta014\OneDrive - UiT Office 365\work\data\sentinel-1\S1\results\S1A_EW_GRDM_1SDH_20210210T044428_20210210T044533_036523_0449EE_0B02"

# main(feat_folder, result_folder)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <segment_and_label.py> ----