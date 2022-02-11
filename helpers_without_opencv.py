# ---- This is <helpers.py> ----

"""
Module containing functions that are called in segment_and_label.py for image segmentation and GUI creation

@author: cta014
"""
import numpy as np
import os
from loguru import logger
from mixture_model import autofit
from osgeo import gdal
import TLib
import tifffile
import tkinter as tk
from math import sqrt

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
def get_image_dimensions(image_path):
    """Extract spatial image dimensions

    Parameters
    ----------
    image_path : path to image
  
    Returns
    -------
    Nx, Ny : spatial dimension in x and y direction
    Nz: spectral dimension (z direction, if data is 3D)
    
    """
    image = tifffile.imread(image_path)
    dims = image.shape
    
    if len(dims) == 2:
        Nx, Ny = dims[0], dims[1]
        Nz = 0
    
    elif len(dims) == 3:
        Nx, Ny, Nz = dims[0], dims[1], dims[2]
    
    return Nx, Ny, Nz


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def segmentation(data, rows, cols, sub_step_fit_to, MaxC, result_folder):
    """Segmentation of the user-selected region using the automated Mixture of Gaussian segmentation model (Doulgeris, A.P., and Cristea, A.).

    Parameters
    ----------
    data : input to segmentation (i.e. user-selected region)
    rows, cols : spatial dimensions of data
    sub_step_fit_to : sub-stepping value for segmentation
    MaxC: maximum number of classes for segmentation
    result_folder : path to result folder 
        
    Returns
    -------
    model_object : segmentation model object
    labels : labels indicating the different segments
    """ 
    labelFileSuffix = "_results"
    paramFileSuffix = "_parameters"
    output_name = 'segmentation'
    
    Mask = np.ones((rows*cols), dtype=np.uint8)
    flatIA = np.zeros((rows*cols), dtype=np.float32) 
    
    if sub_step_fit_to > 0:
        subMask = np.zeros((rows*cols), dtype=np.uint8)
        subMask[np.round(np.linspace(0, rows*cols, sub_step_fit_to, endpoint=False, dtype=np.int))] = Mask[np.round(np.linspace(0, rows*cols, sub_step_fit_to, endpoint=False, dtype=np.int))]
        logger.info("Sub-sampled fitting to " + str(np.sum(subMask)))
    
    else:
        subMask = Mask
        
    model_object = autofit.fit_mixture_of_gaussians(data[subMask>0, :], max_classes=MaxC)
    
    # logger.debug("Storing model object parameters to toml: "+ output_name + paramFileSuffix + ".toml")
    # model_object.save(output_name + paramFileSuffix + ".toml")
    
    logger.debug("Applying model to predict whole image.")
    labels = (model_object.predict(data, extras={ "IA":flatIA}).reshape((rows, cols))+1)*(Mask.reshape((rows, cols)))
    
    #write output to disk 
    logger.info(model_object.components[0].NAME + " clusters = " + str(model_object.number_of_classes))
    logger.debug("Writing segmentation to disk: "+ output_name + labelFileSuffix + ".tiff")
      
    driver = gdal.GetDriverByName('GTiff')
    out_segment = driver.Create(os.path.join(result_folder, output_name + labelFileSuffix + ".tiff"), cols, rows, 1, gdal.GDT_Byte)
    out_segment_band = out_segment.GetRasterBand(1)
    out_segment_band.SetNoDataValue(0)
    TLib.AddColourTable(out_segment_band, int(np.max(labels.flatten())))
    out_segment_band.WriteArray(labels, 0, 0)
    out_segment_band.FlushCache()
    out_segment_band = None
    out_segment = None
    
    return model_object, labels

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def generate_label_mask(original_rows, original_cols, label_dict, labels, result_folder):
    """Generate label mask of the user-selected region after user has labeled segments.

    Parameters
    ----------
    original_rows : amount of rows in original S1 scene
    original_cols : amount of columns in original S1 scene
    label_dict : dictionary containing pairs of segment-label, as assigned by user in the GUI
    labels: labels indicating the different segments, as produced by segmentation algorithm
    result_path : path to result folder
        
    Returns
    -------
    label_mask : label mask containing the user-labeled segments. Mask has same dimensions as original S1 scene.
    """     
    indices = label_dict['index']
    assigned_labels = label_dict['label']
    conventional_labels = np.select([labels == indices[i] for i in range(len(indices))], assigned_labels, labels)  

    label_mask = np.zeros((original_rows, original_cols), dtype=np.uint8)
    label_mask[:,:] = conventional_labels
    
    logger.debug('Writing label mask to tiff file')
    filename = 'label_mask.tiff'
    tifffile.imsave(os.path.join(result_folder,filename), label_mask)
    return label_mask

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <helpers_without_opencv.py> ----