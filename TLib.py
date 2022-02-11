# This library file is for Tony's general set of routines.
import time
import numpy as np
from osgeo import ogr, osr, gdal, gdalconst
from imageio import imsave
import os
from loguru import logger
import subprocess
from scipy import ndimage as ndim
import math
# ==========================================================================================

class Timer:
	def __init__(self):
		# NB. starts on creation.
		self._start_time_ = time.perf_counter()
		
	def restart(self):
		# NB. re-start timer now.
		self._start_time_ = time.perf_counter()
		
	def tsecs(self):
		# NB. current time delta in seconds, timer keeps going.
		return (time.perf_counter() - self._start_time_)
		
	def tstr(self):
		# NB. current time delta as string, timer keeps going.
		secs = self.tsecs()
		if secs < 0.0001:
			return ("{:.3f} microseconds.".format(secs*1000000))
		elif secs < 0.1:
			return ("{:.3f} milliseconds.".format(secs*1000))
		elif secs < 1:
			return ("{:.3f} seconds.".format(secs))
		elif secs < 60:
			return ("{:.1f} seconds.".format(secs))
		elif secs < 120:
			dm2 = divmod(secs, 60)
			return ("{:d} minute and {:d} seconds.".format(int(dm2[0]), round(dm2[1])))
		elif secs < 3600:
			dm2 = divmod(secs, 60)
			return ("{:d} minutes and {:d} seconds.".format(int(dm2[0]), round(dm2[1])))
		elif secs < 7200:
			dm1 = divmod(secs, 3600)
			dm2 = divmod(dm1[1], 60)
			return ("{:d} hour, {:d} minutes and {:d} seconds.".format(int(dm1[0]), int(dm2[0]), round(dm2[1])))
		elif secs < 86400:
			dm1 = divmod(secs, 3600)
			dm2 = divmod(dm1[1], 60)
			return ("{:d} hours, {:d} minutes and {:d} seconds.".format(int(dm1[0]), int(dm2[0]), round(dm2[1])))
		elif secs < 172800:
			dm = divmod(secs, 86400)
			dm1 = divmod(dm[1], 3600)
			dm2 = divmod(dm1[1], 60)
			return ("{:d} day, {:d} hours, {:d} minutes and {:d} seconds.".format(int(dm[0]), int(dm1[0]), int(dm2[0]), round(dm2[1])))
		else:
			dm = divmod(secs, 86400)
			dm1 = divmod(dm[1], 3600)
			dm2 = divmod(dm1[1], 60)
			return ("{:d} days, {:d} hours, {:d} minutes and {:d} seconds.".format(int(dm[0]), int(dm1[0]), int(dm2[0]), round(dm2[1])))


# ==========================================================================================


def AddColourTable(hBand, numclrs):
	allclrs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0.5], [0.8, 0.5, 0.2], [0.5, 0.2, 0.8], [0.2, 0.8, 0.5], [0.8, 0.2, 0.5], [0.5, 0.8, 0.2], [0.2, 0.5, 0.8], [0.25, 0.25, 0.25], [1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1], [1, 1, 0.5], [1, 0.5, 1], [0.5, 1, 1], [0.75, 0.75, 0.75], [0.5, 0.0, 0.1], [0.0, 0.5, 0.1], [0.0, 0.1, 0.5], [0.3, 1.0, 0.7], [1.0, 0.3, 0.5], [1.0, 0.7, 0.4], [0.3, 0.2, 0.1], [0.2, 0.3, 0.1], [0.2, 0.1, 0.3], [0.3, 0.4, 0.5], [0.4, 0.3, 0.5], [0.5, 0.3, 0.4], [0.3, 0.5, 0.4], [0.4, 0.5, 0.3], [0.5, 0.4, 0.3]]
	hCT = gdal.ColorTable()
	for inx in range(numclrs+1):
		if inx < len(allclrs):
			hCT.SetColorEntry(inx, (int(255.0*allclrs[inx][0]), int(255.0*allclrs[inx][1]), int(255.0*allclrs[inx][2]), 255))
		else:
			hCT.SetColorEntry(inx, (np.random.random_integer(0, 255), np.random.random_integer(0, 255), np.random.random_integer(0, 255), 255))
	#hBand.SetRasterColorTable(hCT)
	return


# ==========================================================================================


def add_UAKDemo_class_colour_table_to_tiff(tiff_file):
	class_colors =  [
  [0, 0, 0], 
  [  0,100,255], 
  [  0,  0,  0], 
  [150,200,255], 
  [85, 125,160], 
  [170,40, 240], 
  [170,40, 240], 
  [215,250,130], 
  [0,  120,  0], 
  [237, 60,  5]
  ]
	tmpDS = gdal.Open(tiff_file, gdal.GA_Update)
	hBand = tmpDS.GetRasterBand(1)
	hCT = gdal.ColorTable()
	for inx in range(len(class_colors)):
		hCT.SetColorEntry(inx, (class_colors[inx][0], class_colors[inx][1], class_colors[inx][2], 255))
	hBand.SetRasterColorTable(hCT)
	hCT = None
	hBand = None
	tmpDS = None
	return



# ==========================================================================================


def MatPlotLibColourTable(numclrs):
    # RGB colors in floating point values (range 0.0-1.0) with alpha channel (transparancy added)
	rgb_floating_point_colors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0.5], [0.8, 0.5, 0.2], [0.5, 0.2, 0.8], [0.2, 0.8, 0.5], [0.8, 0.2, 0.5], [0.5, 0.8, 0.2], [0.2, 0.5, 0.8], [0.25, 0.25, 0.25], [1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1], [1, 1, 0.5], [1, 0.5, 1], [0.5, 1, 1], [0.75, 0.75, 0.75], [0.5, 0.0, 0.1], [0.0, 0.5, 0.1], [0.0, 0.1, 0.5], [0.3, 1.0, 0.7], [1.0, 0.3, 0.5], [1.0, 0.7, 0.4], [0.3, 0.2, 0.1], [0.2, 0.3, 0.1], [0.2, 0.1, 0.3], [0.3, 0.4, 0.5], [0.4, 0.3, 0.5], [0.5, 0.3, 0.4], [0.3, 0.5, 0.4], [0.4, 0.5, 0.3], [0.5, 0.4, 0.3]]
	selected_colors = np.append(rgb_floating_point_colors[0:(numclrs+1)], np.ones(((numclrs+1),1)), axis=1)
	return selected_colors



# ==========================================================================================


def create_dB(fil_in, fil_out, dBmin=None, dBmax=None, fil_type=np.uint8):
	tmpDS = gdal.Open(fil_in, gdal.GA_ReadOnly)
	lines, samps = tmpDS.RasterYSize, tmpDS.RasterXSize
	tmpA = tmpDS.ReadAsArray().astype(np.float32)
# 	logger.debug("tmpA.shape "+str(tmpA.shape))
	mx = np.nanmax(tmpA)
	with np.errstate(invalid='ignore'):
		mn = np.nanmin(tmpA[tmpA > 0])
	#logger.debug("mn ", mn, "    mx ", mx)
	tmpI = np.zeros((lines, samps), dtype=np.float32)
# 	logger.debug("tmpI.shape "+str(tmpI.shape))
	tmpI[tmpA > 0] = 10.0*np.log10(tmpA[tmpA > 0])#  convert intensity array to dB
	with np.errstate(invalid='ignore'):
		tmpI[tmpA <= 0] = 10.0*np.log10(mn/2)
	tmpI[np.isfinite(tmpA) == False] =  10.0*np.log10(mn/2)
	# auto-scale and convert to uint8 greyscale.
	mx = np.nan
	mn = np.nan
	if (dBmax is not None):
		mx = dBmax
	if (dBmin is not None):
		mn = dBmin
	if (mx == np.nan):
		mx = np.nanmax(tmpI)
	if (mn == np.nan):
		mn = np.nanmin(tmpI)
	logger.debug("mn " + str(mn) + "    mx " + str(mx))
	logger.debug("mn " + str(np.nanmin(tmpI)) + "    mx " + str(np.nanmax(tmpI)))
	with np.errstate(invalid='ignore'):
		tmpI[tmpI < mn] = mn   #  clip to limits and re-scale
		tmpI[tmpI > mx] = mx
	if fil_type == np.uint8:
		tmpI = 255.0*(tmpI - mn)/(mx-mn)
		tmpI[np.logical_not(np.isfinite(tmpI))] = 0.0
	tmpI = tmpI.astype(fil_type)
	tmpI[np.logical_not(np.isfinite(tmpI))] = np.nan
	#scipy.misc.imsave(fil_out, tmpI)
	imsave(fil_out, tmpI)
	outDS = None
	tmpI = None
	tmpA = None
	tmpDS = None
	driver = None
	return



# ==========================================================================================

def getTPs(latFile, lonFile, TiePointsPerDim=11):
	# expects full file name plus extension. Returns (gcps, tpWKT)
	logger.debug("getting TPs from " + latFile + " & " + lonFile)
	
	latDS = gdal.Open(latFile, gdal.GA_ReadOnly)
	lines, samps = latDS.RasterYSize, latDS.RasterXSize
	tmpA = latDS.ReadAsArray().astype(np.float32)
	# then need to extract tie-point-grid.
	tpx = np.linspace(0, samps-1, TiePointsPerDim, endpoint=True).astype(np.integer)
	tpy = np.linspace(0, lines-1, TiePointsPerDim, endpoint=True).astype(np.integer)
	tpLats = tmpA[np.ix_(tpy, tpx)]
	tmpA = None
	latDS = None
	
	lonDS = gdal.Open(lonFile, gdal.GA_ReadOnly)
	tmpA = lonDS.ReadAsArray().astype(np.float32)
	tpLons = tmpA[np.ix_(tpy, tpx)]
	tmpA = None
	lonDS = None

	gcps = []
	for xi in range(TiePointsPerDim):
		for yi in range(TiePointsPerDim):
			tpgcp = gdal.GCP(tpLons[yi, xi].astype(np.float), tpLats[yi, xi].astype(np.float), 0, tpx[xi].astype(np.float)+1.0, tpy[yi].astype(np.float)+1.0)
			gcps.append(tpgcp)
	# make WKT at this time too.
	source_epsg = 4326	# WGS84 geographic Lat,Lon.
	srs = osr.SpatialReference()
	srs.ImportFromEPSG(source_epsg)
	tpWKT = srs.ExportToWkt()
	# clean up
	tpx = None
	tpy = None
	tpgcp = None
	tpLats = None
	tpLons = None
	srs = None
	source_epsg = None
	# left with gcps and tpWKT for embedding.
	
	return (gcps, tpWKT)


def create_tiepoint_embedded_GeoTiff(data_file_in, data_file_out , lat_image_file, lon_image_file, TiePointsPerDim=11):
	# Takes a target data image and its matching Lat, Lon images, and outputs a tie-point embedded, but un-projected GeoTiff file.
	# Any GDAL readable single band input files (full or relative paths), single band GDAL GTiff output format.
	
	# Get data file to find sizes.
	inDS = gdal.Open(data_file_in, gdal.GA_ReadOnly)
	inLines, inSamps = inDS.RasterYSize, inDS.RasterXSize
	inBand = inDS.GetRasterBand(1)
	inDataType = inBand.DataType
	inData = inDS.ReadAsArray().astype(np.float32)
	inBand = None
	inDS = None

	(gcps, tpWKT) = getTPs(lat_image_file, lon_image_file, TiePointsPerDim)

	# create a new GTiff file for output.
	GTdriver = gdal.GetDriverByName('GTiff')
	#gdal.SetConfigOption("COMPRESS", "DEFLATE")
	outDS = GTdriver.Create(data_file_out, inSamps, inLines, 1, inDataType)
	if outDS == None:
		logger.error("Can't create output file: %s", data_file_out)
	else:
		tmpB = outDS.GetRasterBand(1)
		tmpB.WriteArray(inData, 0, 0)
		tmpB.FlushCache()
	# now embed the tie-points.
	outDS.SetGCPs(gcps, tpWKT)
	outDS.FlushCache()
	tmpB = None
	#gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
	#outDS.BuildOverviews("AVERAGE", [4,16,64])
	outDS = None
	inData = None
	gcps = None
	tpWKT = None


	return


# ==========================================================================================

def PrettyText(MObj, Fil):
	M = MObj.number_of_classes
	d = MObj.components[0].dimensions
	
	TF = open(Fil, "w")

	TF.write("Classes =  {:d}\n\n".format(M))

# 	TF.write("NLS = [")
# 	for sw in range(len(MObj.NLS)):
# 	    TF.write(" {:5.3f} ".format(MObj.NLS[sw]))
# 	TF.write("]\n\n")

	for mi in range(M):
		if mi == 0:
			TF.write("Pjs ({:2d}) ".format(mi))
		else:
			TF.write("    ({:2d}) ".format(mi))
		TF.write(" {: 7.3f}\n".format(MObj.priors[mi]))
	TF.write("\n")

	for mi in range(M):
		if mi == 0:
			TF.write("Mus ({:2d}) ".format(mi))
		else:
			TF.write("    ({:2d}) ".format(mi))
		TF.write(" [")
		for di in range(d):
			TF.write(" {: 7.3f} ".format(MObj.components[mi].mu[di]))
		TF.write("]    ")

		if type(MObj.components[mi]).__name__ is "GLIA":
			if mi == 0:
				TF.write("Slps ({:2d}) ".format(mi))
			else:
				TF.write("     ({:2d}) ".format(mi))
			TF.write(" [")
			for di in range(d):
				TF.write(" {: 7.3f} ".format(MObj.components[mi].beta[di]))
			TF.write("]")
		TF.write("\n")
	TF.write("\n")

	for mi in range(M):
		for di in range(d):
			if di == 0:
				TF.write("Sigs ({:2d}) = |".format(mi))
			else:
				TF.write("            |")
			for dj in range(d):
				TF.write(" {: 7.3f} ".format(MObj.components[mi].cov[di, dj]))
			TF.write("|\n")
		TF.write("\n")


	TF.close()

	return
	

# ==========================================================================================


def SNAPExtractSentinel1Files(s1_input_path, output_parent_directory_path, SNAP_gpt_path="/Applications/snap/bin/gpt", SNAP_graphs_dir = "./graphs", embedLatLonGCPs=True, getLandMask=True, TiePointsPerDim=11, incaName="incAngle", maskName="validMask", LatName="LAT", LonName="LON"):
    
	currentTMR = Timer()
	currentTMR.restart()

	# Quality checking requirements.
    
	if not os.path.exists(s1_input_path):
		logger.error("Input path does not exist: %s", s1_input_path)
		return (1)
        
	if not os.path.isdir(output_parent_directory_path):
		os.makedirs(output_parent_directory_path)
		# time.sleep(0.5)
	if not os.path.isdir(output_parent_directory_path):
		logger.error("Output parent path could not be created: %s", output_parent_directory_path)
		return (2)
		
	parent_dir, file_name = os.path.split(s1_input_path)
	base_name, file_extn = os.path.splitext(file_name)
	if (file_extn not in {'.zip', '.SAFE'}):	# NB: SNAP should be able to directly use .zip file without expansion.
		logger.error("unidentified Product type: %s", file_extn)
		return (3)
    
	if base_name[15] == "H":
		pols = ["hh", "hv"]
	elif base_name[15] == "V":
		pols = ["vh", "vv"]
	else:
		logger.error("unidentified Polarisation mode (from Sentinel-1 ID name). %s", base_name)
		return (4)
    
	files_sub_directory = os.path.join(output_parent_directory_path, base_name)#  this is sub-folder for each scene.
	if not os.path.isdir(files_sub_directory):
		os.makedirs(files_sub_directory)
		# time.sleep(0.5)
	if not os.path.isdir(files_sub_directory):
		logger.error("Output path could not be created: %s", files_sub_directory)
		return (2)

	if not os.path.exists(SNAP_gpt_path):
		logger.error("SNAP gpt not found: %s", SNAP_gpt_path)
		return (5)

	# expect required SNAP graph files.
	if not os.path.exists(os.path.join(SNAP_graphs_dir, "sentinel_1_calibrate_noise_correct2.xml")):
		logger.error("sentinel_1_calibrate_noise_correct2.xml graph not found in: " + SNAP_graphs_dir)
		return (6)
	if not os.path.exists(os.path.join(SNAP_graphs_dir, "landMask.xml")):
		logger.error("landMask.xml graph not found in: ", SNAP_graphs_dir)
		return (6)
	if not os.path.exists(os.path.join(SNAP_graphs_dir, "incAngle.xml")):
		logger.error("incAngle.xml graph not found in: ", SNAP_graphs_dir)
		return (6)
	if embedLatLonGCPs:
		if not os.path.exists(os.path.join(SNAP_graphs_dir, "LAT.xml")):
			logger.error("LAT.xml graph not found in: ", SNAP_graphs_dir)
			return (6)
		if not os.path.exists(os.path.join(SNAP_graphs_dir, "LON.xml")):
			logger.error("LON.xml graph not found in: ", SNAP_graphs_dir)
			return (6)



    # Now can actually start actual import/extract. Already have s1_input_path to .SAFE or .zip S-1 product and files_sub_directory for output tiffs.
    	# already have pols.
	if file_extn == ".zip":
		SourceProduct = s1_input_path
	else: # for .SAFE find the manifest.
		SourceProduct = os.path.join(s1_input_path, "manifest.safe")
		# before calling SNAP, go through unpacked product folder and remove any ".DS_Store" files, particularly from the /annotations sub-folder.
		if os.path.exists(os.path.join(os.path.join(s1_input_path, "annotation"), ".DS_Store")):
			os.remove(os.path.join(os.path.join(s1_input_path, "annotation"), ".DS_Store"))


	if embedLatLonGCPs:
		target = os.path.join(files_sub_directory, LatName + ".tif")
		if not os.path.exists(target):
			runstr = " " + os.path.join(SNAP_graphs_dir, "LAT.xml") + " -Psource=" + SourceProduct + " -PoutputFolder=" + files_sub_directory + " -PoutputName=" + LatName
			logger.debug(SNAP_gpt_path + runstr)
			subprocess.run(SNAP_gpt_path + runstr, shell=True)
		
		target = os.path.join(files_sub_directory, LonName + ".tif")
		if not os.path.exists(target):
			runstr = " " + os.path.join(SNAP_graphs_dir, "LON.xml") + " -Psource=" + SourceProduct + " -PoutputFolder=" + files_sub_directory + " -PoutputName=" + LonName
			logger.debug(SNAP_gpt_path + runstr)
			subprocess.run(SNAP_gpt_path + runstr, shell=True)

		tmpFBN = os.path.join(files_sub_directory, LatName)
		tmpDS = gdal.Open(tmpFBN + '.tif', gdal.GA_ReadOnly)
		lines, samps = tmpDS.RasterYSize, tmpDS.RasterXSize
		tmpA = tmpDS.ReadAsArray().astype(np.float32)
		# then need to extract tie-point-grid.
		tpx = np.linspace(0, samps-1, TiePointsPerDim, endpoint=True).astype(np.integer)
		tpy = np.linspace(0, lines-1, TiePointsPerDim, endpoint=True).astype(np.integer)
		tpLats = tmpA[np.ix_(tpy, tpx)]
		tmpA = None
		tmpDS = None
		
		tmpFBN = os.path.join(files_sub_directory, LonName)
		tmpDS = gdal.Open(tmpFBN + '.tif', gdal.GA_ReadOnly)
		tmpA = tmpDS.ReadAsArray().astype(np.float32)
		tpLons = tmpA[np.ix_(tpy, tpx)]
		tmpA = None
		tmpDS = None

		gcps = []
		for xi in range(TiePointsPerDim):
			for yi in range(TiePointsPerDim):
				tpgcp = gdal.GCP(tpLons[yi, xi].astype(np.float), tpLats[yi, xi].astype(np.float), 0, tpx[xi].astype(np.float)+1.0, tpy[yi].astype(np.float)+1.0)
				gcps.append(tpgcp)
		# make WKT at this time too.
		source_epsg = 4326	# WGS84 geographic Lat,Lon.
		srs = osr.SpatialReference()
		srs.ImportFromEPSG(source_epsg)
		tpWKT = srs.ExportToWkt()
		# clean up
		tpx = None
		tpy = None
		tpgcp = None
		tpLats = None
		tpLons = None
		srs = None
		source_epsg = None
		# left with gcps and tpWKT for embedding.

	DataMask = None
	DMrows = 0
	DMcols = 0
	for pl in pols:
		target = os.path.join(files_sub_directory, pl.upper() + ".tif")
		dBtarget = os.path.join(files_sub_directory, "Sigma0_" + pl.upper() + "_db.tif")
		if not os.path.exists(target):
			runstr = " " + os.path.join(SNAP_graphs_dir, "sentinel_1_calibrate_noise_correct2.xml") + " -Psource=" + SourceProduct + " -PoutputFolder=" + files_sub_directory + " -Ppolarization=" + pl.upper()
			logger.debug(SNAP_gpt_path + runstr)
			subprocess.call(SNAP_gpt_path + runstr, shell=True)
			if embedLatLonGCPs:
				tmpDS = gdal.Open(target, gdal.GA_Update)
				tmpDS.SetGCPs(gcps, tpWKT)
				tmpDS.FlushCache()
				tmpDS = None
		# Open it and look for exactly zero data values to add these to the mask.
		tmpDS = gdal.Open(target, gdal.GA_ReadOnly)
		DMrows, DMcols = tmpDS.RasterYSize, tmpDS.RasterXSize
		tmpD = np.greater(tmpDS.ReadAsArray().astype(np.float32), 0.0) # NB: data is still linear, so zero values are bad-data.
		if DataMask is None:
			DataMask = tmpD
		else:
			DataMask *= tmpD
		tmpD = None
		tmpDS = None
		
		if not os.path.exists(dBtarget):
			logger.info("Creating dB version of Sigma0_" + pl.upper() + ".tif")
# 				if (pl[0] == pl[1]):
# 					minLim = -30
# 					maxLim = 0
# 				else:
# 					minLim = -35
# 					maxLim = -5
# 				create_dB(target, dBtarget, dBmin=minLim, dBmax=maxLim)
			create_dB(target, dBtarget, dBmin=None, dBmax=None, fil_type=np.float32) # without limit clipping.
			if embedLatLonGCPs:
				tmpDS = gdal.Open(dBtarget, gdal.GA_Update)
				tmpDS.SetGCPs(gcps, tpWKT)
				tmpDS.FlushCache()
				tmpDS = None


	iatarget = os.path.join(files_sub_directory, incaName + ".tif")
	if not os.path.exists(iatarget):
		runstr = " " + os.path.join(SNAP_graphs_dir, "incAngle.xml") + " -Psource=" + SourceProduct + " -PoutputFolder=" + files_sub_directory + " -PoutputName=" + incaName
		logger.debug(SNAP_gpt_path + runstr)
		subprocess.call(SNAP_gpt_path + runstr, shell=True)
		if embedLatLonGCPs:
			tmpDS = gdal.Open(iatarget, gdal.GA_Update)
			tmpDS.SetGCPs(gcps, tpWKT)
			tmpDS.FlushCache()
			tmpDS = None

	# get a land mask, if requested, and merge with zero-data mask.
	if getLandMask:
		lmtarget = os.path.join(files_sub_directory, maskName + ".tif")
		if not os.path.exists(lmtarget):
			runstr = " " + os.path.join(SNAP_graphs_dir, "landMask.xml") + " -Psource=" + SourceProduct + " -PoutputFolder=" + files_sub_directory + " -PoutputName=" + maskName
			logger.debug(SNAP_gpt_path + runstr)
			subprocess.call(SNAP_gpt_path + runstr, shell=True)
			# SNAP doesn't add colour table, so add it for easy previewing in Finder.
			if os.path.exists(lmtarget):
				tmpDS = gdal.Open(lmtarget, gdal.GA_Update)
				if tmpDS == None:
					logger.info("****************")
					logger.error("Can't open SNAPs mask file: %s", lmtarget)
					logger.info("****************")
				else:
					tmpB = tmpDS.GetRasterBand(1)
					tmpA = tmpB.ReadAsArray()
					# and combine with any DataMask from the polarisation channels.
					if np.nansum(tmpA) == 0:
						tmpA = np.tile(1, (tmpDS.RasterYSize, tmpDS.RasterXSize))
						logger.warning("empty mask file substituted to full image.")
					if DataMask is not None:
						logger.debug("merged landmask with datamask.")
						tmpA *= DataMask
					tmpB.WriteArray(tmpA, 0, 0)
					AddColourTable(tmpB, 1)
					tmpB.FlushCache()
				tmpB = None
				tmpDS = None
			else:
				logger.warning("Mask target not created: " + lmtarget)
	else:
		lmtarget = os.path.join(files_sub_directory, maskName + ".tif")
		if not os.path.exists(lmtarget):
			GTdriver = gdal.GetDriverByName('GTiff')
			#gdal.SetConfigOption("COMPRESS", "DEFLATE")
			outDS = GTdriver.Create(lmtarget, DMcols, DMrows, 1, gdal.GDT_Byte)
			if outDS == None:
				logger.error("Can't create output file: %s", lmtarget)
			else:
				tmpB = outDS.GetRasterBand(1)
				if DataMask is not None:
					logger.debug("Setting datamask as valid.")
					tmpB.WriteArray(DataMask, 0, 0)
				else:
					logger.debug("No datamask, set to ones.")
					tmpB.WriteArray(np.ones((DMrows, DMcols), dtype=bool), 0, 0)
				tmpB.FlushCache()
			tmpB = None
			#gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
			#outDS.BuildOverviews("AVERAGE", [4,16,64])
			outDS = None
			GTdriver = None



	logger.info(" ")
	logger.info("SNAP processing took " + currentTMR.tstr())
	logger.info(" ")

	return (0)

# ==========================================================================================


# ==========================================================================================
#	STILL IN DEVELOPMENT, NOT YET TESTED.
# ==========================================================================================
def MultilookAndSubstepFiles(fileStem_list, files_parent_directory_path, Looks=[1, 1], Steps=[1, 1], embedLatLonGCPs=True, TiePointsPerDim=11, LatName="LAT", LonName="LON", fextn=".tif", useMaskStem=None):
    
	def multilookstep_file(fil_in, fil_out, Looks, steps, vMask=None):
		inDS = gdal.Open(fil_in, gdal.GA_ReadOnly)
		inLines, inSamps = inDS.RasterYSize, inDS.RasterXSize
		inBand = inDS.GetRasterBand(1)
		inDataType = inBand.DataType
		inData = inDS.ReadAsArray().astype(np.float32)
		inBand = None
		inDS = None
		if (Looks[0]*Looks[1] > 1):
			# Multi-look it.
			if vMask is not None: # then will just mask.sum in box and divide by count afterwards.
				f = np.ones((int(Looks[0]), int(Looks[1])))
				inData = ndim.convolve(vMask*inData, f, mode="reflect")
				# now need to divide by mask sum count.
				vSum = ndim.convolve(vMask, f, mode="reflect")
				vSum[vSum == 0] = 1 # to avoid div.by.zero.
				inData /= vSum
			else: # just normal running mean.
				f = np.ones((int(Looks[0]), int(Looks[1])))/(Looks[0]*Looks[1])
				inData = ndim.convolve(inData, f, mode="reflect")
		if (steps[0]*steps[1] > 1):
			# Sub-step it.
			inData = inData[range(int(math.floor(Steps[0]/2.0)), inLines, np.int(Steps[0]))]
			inData = inData[:, range(int(math.floor(Steps[1]/2.0)), inSamps, np.int(Steps[1]))]
		if (Looks[0]*Looks[1] > 1) or (steps[0]*steps[1] > 1):
			outLines, outSamps = inData.shape[0], inData.shape[1]
			# create a new GTiff file for output.
			GTdriver = gdal.GetDriverByName('GTiff')
			#gdal.SetConfigOption("COMPRESS", "DEFLATE")
			outDS = GTdriver.Create(fil_out, outSamps, outLines, 1, inDataType)
			if outDS == None:
				logger.error("Can't create output file: %s", fil_out)
			else:
				tmpB = outDS.GetRasterBand(1)
				tmpB.WriteArray(inData, 0, 0)
				tmpB.FlushCache()
			tmpB = None
			#gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
			#outDS.BuildOverviews("AVERAGE", [4,16,64])
			outDS = None
			GTdriver = None
		inData = None
		f = None
		inDataType = None
		
		return
		
	# main MultilookAndSubstepFiles() starts here.
		
	currentTMR = Timer()
	currentTMR.restart()

	if embedLatLonGCPs:
		if (not os.path.exists(os.path.join(files_parent_directory_path, LatName + fextn))) or (not os.path.exists(os.path.join(files_parent_directory_path, LonName + fextn))):
			logger.error("Needs both " + LatName + " and " + LonName + " files.")
			return (1, "")
		if (LatName not in fileStem_list):
			fileStem_list.append(LatName)
		if (LonName not in fileStem_list):
			fileStem_list.append(LonName)
			# to make sure that they will be multi-looked and stepped ready for embedding.
		
	Lstr = ""
	Sstr = ""
	L = Looks[0]*Looks[1]
	S = Steps[0]*Steps[1]
	if L > 1:
		Lstr = "_L" + str(int(Looks[0])) + 'x' + str(int(Looks[1]))
	if S > 1:
		Sstr = "_S" + str(int(Steps[0])) + 'x' + str(int(Steps[1]))
	suffix = Lstr + Sstr
	
	if (L > 1) or (S > 1): # else nothing to do.
		# first do all the files, which may include Lat & Lon.
		logger.info("pre-processing to " + suffix)

		useMask = None
		if useMaskStem is not None:
			# get mask array first.
			mskDS = gdal.Open(os.path.join(files_parent_directory_path, useMaskStem + fextn), gdal.GA_ReadOnly)
			mskBand = mskDS.GetRasterBand(1)
			useMask = mskDS.ReadAsArray().astype(np.float32)
			mskBand = None
			mskDS = None


		modified_list = np.zeros((len(fileStem_list)))
		for f in range(len(fileStem_list)):
			fil = fileStem_list[f]
			if not os.path.exists(os.path.join(files_parent_directory_path, fil + suffix + fextn)):
				if useMask is not None:
					logger.debug("masked-pre-processing " + fil)
				else:
					logger.debug("pre-processing " + fil)
				modified_list[f] = 1
				multilookstep_file(os.path.join(files_parent_directory_path, fil + fextn), os.path.join(files_parent_directory_path, fil + suffix + fextn), Looks, Steps, vMask=useMask)
		if np.sum(modified_list) == 0:
			logger.debug("no files needed pre-processing.")
		else:
			if embedLatLonGCPs:
				logger.info("embedding TPs.")
				(gcps, tpWKT) = getTPs(os.path.join(files_parent_directory_path, LatName + Lstr + Sstr + fextn), os.path.join(files_parent_directory_path, LonName + suffix + fextn), TiePointsPerDim)
				for f in range(len(fileStem_list)):
					fil = fileStem_list[f]
					if (fil == LonName) or (fil == LatName):
						# don't need to embed GCPs.
						logger.debug("don't embed TPs to " + fil)
					elif modified_list[f]:
						target = os.path.join(files_parent_directory_path, fil + suffix + fextn)
						tmpDS = gdal.Open(target, gdal.GA_Update)
						tmpDS.SetGCPs(gcps, tpWKT)
						tmpDS = None
						logger.debug("Embedded TPs to " + fil)

		if useMask is not None:
			useMask = None
	# may want to add a clean up previous files later.
	
	logger.info(" ")
	logger.info("Scene pre-processing took " + currentTMR.tstr())
	logger.info(" ")

	return (0, suffix)

# ==========================================================================================


