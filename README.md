# data_labeling_GUI
The difference between this tool and the data_labeling_full_tool, is that this tool assumes the user inputs crops, while the other tool lets the user crop a region from the entire SAR scene.

Tool for quick labeling of remote sensing (SAR) data. Uses the automated Mixture of Gaussian segmentation model (Doulgeris, A.P., and Cristea, A.) to produce segments and asks the user to label these segments via a GUI.

**1. Installation**<br>
Create a virtual environment called `data_labeling`, including specific packages.<br>
For **Windows** and **Linux** systems, this can be done by using the following command in Anaconda:

    conda create -y -n data_labeling python=3.8 gdal=2.3 matplotlib=3.4 scipy=1.7 imageio=2.9 toml=0.10 tifffile lxml

Then activate the environment and install remaining packages:

    conda activate data_labeling
    conda install pyqt qt
    conda install -c anaconda ipython
    conda install loguru

Now clone the repository via the `git clone` command. Insert the url of this repository.
    
    git clone <url>

**2. Usage**<br>
Now you should be able to run the app either from the command line or inside ipython.<br>

Following _positional_ arguments need to be given: 
- path to feature folder
- path to results folder

_Optional_ arguments that can be specified:
- goal: icetypes or ice-water (default is 'icetypes')
- loglevel

**Complete information about the arguments can be found by calling `python run_app_without_cropping.py -h`.**<br>

The feature folder should contain the following: 
- All bands you want to use for segmentation (and subsequent labeling), each band as a seperate file. In .tiff or .img format (the last one is under construction!) (e.g.: 'HH_db.tiff', 'HH_db_ML_7x7.tiff')

NOTE: currently the app requires exactly 2 bands as input! I am working on implementing support of more or less bands.

**Run from command line:** <br>

    python run_app_without_cropping.py path_to_feature_folder path_to_results_folder -goal -loglevel


**Run inside ipython:**<br>

    from segment_and_label import main
    main(path_to_feature_folder, path_to_results_folder, goal, loglevel)
