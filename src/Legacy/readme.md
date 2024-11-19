# Calcium imaging data analysis  


## This repository is for analyzing calcium imaging data in C. elegans

### how to use

raw data
.czi (which is obtained by LSM800)

1. analysis based on Fiji and Trackmate  
   1. open .czi file with fiji (if you use 1st time, you should install TrackMate_extras-0.0.4.jar)
   2. run Trackmate
    3. Conduct tracking (usually, diameter is 5, threshold is 0.01)
    4. save csv file (Experimeng-XX.csv)
   5. save .png of the color merged image (Experiment-XX.png)

2. make directories
   1. run file_move_and_make_dirs.py
   2. select the directory containing subdirectories whose names are Experiment-XX 
   3. save DIC imase sequences to image_seq folder (use /file/save as/image sequences, png or tiff)
   
3. analyzing data using custom made scripts
   1. run Ca_imaging_daga_analysis_RR.py  
Note: This script might give errors if there is only one track for a neuron due to the following functions:
      1. correlation_among_neurons
      2. intensity_histogram

4. extract and summarize the data 
   1. run data_extraction.py