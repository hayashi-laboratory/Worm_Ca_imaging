import os
import tkinter
from tkinter import filedialog

import czifile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from scipy.signal import find_peaks

import datetime

# Parameters
timedelta = 0.1  # (sec)
# parameters
motion_pixel_threshold = 100
interval_between_bouts = 6
shortest_bout_duration = 12
data_extract_duration_for_m_av_analysis = 5
averaging_window_size = 0.25
tracking_area_lower_threshold = 10
tracking_area_upper_threshold = 100


def extract_images(path):
    img = czifile.imread(path)
    channels = []
    for channel in range(3):
        channel_raster = img[0, :, channel, 0, :, :, 0]
        channels.append(channel_raster)
    channels = np.array(channels)
    GFP = channels[0].astype(np.int64)
    RFP = channels[1].astype(np.int64)
    DIC = channels[2].astype(np.int64)
    return GFP, RFP, DIC


def Calculate_Locomotion(DIC):
    # parameter
    rol_DIC = np.roll(DIC, -1, axis=0)
    subtracted_DIC = DIC - rol_DIC
    subtracted_DIC = subtracted_DIC[:-1]
    subtract_result = subtracted_DIC.reshape(512, subtracted_DIC.shape[0]*56)
    return subtract_result


def select_dir():
    root = tkinter.Tk()
    root.withdraw()
    selected_dir = filedialog.askdirectory()
    os.chdir(selected_dir)
    return selected_dir


def czi_file_extraction(selected_dir):
    files = os.listdir(selected_dir)
    czi_files = [i for i in files if os.path.splitext(i)[1] == '.czi']
    name_list = [os.path.splitext(os.path.basename(i))[0] for i in czi_files]
    return czi_files, name_list


def main():
    selected_dir = select_dir()
    czi_files, name_list = czi_file_extraction(selected_dir)
    GFP, RFP, DIC = extract_images(czi_files[0])
    subtract_result = Calculate_Locomotion(DIC)
    np.savetxt("subtract_result.csv", subtract_result)


if __name__ == '__main__':
    main()
