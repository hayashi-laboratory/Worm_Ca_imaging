#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jun 12 10:01:09 2021

@author: miyazakishinichi

workflow
1. select 2 folders, one is for csv and png, the other is for images
2. make folders
3. image subtraction analysis
4. calc fluo and make figures
"""

import os, sys
import pandas as pd
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import datetime
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import argrelmax
from scipy.signal import find_peaks

from tqdm import tqdm
from tkinter import filedialog
import tkinter
from tkinter import messagebox

# static variables
# max_data_length: analyzed data length
max_data_length =2000
# spatial correction
correct_value = 0.614
# parameter
threshold = 100  # pixel
shortest_motion_bout_duration = 12  # sec6
interval_between_bouts = 6

# averaging parameters
averaging_window_size = 0.25  # (sec)

# locomotor active neuron criteria
# if the average activity was increased over 0.2, (baseline is before)
# the neuron was assigned as locomotor active neuron
neural_activity_threshold = 0.2
data_extract_duration_for_m_av_analysis = 5

# font path
if os.name == "nt":
    font = ImageFont.truetype("C:/Windows/WinSxS/amd4_microsoft" \
                              "-windows-font-truetype-arial_" \
                              "31bf3856ad364e35_10.0.18362.1" \
                              "_none_44e0e02b2a9382cc/arial.ttf", 14)
elif os.name == "posix":
    font = ImageFont.truetype("/Library/Fonts/Microsoft/Arial.ttf")
else:
    messagebox.showinfo('quit', 'os is not recognized')
    sys.exit()

####graph params#####
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams["figure.dpi"] = 300


############################################################################

# 1. select folders
def dir_select(folderpath):
    # select the directory which contain multiple Ex-XX
    os.chdir(folderpath)
    # obtain experiment num, csv path
    csv_path = [i for i in os.listdir("./") \
                if os.path.splitext(i)[1] == '.csv']
    if not csv_path:
        messagebox.showinfo('cancel', 'there is no csv file in the dir')
        sys.exit()
    elif len(csv_path) > 1:
        messagebox.showinfo('cancel', 'there are multiple csv files in the dir')
        sys.exit()
    elif csv_path[0].split("-")[0] != "Experiment":
        messagebox.showinfo('cancel', 'csv file name is not Experiment-xx')
        sys.exit()
    else:
        Ex_name = csv_path[0].split(".")[0]
        csv_path = csv_path[0]

    # get png path
    png_path = Ex_name + ".png"
    if os.path.isfile(png_path):
        pass
    else:
        messagebox.showinfo('cancel', 'there is no png image')
        sys.exit()

    # get image seq path
    if os.path.isdir("./image_seq"):
        image_seq_path = "./image_seq"
    else:
        messagebox.showinfo('cancel', 'there is no image_seq dir')
        sys.exit()
    image_seq_list = [i for i in os.listdir(image_seq_path) \
                      if os.path.splitext(i)[1] == '.tif' or \
                      os.path.splitext(i)[1] == '.png'][0:max_data_length]

    return Ex_name, csv_path, png_path, image_seq_list


# 2. make dirs
def makedirs(Ex_name):
    Ex_num = Ex_name.split("-")[1]
    date = datetime.date.today()
    os.makedirs('./Ex-{0}_date_{1}'.format(Ex_num, date), exist_ok=True)
    os.chdir('./Ex-{0}_date_{1}'.format(Ex_num, date))
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./fig', exist_ok=True)
    result_data_path = './Ex-{0}_date_{1}/data'.format(Ex_num, date)
    result_fig_path = './Ex-{0}_date_{1}/fig'.format(Ex_num, date)
    os.chdir("../")
    return result_data_path, result_fig_path


def rgb_to_gray(src):
    # obtain individual values
    b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    # RGB to gray
    return np.array(0.2989 * r + 0.5870 * g + 0.1140 * b, dtype='float32')


def image_subtraction_analysis(image_seq_list):
    path_list = ["./image_seq/" + i for i in image_seq_list]
    image_number = len(image_seq_list) - 1
    if image_number < 100:
        messagebox.showinfo('cancel', 'there are not enough amount of images')
    else:
        pass
    sampling_cycle = (image_number // 100) - 1
    threshold_list = []
    for p in tqdm(range(100)):
        sample_num = p * sampling_cycle
        sample_temp = path_list[sample_num]
        next_to_sample_temp = path_list[sample_num + 1]
        sample_img = cv2.imread(sample_temp)
        sample_img = rgb_to_gray(sample_img)
        next_to_sample_temp_img = cv2.imread(next_to_sample_temp)
        next_to_sample_temp_img = rgb_to_gray(next_to_sample_temp_img)
        if p == 0:
            sample_subtract_img = np.abs(sample_img - next_to_sample_temp_img)
        else:
            temp_subtract_img = np.abs(sample_img - next_to_sample_temp_img)
            sample_subtract_img = np.vstack([sample_subtract_img, temp_subtract_img])
    sample_mean = sample_subtract_img.mean()
    sample_std = sample_subtract_img.std()
    threshold_pixel = float(3) * sample_std + sample_mean

    loc_data = [0]
    for n in tqdm(range(len(path_list) - 1)):
        image1 = path_list[n]
        img1 = cv2.imread(image1)
        image2 = path_list[(int(n) + 1)]
        img2 = cv2.imread(image2)
        image1_gray = rgb_to_gray(img1)
        image2_gray = rgb_to_gray(img2)
        subimage = image1_gray - image2_gray
        blured = cv2.GaussianBlur(subimage, (3, 3), 0)
        count = np.count_nonzero(np.abs(blured) > threshold_pixel)
        loc_data.append(count)
    return loc_data


# calc fluo and make data
def csv_file_read(filepath):
    file_dir, file_name = os.path.split(filepath)
    base, ext = os.path.splitext(file_name)
    if ext == '.csv':
        data = pd.read_csv(filepath)
        return data
    else:
        return messagebox.showinfo('error',
                                   'selected file is not csv file')


def data_extraction(csv_path, max_data_length):
    data = csv_file_read(csv_path)
    dataframe_list = []
    n = 0
    for Track_ID, dataframe in data.groupby('TRACK_ID'):
        if len(dataframe) >= max_data_length and dataframe["FRAME"].min() == 0 \
                and dataframe.iloc[0]["TRACK_ID"] != "None":
            dataframe_list.append(dataframe[0:max_data_length])
            n += 1
        else:
            pass
            n += 1
    if len(dataframe_list) == 0:
        escape = True
    else:
        escape = False
    return dataframe_list, escape


def calc_fluo(dataframe_list, result_data_path, loc_data):
    Fluo_data = []
    only_Fluo_data = []
    data_for_correlation = []
    ID_list = []
    os.makedirs(result_data_path + "/raw_data", exist_ok=True)
    for i in range(len(dataframe_list)):
        temp_dataframe = dataframe_list[i]
        # obtain ID for temp
        ID = temp_dataframe.iloc[0]["TRACK_ID"]
        # add ID to list
        ID_list.append("ID_" + str(ID))
        temp_dataframe["R"] = temp_dataframe["MEAN_INTENSITY_CH1"] / temp_dataframe["MEAN_INTENSITY_CH2"]
        temp_dataframe["R0"] = temp_dataframe["R"].mean()
        temp_dataframe["deltaR_R"] = (temp_dataframe["R"] - temp_dataframe["R0"]) / temp_dataframe["R0"]
        temp_dataframe["locomotion"] = loc_data
        temp_dataframe.to_csv(result_data_path + '/raw_data/Track_ID{}.csv'.format(ID))
        Fluo_data.append(temp_dataframe[["FRAME", "deltaR_R", "TRACK_ID"]])
        only_Fluo_data.append(temp_dataframe[["deltaR_R"]])
        # add fluo data to correlation list
        data_for_correlation.append(np.squeeze(temp_dataframe[["deltaR_R"]].values))
    # make dataframe
    df_for_correlation = pd.DataFrame(np.array(data_for_correlation).T,
                                      columns=ID_list)
    # save all fluo data
    All_fluo_df = pd.DataFrame(np.squeeze(np.asarray(only_Fluo_data).T),
                               columns=ID_list)
    All_fluo_df["time"] = temp_dataframe["POSITION_T"].values
    All_fluo_df["locomotion"] = loc_data
    timeaxis = temp_dataframe["POSITION_T"].values
    All_fluo_df.to_csv(result_data_path + "/all_fluo.csv")
    return Fluo_data, only_Fluo_data, df_for_correlation, \
           timeaxis, All_fluo_df, ID_list


def all_data_visualization(dataframe_list, Fluo_data,
                           max_data_length, timeaxis,
                           result_fig_path):
    Num_of_pages = (len(dataframe_list) - 1) // 20 + 1
    for i in range(Num_of_pages):
        temp_Fluo_data = Fluo_data[20 * i:20 * (i + 1)]
        fig, ax = plt.subplots(4, 5, figsize=(15, 12), sharex=True, sharey=True)
        fig.text(0.5, 0.04, 'Frame', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'deltaR_R', va='center', rotation='vertical', fontsize=20)
        for j, axi in enumerate(ax.flat):
            try:
                axi.plot(np.arange(0, max_data_length),
                         temp_Fluo_data[j]["deltaR_R"], color="black")
                plt.xlim(0, max_data_length)
                plt.ylim(-1, 1)
                axi.text(900, -0.8, "ID: {}" \
                         .format(temp_Fluo_data[j]["TRACK_ID"].values[0]),
                         fontsize=14)
            except IndexError:
                axi.plot(np.arange(0, max_data_length),
                         np.zeros(len(range(max_data_length))), color="black")
                plt.xlim(0, max_data_length)
                plt.ylim(-1, 1)
        plt.savefig(result_fig_path + "/Intensity_graph{}.png".format(i))


def intensity_trace(max_data_length, only_Fluo_data,
                    timeaxis, result_fig_path, ID_list):
    NeuroID = np.array(ID_list)
    # dataprep
    meshdata = np.squeeze(np.array([np.array(i) for i in only_Fluo_data]))
    if len(ID_list) == 1:
        meshdata = meshdata.reshape(1, len(meshdata))
    else:
        pass
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    heatmap = ax1.pcolormesh(timeaxis, NeuroID, meshdata, cmap=plt.cm.jet)
    plt.ylabel("Neuron ID", fontsize=24)
    plt.xlabel("Time (s)", fontsize=24)
    heatmap.set_clim(-0.2, 1)
    fig1.colorbar(heatmap, ax=ax1)
    plt.savefig(result_fig_path + '/Intensity_trace.png')
    plt.show()
    return meshdata, NeuroID


def intensity_histogram(meshdata, result_fig_path):
    fig2, ax2 = plt.subplots(figsize=(18, 8))
    flat_Fluo_data = meshdata.ravel()
    plt.hist(flat_Fluo_data, bins=300, density=True, alpha=0.2,
             histtype='stepfilled', color='r')
    plt.xlim(0, 2)
    plt.xlabel('deltaR_R', fontsize=24)
    plt.ylabel('Relative probability', fontsize=24)
    plt.savefig(result_fig_path + "/Histogram.png")


def draw_heatmap(a, timeaxis, NeuroID, result_fig_path, cmap=plt.cm.jet):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram
    metric = 'correlation'
    method = 'ward'
    fig7, ax7 = plt.subplots(figsize=(10, 8))
    main_axes = ax7
    # for adjust colorbar
    divider = make_axes_locatable(main_axes)
    plt.sca(divider.append_axes("left", 1.0, pad=0))
    ylinkage = linkage(pdist(a, metric=metric), method=method, metric=metric)
    ydendro = dendrogram(ylinkage, orientation='left', no_labels=True,
                         distance_sort='descending',
                         link_color_func=lambda x: 'black')
    a = a.loc[[a.index[i] for i in ydendro['leaves']]]
    plt.gca().set_axis_off()
    plt.sca(main_axes)
    heat = ax7.pcolormesh(timeaxis, NeuroID, a, cmap=plt.cm.jet)
    heat.set_clim(-0.2, 1)
    # imshow(a, aspect='auto', interpolation='none',cmap=cmap, vmin=-0.5, vmax=2.0)
    cbar = fig7.colorbar(heat)
    cbar.set_label("$\Delta$" + "F/F", fontsize=24)
    # ax7.axes.xaxis.set_ticks(timeaxis)
    ax7.axes.yaxis.set_ticks([])
    plt.xlabel("Time (sec)", fontsize=24)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().invert_yaxis()
    plt.savefig(result_fig_path + '/Activity_and_dendrogram.png',
                format='png',
                dpi=300,
                bbox_inches='tight')


def position_of_neurons(dataframe_list, png_path, result_fig_path):
    image_path = png_path
    # image_directory = os.path.dirname(image_path)
    os.makedirs(result_fig_path + "/Neural_positions", exist_ok=True)
    for i in range(len(dataframe_list)):
        img = Image.open(image_path)
        tempdataframe = dataframe_list[i]
        position = tempdataframe.iloc[0]["POSITION_X":"POSITION_Y"]
        ID = tempdataframe.iloc[0]["TRACK_ID"]
        text = "ID{}".format(ID)
        draw = ImageDraw.Draw(img)
        neuron_position = (int(position[0] / correct_value),
                           int(position[1] / correct_value))
        random_value = np.random.randint(4, 10)
        if neuron_position[1] > img.size[1] / 2:
            draw.line(((int(position[0] / correct_value),
                        int(position[1] / correct_value)),
                       (int(position[0] / correct_value),
                        int(position[1] / correct_value) + random_value)),
                      fill=(255, 255, 0))
            draw.text((int(position[0] / correct_value),
                       int(position[1] / correct_value) + random_value),
                      text, font=font, fill='#FFF')
        else:
            draw.line(((int(position[0] / correct_value),
                        int(position[1] / correct_value)),
                       (int(position[0] / correct_value),
                        int(position[1] / correct_value) - random_value)),
                      fill=(255, 255, 0))
            draw.text((int(position[0] / correct_value),
                       int(position[1] / correct_value) - random_value - 11),
                      text, font=font, fill='#FFF')
        img.save(result_fig_path + "/Neural_positions/{0}.png".format(ID),
                 'PNG', quality=300, optimize=True)


def correlation_among_neurons(df_for_correlation, loc_data,
                              result_data_path, result_fig_path):
    import seaborn as sns
    # add locomotion column
    df_for_correlation["Loc"] = loc_data
    df_corr = df_for_correlation.corr()
    fig, ax = plt.subplots(figsize=(25, 20))
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    sns.heatmap(df_corr, square=True, vmax=1, vmin=-1, center=0)
    plt.savefig(result_fig_path + '/Correlation_among_neurons.png')
    df_corr.to_csv(result_data_path + '/df_for_correlation.csv')


def maxisland_start_len_mask(a, fillna_index=-1, fillna_len=0):
    # a is a boolean array
    a = np.array(a)
    a = a.reshape(len(a), 1)
    pad = np.zeros(a.shape[1], dtype=bool)
    mask = np.vstack((pad, a, pad))

    mask_step = mask[1:] != mask[:-1]
    # Return indices that are non-zero in the flattened version of a
    idx = np.flatnonzero(mask_step.T)
    if len(idx) == 0:
        island_starts, island_lens = [], []
        return island_starts, island_lens

    else:
        island_starts = idx[::2]
        island_lens = idx[1::2] - idx[::2]
        n_islands_percol = mask_step.sum(0) // 2

        bins = np.repeat(np.arange(a.shape[1]), n_islands_percol)
        scale = island_lens.max() + 1

        scaled_idx = np.argsort(scale * bins + island_lens)
        grp_shift_idx = np.r_[0, n_islands_percol.cumsum()]
        max_island_starts = island_starts[scaled_idx[grp_shift_idx[1:] - 1]]

        max_island_percol_start = max_island_starts % (a.shape[0] + 1)

        valid = n_islands_percol != 0
        cut_idx = grp_shift_idx[:-1][valid]
        max_island_percol_len = np.maximum.reduceat(island_lens, cut_idx)

        out_len = np.full(a.shape[1], fillna_len, dtype=int)
        out_len[valid] = max_island_percol_len
        out_index = np.where(valid, max_island_percol_start, fillna_index)
        return island_starts, island_lens


def calculate_normalized_fluo(data, tr_frame):
    timewindow = shortest_motion_bout_duration
    time_interval = averaging_window_size
    columns = data.columns
    # get time
    meanlist = []
    num_of_data = int((timewindow) * 2 / time_interval) + 1
    start = data["time"].iloc[0]
    for n in range(num_of_data):
        end = start + time_interval
        data_extraction_mask = np.where((data["time"] >= start) \
                                        & (data["time"] <= end), True, False)
        extracted_data = data[data_extraction_mask]
        meanlist.append(extracted_data.mean())
        start = end
    averaged_df = pd.DataFrame(meanlist, columns=columns)
    return averaged_df


def timeaxis_normalization(df):
    interval = df["time"].iloc[-1] - df["time"].iloc[0]
    sampling_sec = interval / 100
    # get time axis
    meanlist = []
    columns = df.columns
    start = df["time"].iloc[0]
    for n in range(100):
        end = start + sampling_sec
        data_extraction_mask = np.where((df["time"] >= start) \
                                        & (df["time"] <= end), True, False)
        extracted_data = df[data_extraction_mask]
        meanlist.append(extracted_data.mean())
        start = end
    averaged_df = pd.DataFrame(meanlist, columns=columns)
    return averaged_df


def locomotor_active_neuron_analysis(before_df, after_df, during_df, result_data_path, island_start):
    # extract_locomotor_active_neurons
    temp_during_df = during_df.rolling(50, center=True).mean()
    temp_before_df = before_df.rolling(50, center=True).mean()
    temp_after_df = after_df.rolling(50, center=True).mean()
    max_df = temp_during_df.max()
    mean_df_before = temp_before_df.mean()
    mean_df_after = temp_after_df.mean()
    baseline_fluo_intensity = pd.concat([mean_df_before, mean_df_after], axis=1).T.max()
    max_differences = max_df - baseline_fluo_intensity
    # indices which neural ID above threshold
    indices_list = list(max_differences[max_differences > neural_activity_threshold].index)
    indices_list = [i for i in indices_list if re.match("ID_\d+", i)]
    data_list = []
    data_list.append(during_df["time"])
    data_list.append(during_df["locomotion"])
    if len(indices_list) == 0:
        pass
    else:
        for ix in indices_list:
            data_list.append(during_df[ix])
    locomotor_active_neuron_df = pd.DataFrame(data_list).T
    locomotor_active_neuron_df.to_csv(result_data_path + "./locomotor_active_df_{}.csv".format(island_start))

    # peak to MQ transition
    columns = during_df.columns.values
    peak_to_MQ = during_df["time"].iloc[-1] - during_df["time"][during_df.idxmax()]
    peak_analysis = np.vstack([columns,
                               peak_to_MQ.values]).T
    peak_analysis = pd.DataFrame(peak_analysis, columns=["ID",
                                                         "peak_to_MQ"]).T
    peak_analysis.to_csv(result_data_path + "./locomotor_active_peak_to_MQ{}.csv".format(island_start))

    # peak to MQ transition ver2
    # search peak among 12 sec before transition
    columns = during_df.columns.values
    end_12sec_df = during_df[during_df["time"] > (during_df["time"].iloc[-1] - 12)]
    peak_to_MQ = end_12sec_df["time"].iloc[-1] - end_12sec_df["time"][end_12sec_df.idxmax()]
    peak_analysis = np.vstack([columns,
                               peak_to_MQ.values]).T
    peak_analysis = pd.DataFrame(peak_analysis, columns=["ID",
                                                         "peak_to_MQ"]).T
    peak_analysis.to_csv(result_data_path + "./locomotor_active_peak_to_MQ_12sec{}.csv".format(island_start))


def motion_bout_detector(All_fluo_data, result_data_path,
                         result_fig_path):
    data = All_fluo_data
    # isolate time axis and motion data
    timeaxis = data["time"].values
    locomotion_data = data["locomotion"].values

    tempstart = 0
    motion = np.zeros(len(timeaxis))
    # i: index of timepoints when the worms locomotion is
    for i in np.where(locomotion_data > threshold)[0]:
        # time from previous active point
        timeduration = float(timeaxis[i] - timeaxis[tempstart])
        if timeduration < interval_between_bouts:
            motion[tempstart:i] = 1
        else:
            pass
        tempstart = i
    island_starts, island_lens = maxisland_start_len_mask(motion)

    # peak analysis (20211122 renew)
    MQ_tr_timepoints = []
    for j in range(len(island_starts)):
        MQ_tr_timepoints.append(island_starts[j] + island_lens[j])
    MQ_tr_timepoints_df = pd.DataFrame(timeaxis[MQ_tr_timepoints])
    MQ_tr_timepoints_df.to_csv(result_data_path + "/MQtr_timepoints.csv")

    columns = data.columns.values
    for k in range(len(columns)):
        temp_data = data[columns[k]]
        smth_data = temp_data.rolling(51, center=True, min_periods=2).mean()
        arg_r_max = argrelmax(smth_data.values, order=400)
        if k == 0:
            argmax_data = pd.Series(timeaxis[arg_r_max[0]], name=columns[k])
        else:
            temp_data = pd.Series(timeaxis[arg_r_max[0]], name=columns[k])
            argmax_data = pd.concat([argmax_data, temp_data], axis=1)
    argmax_data.to_csv(result_data_path + "/argmax.csv")

    # peak analysis (20211123 renew)
    columns = data.columns.values
    for k in range(len(columns)):
        temp_data = data[columns[k]]
        smth_data = temp_data.rolling(50, center=True, min_periods=2).mean()
        peaks, _ = find_peaks(smth_data, prominence=(0.02, None))
        if k == 0:
            argmax_data = pd.Series(timeaxis[peaks], name=columns[k])
        else:
            temp_data = pd.Series(timeaxis[peaks], name=columns[k])
            argmax_data = pd.concat([argmax_data, temp_data], axis=1)
    argmax_data.to_csv(result_data_path + "/argmax_findpeak.csv")

    delete_list = []
    for j in range(len(island_starts)):
        start_m_bout = island_starts[j]
        end_m_bout = island_starts[j] + island_lens[j]
        duration = timeaxis[end_m_bout] - timeaxis[start_m_bout]
        if duration < shortest_motion_bout_duration:
            delete_list.append(j)
        elif (timeaxis[start_m_bout] < interval_between_bouts) & \
                (timeaxis[-1] - timeaxis[end_m_bout] > interval_between_bouts):
            os.makedirs(result_data_path + "/only_MQtr", exist_ok=True)
            extracted_end_time = timeaxis[end_m_bout] + shortest_motion_bout_duration
            extract_index = np.where(timeaxis < extracted_end_time)
            extracted_data = data.iloc[extract_index]
            extracted_data.to_csv(result_data_path + "/only_MQtr/alldata.csv")
            # extract data during motion bout
            extract_end_time_only_motion = timeaxis[end_m_bout]
            extract_index_motion_bout = np.where(timeaxis < extract_end_time_only_motion)
            extracted_motion_bout = data.iloc[extract_index_motion_bout]
            extracted_motion_bout.to_csv(result_data_path + "/only_MQtr/motion_bout_data.csv")
            # extract data from 5 sec before motion bouts end
            start_of_end_of_motion = timeaxis[end_m_bout] - data_extract_duration_for_m_av_analysis
            extract_index_end_of_motion = np.where((start_of_end_of_motion < timeaxis) & \
                                                   (timeaxis < extract_end_time_only_motion))
            extracted_end_of_motion = data.iloc[extract_index_end_of_motion]
            extracted_end_of_motion.to_csv(result_data_path + "/only_MQtr/end_of_motion_data.csv")
            # MQtr
            timeaxis_for_av = np.arange(0, shortest_motion_bout_duration * 2 + averaging_window_size,
                                        averaging_window_size) \
                              - shortest_motion_bout_duration
            MQ_tr_index = np.where((timeaxis[end_m_bout] - shortest_motion_bout_duration < timeaxis) \
                                   & (timeaxis[end_m_bout] + shortest_motion_bout_duration > timeaxis))
            MQ_tr_data = data.iloc[MQ_tr_index]
            MQ_tr_data.to_csv(result_data_path + "/only_MQtr/MQtr.csv")
            tr_frame = MQ_tr_data[MQ_tr_data["time"] == timeaxis[end_m_bout]].index
            average_df = calculate_normalized_fluo(MQ_tr_data, tr_frame)
            average_df["time_cor"] = timeaxis_for_av
            average_df.to_csv(result_data_path + "/only_MQtr/averaged_MQ.csv")
            # peak to MQ
            columns = extracted_data.columns.values
            peak_to_MQ = timeaxis[end_m_bout] - extracted_data["time"][extracted_data.idxmax()]
            peak_analysis = np.vstack([columns,
                                       peak_to_MQ.values]).T
            peak_analysis = pd.DataFrame(peak_analysis, columns=["ID",
                                                                 "peak_to_MQ"]).T
            peak_analysis.to_csv(result_data_path + "/only_MQtr/peak_to_MQ.csv")

            # peak to MQ (12sec)
            columns = extracted_data.columns.values
            end_12sec_df = extracted_data[extracted_data["time"] > (extracted_data["time"].iloc[-1] - 12)]
            peak_to_MQ = end_12sec_df["time"].iloc[-1] - end_12sec_df["time"][end_12sec_df.idxmax()]
            peak_analysis = np.vstack([columns,
                                       peak_to_MQ.values]).T
            peak_analysis = pd.DataFrame(peak_analysis, columns=["ID",
                                                                 "peak_to_MQ"]).T
            peak_analysis.to_csv(result_data_path + "/only_MQtr/peak_to_MQ_12sec.csv")

            delete_list.append(j)
        elif (timeaxis[end_m_bout] > timeaxis[-1] - interval_between_bouts) & \
                (timeaxis[start_m_bout] - timeaxis[0] > interval_between_bouts):
            os.makedirs(result_data_path + "/only_QMtr",
                        exist_ok=True)
            extracted_start_time = timeaxis[start_m_bout] - shortest_motion_bout_duration
            extract_index = np.where(timeaxis > extracted_start_time)
            extracted_data = data.iloc[extract_index]
            extracted_data.to_csv(result_data_path + "/only_QMtr/data.csv")
            timeaxis_for_av = np.arange(0, shortest_motion_bout_duration * 2 + averaging_window_size,
                                        averaging_window_size) - shortest_motion_bout_duration
            # extract data during motion bout
            extracted_start_time_only_motion = timeaxis[start_m_bout]
            extract_index_motion_bout = np.where(timeaxis > extracted_start_time_only_motion)
            extracted_motion_bout = data.iloc[extract_index_motion_bout]
            extracted_motion_bout.to_csv(result_data_path + "/only_QMtr/motion_bout_data.csv")
            # QM tr
            QM_tr_index = np.where((timeaxis[start_m_bout] - shortest_motion_bout_duration < timeaxis) \
                                   & (timeaxis[start_m_bout] + shortest_motion_bout_duration > timeaxis))
            QM_tr_data = data.iloc[QM_tr_index]
            QM_tr_data.to_csv(result_data_path + "/only_QMtr/QMtr.csv")
            tr_frame = QM_tr_data[QM_tr_data["time"] == timeaxis[start_m_bout]].index
            average_df = calculate_normalized_fluo(QM_tr_data, tr_frame)
            average_df["time_cor"] = timeaxis_for_av
            average_df.to_csv(result_data_path + "/only_QMtr/averaged_QM.csv")
            delete_list.append(j)
        elif (timeaxis[start_m_bout] < shortest_motion_bout_duration) & \
                (timeaxis[-1] - timeaxis[end_m_bout] < shortest_motion_bout_duration):
            delete_list.append(j)
        elif (timeaxis[end_m_bout] > timeaxis[-1] - shortest_motion_bout_duration) & \
                (timeaxis[start_m_bout] - timeaxis[0] < shortest_motion_bout_duration):
            delete_list.append(j)
        else:
            pass
    island_starts = np.delete(island_starts, delete_list)
    island_lens = np.delete(island_lens, delete_list)
    for k in range(len(island_starts)):
        island_start = island_starts[k]
        island_duration = island_lens[k]
        island_end = island_start + island_duration
        extract_start_time = timeaxis[island_start] - interval_between_bouts
        extract_end_time = timeaxis[island_start + island_duration] + interval_between_bouts
        extract_indices = np.where((timeaxis > extract_start_time) \
                                   & (timeaxis < extract_end_time))
        extracted_data = data.iloc[extract_indices]
        # extracted_df = pd.DataFrame(extracted_data, columns = ["time",
        # "locomotion",
        # "Fluo"])
        before_indices = np.where((timeaxis > extract_start_time) \
                                  & (timeaxis < timeaxis[island_start]))
        before = data.iloc[before_indices]
        after_indices = np.where((timeaxis > timeaxis[island_end]) \
                                 & (timeaxis < extract_end_time))
        after = data.iloc[after_indices]
        during_indices = np.where((timeaxis > timeaxis[island_start]) \
                                  & (timeaxis < timeaxis[island_end]))
        during = data.iloc[during_indices]
        locomotor_active_neuron_analysis(before, after, during, result_data_path, island_start)

        # peak to transition analysis
        columns = during.columns.values
        QM_to_peak = during["time"][during.idxmax()] - during["time"].iloc[0]
        peak_to_MQ = during["time"].iloc[-1] - during["time"][during.idxmax()]
        peak_analysis = np.vstack([columns,
                                   QM_to_peak.values,
                                   peak_to_MQ.values]).T
        peak_analysis = pd.DataFrame(peak_analysis, columns=["ID",
                                                             "QM_to_peak",
                                                             "peak_to_MQ"]).T
        # peak to transition analysis (averaging)
        columns = during.columns.values
        during_smth = during.rolling(50, center=True, min_periods=2).mean()
        QM_to_peak = during_smth["time"][during_smth.idxmax()] - during_smth["time"].iloc[0]
        peak_to_MQ = during_smth["time"].iloc[-1] - during_smth["time"][during_smth.idxmax()]
        peak_analysis_smth = np.vstack([columns,
                                        QM_to_peak.values,
                                        peak_to_MQ.values]).T
        peak_analysis_smth = pd.DataFrame(peak_analysis_smth, columns=["ID",
                                                                       "QM_to_peak",
                                                                       "peak_to_MQ"]).T

        # peak to transition analysis(12sec)
        columns = during.columns.values
        end_12sec_df = during[during["time"] > (during["time"].iloc[-1] - 12)]
        peak_to_MQ = end_12sec_df["time"].iloc[-1] - end_12sec_df["time"][end_12sec_df.idxmax()]
        peak_analysis_12sec = np.vstack([columns,
                                         peak_to_MQ.values]).T
        peak_analysis_12sec = pd.DataFrame(peak_analysis_12sec, columns=["ID",
                                                                         "peak_to_MQ"]).T

        # resample and make averaged data
        resampled = timeaxis_normalization(during)

        timeaxis_for_av = np.arange(0, shortest_motion_bout_duration * 2 + averaging_window_size,
                                    averaging_window_size) - shortest_motion_bout_duration
        # QM tr
        QM_tr_index = np.where((timeaxis[island_start] - shortest_motion_bout_duration < timeaxis) \
                               & (timeaxis[island_start] + shortest_motion_bout_duration > timeaxis))
        QM_tr_data = data.iloc[QM_tr_index]
        QM_tr_time = timeaxis[island_start]
        tr_frame = QM_tr_data[QM_tr_data["time"] == QM_tr_time].index
        average_df_QM = calculate_normalized_fluo(QM_tr_data, tr_frame)
        average_df_QM["time_cor"] = timeaxis_for_av

        # MQ tr
        MQ_tr_index = np.where((timeaxis[island_end] - shortest_motion_bout_duration < timeaxis) \
                               & (timeaxis[island_end] + shortest_motion_bout_duration > timeaxis))
        MQ_tr_data = data.iloc[MQ_tr_index]
        MQ_tr_time = timeaxis[island_end]
        tr_frame = MQ_tr_data[MQ_tr_data["time"] == MQ_tr_time].index
        average_df_MQ = calculate_normalized_fluo(MQ_tr_data, tr_frame)
        average_df_MQ["time_cor"] = timeaxis_for_av
        os.makedirs(result_data_path + "/island{}".format(island_start), exist_ok=True)

        extracted_data.to_csv(result_data_path + "/island{}/island_all.csv".format(island_start))
        before.to_csv(result_data_path + "/island{}/before.csv".format(island_start))
        after.to_csv(result_data_path + "/island{}/after.csv".format(island_start))
        during.to_csv(result_data_path + "/island{}/during.csv".format(island_start))
        QM_tr_data.to_csv(result_data_path + "/island{}/QMtr.csv".format(island_start))
        MQ_tr_data.to_csv(result_data_path + "/island{}/MQtr.csv".format(island_start))
        average_df_MQ.to_csv(result_data_path + "/island{}/averaged_MQ.csv".format(island_start))
        average_df_QM.to_csv(result_data_path + "/island{}/averaged_QM.csv".format(island_start))
        peak_analysis.to_csv(result_data_path + \
                             "/island{}/peak_analysis.csv".format(island_start))
        peak_analysis_12sec.to_csv(result_data_path + \
                                   "/island{}/peak_analysis_12sec.csv".format(island_start))
        peak_analysis_smth.to_csv(result_data_path + \
                                  "/island{}/peak_analysis_smth.csv".format(island_start))
        resampled.to_csv(result_data_path + "/island{}/resampled.csv". \
                         format(island_start))


def execute_analysis(filepath, result_data_path,
                     result_fig_path, loc_data,
                     png_path):
    # get file path
    filename = os.path.splitext(os.path.basename(filepath))[0]
    print("processing: {}".format(filename))
    # data extraction
    dataframe_list, escape = data_extraction(filepath, max_data_length)
    if escape == False:
        Fluo_data, only_Fluo_data, df_for_correlation, \
        timeaxis, All_fluo_df, ID_list = calc_fluo(dataframe_list,
                                                   result_data_path,
                                                   loc_data)
        all_data_visualization(dataframe_list, Fluo_data,
                               max_data_length, timeaxis,
                               result_fig_path)

        meshdata, NeuroID = intensity_trace(max_data_length,
                                            only_Fluo_data,
                                            timeaxis,
                                            result_fig_path, ID_list)
        heatmap_data = pd.DataFrame(meshdata,
                                    index=np.linspace(0, len(dataframe_list) - 1,
                                                      len(dataframe_list)),
                                    columns=timeaxis)
        intensity_histogram(meshdata, result_fig_path)
        if len(ID_list) == 1:
            pass
        else:
            draw_heatmap(heatmap_data, timeaxis, NeuroID, result_fig_path,
                         cmap=plt.cm.jet)
        position_of_neurons(dataframe_list, png_path, result_fig_path)
        correlation_among_neurons(df_for_correlation, loc_data,
                                  result_data_path, result_fig_path)
        motion_bout_detector(All_fluo_df, result_data_path,
                             result_fig_path)
    else:
        print("not adequate data length")


def main():
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showinfo('select dir', 'select dir')
    path = filedialog.askdirectory()
    files = os.listdir(path)
    all_dir_names = [f for f in files if os.path.isdir(os.path.join(path, f))]
    dir_names = [s for s in all_dir_names if re.match('Experiment-\d+', s)]
    for folderpath in dir_names:
        os.chdir(path)
        Ex_name, csv_path, png_path, image_seq_list = dir_select(folderpath)
        result_data_path, result_fig_path = makedirs(Ex_name)
        loc_data = image_subtraction_analysis(image_seq_list)
        execute_analysis(csv_path, result_data_path, result_fig_path, loc_data,
                         png_path)


if __name__ == '__main__':
    main()
