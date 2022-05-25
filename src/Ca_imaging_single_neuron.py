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

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams["figure.dpi"] = 300

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


def Calculate_Fluorescence(GFP, RFP):
    mask = []
    tracking_judge = []
    area = []
    for i in range(len(RFP)):
        temp_img = RFP[i]
        threshold_for_mask_img = threshold_otsu(temp_img)
        mask_img = temp_img >= threshold_for_mask_img
        area.append(mask_img.sum())
        if (mask_img.sum() < tracking_area_lower_threshold) or (mask_img.sum() > tracking_area_upper_threshold):
            tracking_judge.append(0)
        else:
            tracking_judge.append(1)
        mask.append(mask_img)
    mask = np.array(mask)
    tracking_judge = np.array(tracking_judge)
    # calc_masked_data
    masked_GFP = GFP * mask
    masked_RFP = RFP * mask
    fluorescent_ratio = masked_GFP / masked_RFP
    fluorescent_ratio = np.nan_to_num(fluorescent_ratio)
    R_data = []
    meanGFP = []
    meanRFP = []
    for i in range(len(fluorescent_ratio)):
        temp_img = fluorescent_ratio[i]
        temp_sum = np.sum(temp_img)
        temp_count = np.count_nonzero(temp_img)
        temp_mean = temp_sum / temp_count
        temp_meanGFP = np.sum(masked_GFP[i]) / temp_count
        temp_meanRFP = np.sum(masked_RFP[i]) / temp_count
        R_data.append(temp_mean)
        meanGFP.append(temp_meanGFP)
        meanRFP.append(temp_meanRFP)
    R_data = np.array(R_data)
    meanGFP = np.array(meanGFP)
    meanRFP = np.array(meanRFP)

    # calc mean R
    mean_R = R_data.mean()

    # calc deltaR/R
    dR_R = (R_data - mean_R) / mean_R
    dR_R = dR_R * tracking_judge
    return dR_R, meanGFP, meanRFP, area, tracking_judge


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


def detect_motion_or_sleep_timepoints(timeaxis, locomotion_data):
    # detect motion or not based on threshold pixel value
    tempstart = 0
    Motion_or_Sleep_timepoints = np.zeros(len(timeaxis))
    # i: index of timepoints when the worms locomotion is above threshold
    for i in np.where(locomotion_data > motion_pixel_threshold)[0]:
        # time from previous active point
        timeduration = float(timeaxis[i] - timeaxis[tempstart])
        if timeduration < interval_between_bouts:
            Motion_or_Sleep_timepoints[tempstart:i] = 1
        else:
            pass
        tempstart = i
    return Motion_or_Sleep_timepoints


def find_peak_during_motion_bouts(data, timeaxis):
    columns = data.columns.values
    for k in range(len(columns)):
        temp_data = data[columns[k]]
        smth_data = temp_data.rolling(50, center=True, min_periods=2).mean()
        peaks, _ = find_peaks(smth_data, prominence=(0.02, None))
        if k == 0:
            peak_data = pd.Series(timeaxis[peaks], name=columns[k])
        else:
            temp_data = pd.Series(timeaxis[peaks], name=columns[k])
            peak_data = pd.concat([peak_data, temp_data], axis=1)
    peak_data.to_csv("./argmax_findpeak.csv")
    return peak_data


def detect_exclusion_bouts(data, bouts_starts, bouts_lengths, timeaxis):
    delete_list = []
    for j in range(len(bouts_starts)):
        bout_start_idx = bouts_starts[j]
        bout_end_idx = bouts_starts[j] + bouts_lengths[j] - 1
        bout_duration = timeaxis[bout_end_idx] - timeaxis[bout_start_idx]
        # too short bout
        if bout_duration < shortest_bout_duration:
            delete_list.append(j)
        # too early start bout, but end during imaging session
        elif (timeaxis[bout_start_idx] < interval_between_bouts) & \
                (timeaxis[-1] - timeaxis[bout_end_idx] > interval_between_bouts):
            os.makedirs("./only_MStr", exist_ok=True)

            # extract bout + following 6sec
            extracted_time_end_bout_and_following = timeaxis[bout_end_idx] + shortest_bout_duration
            extract_index = np.where(timeaxis < extracted_time_end_bout_and_following)
            bout_and_follwing = data.iloc[extract_index]
            bout_and_follwing.to_csv("./only_MStr/bout_and_following_6sec.csv")

            # extract only bout
            extract_time_end_only_bout = timeaxis[bout_end_idx]
            extract_index_motion_bout = np.where(timeaxis < extract_time_end_only_bout)
            extracted_motion_bout = data.iloc[extract_index_motion_bout]
            extracted_motion_bout.to_csv("./only_MStr/only_bout.csv")

            # transition analysis
            tr_idx = np.where((timeaxis[bout_end_idx] - shortest_bout_duration < timeaxis)
                              & (timeaxis[bout_end_idx] + shortest_bout_duration > timeaxis))
            tr_data = data.iloc[tr_idx]
            tr_data.to_csv("./only_MStr/MStr.csv")
            delete_list.append(j)

        elif (timeaxis[bout_end_idx] > timeaxis[-1] - interval_between_bouts) & \
                (timeaxis[bout_start_idx] - timeaxis[0] > interval_between_bouts):
            os.makedirs("./only_SMtr", exist_ok=True)

            # extract bout + before 6sec
            extracted_start_time = timeaxis[bout_start_idx] - shortest_bout_duration
            extract_index = np.where(timeaxis > extracted_start_time)
            bout_and_follwing = data.iloc[extract_index]
            bout_and_follwing.to_csv("./only_SMtr/data.csv")

            # extract data during motion bout
            extracted_start_time_only_motion = timeaxis[bout_start_idx]
            extract_index_motion_bout = np.where(timeaxis > extracted_start_time_only_motion)
            extracted_motion_bout = data.iloc[extract_index_motion_bout]
            extracted_motion_bout.to_csv("./only_SMtr/motion_bout_data.csv")

            # QM tr
            QM_tr_index = np.where((timeaxis[bout_start_idx] - shortest_bout_duration < timeaxis)
                                   & (timeaxis[bout_start_idx] + shortest_bout_duration > timeaxis))
            QM_tr_data = data.iloc[QM_tr_index]
            QM_tr_data.to_csv("./only_SMtr/SMtr.csv")
            delete_list.append(j)

        # constitute motion or sleep
        elif (timeaxis[bout_start_idx] < shortest_bout_duration) & \
                (timeaxis[-1] - timeaxis[bout_end_idx] < shortest_bout_duration):
            delete_list.append(j)

        else:
            pass
    return delete_list


def detect_exclusion_bouts_for_sleep(data, bouts_starts, bouts_lengths, timeaxis):
    delete_list = []
    for j in range(len(bouts_starts)):
        bout_start_idx = bouts_starts[j]
        bout_end_idx = bouts_starts[j] + bouts_lengths[j] - 1
        bout_duration = timeaxis[bout_end_idx] - timeaxis[bout_start_idx]
        # too short bout
        if bout_duration < shortest_bout_duration:
            delete_list.append(j)
        # too early start bout, but end during imaging session
        elif (timeaxis[bout_start_idx] < interval_between_bouts) & \
                (timeaxis[-1] - timeaxis[bout_end_idx] > interval_between_bouts):
            delete_list.append(j)
        # too late end bout, but start during imaging session
        elif (timeaxis[bout_end_idx] > timeaxis[-1] - interval_between_bouts) & \
                (timeaxis[bout_start_idx] - timeaxis[0] > interval_between_bouts):
            delete_list.append(j)
        # constitute motion or sleep
        elif (timeaxis[bout_start_idx] < shortest_bout_duration) & \
                (timeaxis[-1] - timeaxis[bout_end_idx] < shortest_bout_duration):
            delete_list.append(j)
        else:
            pass
    return delete_list


def Motion_island_analysis(data, bouts_starts, bouts_lengths, timeaxis):
    for k in range(len(bouts_starts)):
        island_start_idx = bouts_starts[k]
        island_duration_idx = bouts_lengths[k]
        island_end_idx = island_start_idx + island_duration_idx - 1

        # extract with before and following 6sec
        extract_start_time = timeaxis[island_start_idx] - interval_between_bouts
        extract_end_time = timeaxis[island_start_idx + island_duration_idx] + interval_between_bouts
        extract_indices = np.where((timeaxis > extract_start_time)
                                   & (timeaxis < extract_end_time))
        extracted_data = data.iloc[extract_indices]

        # extract before island
        before_indices = np.where((timeaxis > extract_start_time)
                                  & (timeaxis < timeaxis[island_start_idx]))
        before = data.iloc[before_indices]
        # extract after island
        after_indices = np.where((timeaxis > timeaxis[island_end_idx])
                                 & (timeaxis < extract_end_time))
        after = data.iloc[after_indices]
        # extract only during island
        during_indices = np.where((timeaxis > timeaxis[island_start_idx])
                                  & (timeaxis < timeaxis[island_end_idx]))
        during = data.iloc[during_indices]
        # Extraction before and during island data
        before_and_during_island = data.iloc[
            np.where((timeaxis > extract_start_time) & (timeaxis < timeaxis[island_end_idx]))]

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

        # MQ tr
        MS_tr_index = np.where((timeaxis[island_end_idx] - shortest_bout_duration < timeaxis)
                               & (timeaxis[island_end_idx] + shortest_bout_duration > timeaxis))
        MS_tr_data = data.iloc[MS_tr_index]

        # QM tr
        SM_tr_index = np.where((timeaxis[island_start_idx] - shortest_bout_duration < timeaxis)
                               & (timeaxis[island_start_idx] + shortest_bout_duration > timeaxis))
        SM_tr_data = data.iloc[SM_tr_index]

        os.makedirs("./island{}".format(island_start_idx), exist_ok=True)
        extracted_data.to_csv("./island{}/before_and_bout_and_after.csv".format(island_start_idx))
        before_and_during_island.to_csv("./island{}/before_and_bout.csv".format(island_start_idx))
        before.to_csv("./island{}/before.csv".format(island_start_idx))
        after.to_csv("./island{}/after.csv".format(island_start_idx))
        during.to_csv("./island{}/bout.csv".format(island_start_idx))
        SM_tr_data.to_csv("./island{}/SMtr.csv".format(island_start_idx))
        MS_tr_data.to_csv("./island{}/MStr.csv".format(island_start_idx))
        peak_analysis.to_csv("./island{}/peak_analysis.csv".format(island_start_idx))
        peak_analysis_12sec.to_csv("./island{}/peak_analysis_12sec.csv".format(island_start_idx))
        peak_analysis_smth.to_csv("./island{}/peak_analysis_smth.csv".format(island_start_idx))


def Sleep_island_analysis(data, bouts_starts, bouts_lengths, timeaxis):
    for k in range(len(bouts_starts)):
        island_start_idx = bouts_starts[k]
        island_duration_idx = bouts_lengths[k]
        island_end_idx = island_start_idx + island_duration_idx - 1

        # extract with before and following 6sec
        extract_start_time = timeaxis[island_start_idx] - interval_between_bouts
        extract_end_time = timeaxis[island_start_idx + island_duration_idx] + interval_between_bouts
        extract_indices = np.where((timeaxis > extract_start_time)
                                   & (timeaxis < extract_end_time))
        extracted_data = data.iloc[extract_indices]

        # extract before island
        before_indices = np.where((timeaxis > extract_start_time)
                                  & (timeaxis < timeaxis[island_start_idx]))
        before = data.iloc[before_indices]
        # extract after island
        after_indices = np.where((timeaxis > timeaxis[island_end_idx])
                                 & (timeaxis < extract_end_time))
        after = data.iloc[after_indices]
        # extract only during island
        during_indices = np.where((timeaxis > timeaxis[island_start_idx])
                                  & (timeaxis < timeaxis[island_end_idx]))
        during = data.iloc[during_indices]
        # Extraction before and during island data
        before_and_during_island = data.iloc[
            np.where((timeaxis > extract_start_time) & (timeaxis < timeaxis[island_end_idx]))]

        os.makedirs("./Sleep_island{}".format(island_start_idx), exist_ok=True)
        extracted_data.to_csv("./Sleep_island{}/before_and_bout_and_after.csv".format(island_start_idx))
        before_and_during_island.to_csv("./Sleep_island{}/before_and_bout.csv".format(island_start_idx))
        before.to_csv("./Sleep_island{}/before.csv".format(island_start_idx))
        after.to_csv("./Sleep_island{}/after.csv".format(island_start_idx))
        during.to_csv("./Sleep_island{}/bout.csv".format(island_start_idx))


def motion_bout_detector(data):
    # isolate time axis and motion data
    timeaxis = data["time"].values
    locomotion_data = data["locomotion"].values
    Motion_or_Sleep_timepoints = detect_motion_or_sleep_timepoints(timeaxis, locomotion_data)
    Sleep_or_Motion_timepoints = np.logical_not(Motion_or_Sleep_timepoints)

    # starts and length analysis for MOTION bouts
    Motion_bouts_starts, Motion_bouts_lengths = maxisland_start_len_mask(Motion_or_Sleep_timepoints)
    # starts and length analysis for SLEEP bouts
    Sleep_bouts_starts, Sleep_bouts_lengths = maxisland_start_len_mask(Sleep_or_Motion_timepoints)

    # save dataframes
    pd.DataFrame(Motion_or_Sleep_timepoints).to_csv("./Motion_or_Sleep_timepoints.csv")
    pd.DataFrame(Sleep_or_Motion_timepoints).to_csv("./Sleep_or_Motion_timepoints.csv")
    pd.DataFrame(np.array([Motion_bouts_starts, Motion_bouts_lengths]).T, columns=["Starts", " Length"]). \
        to_csv("./Motion_bouts_information.csv")
    pd.DataFrame(np.array([Sleep_bouts_starts, Sleep_bouts_lengths]).T, columns=["Starts", " Length"]). \
        to_csv("./Sleep_bouts_information.csv")

    # find peak analysis
    peak_data = find_peak_during_motion_bouts(data, timeaxis)

    # judgement for delete from list
    delete_list = detect_exclusion_bouts(data, Motion_bouts_starts, Motion_bouts_lengths, timeaxis)
    Motion_bouts_starts = np.delete(Motion_bouts_starts, delete_list)
    Motion_bouts_lengths = np.delete(Motion_bouts_lengths, delete_list)
    # island analysis for motion bouts
    Motion_island_analysis(data, Motion_bouts_starts, Motion_bouts_lengths, timeaxis)

    delete_list_for_sleep = detect_exclusion_bouts_for_sleep(data, Sleep_bouts_starts, Sleep_bouts_lengths, timeaxis)
    Sleep_bouts_starts = np.delete(Sleep_bouts_starts, delete_list_for_sleep)
    Sleep_bouts_lengths = np.delete(Sleep_bouts_lengths, delete_list_for_sleep)
    Sleep_island_analysis(data, Sleep_bouts_starts, Sleep_bouts_lengths, timeaxis)


def Calculate_Locomotion(DIC):
    # parameter
    # threshold 1300 for 16bit data
    # 5.5 for 8bit data
    threshold_px_val = 1300
    rol_DIC = np.roll(DIC, -1, axis=0)
    subtracted_DIC = DIC - rol_DIC
    subtracted_DIC = subtracted_DIC[:-1]
    # mean_val = subtracted_DIC.mean()
    # std_val = subtracted_DIC.std()
    # threshold_px = mean_val + 3 * std_val
    subtract_result = []
    for i in range(len(subtracted_DIC)):
        temp_img = subtracted_DIC[i]
        temp_img = gaussian_filter(temp_img, sigma=1)
        subtract_result.append(np.count_nonzero(np.abs(temp_img) > threshold_px_val))
    subtract_result = np.insert(np.array(subtract_result), 0, 0)
    return subtract_result


def draw_graphs(result, name):
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    ax2 = ax1.twinx()
    ax1.plot(result["dR_R"], color="darkred")
    ax2.plot(result["locomotion"], color="black")
    ax1.set_ylim([-1, 1])
    ax2.set_ylim([-100, 5000])
    plt.savefig("./figs/{}.png".format(name))


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
    os.makedirs("./figs", exist_ok=True)
    for i in range(len(czi_files)):
        Ex_num = name_list[i].split("-")[1]
        date = datetime.date.today()
        os.makedirs('./Ex-{0}_date_{1}'.format(Ex_num, date), exist_ok=True)
        GFP, RFP, DIC = extract_images(czi_files[i])
        dR_R, meanGFP, meanRFP, area, tracking_judge = Calculate_Fluorescence(GFP, RFP)
        subtract_result = Calculate_Locomotion(DIC)
        timeaxis = np.arange(0, len(dR_R) / 10, 0.1)
        result = pd.DataFrame(np.vstack([timeaxis, subtract_result, dR_R, meanGFP, meanRFP, area, tracking_judge]).T,
                              columns=["time", "locomotion", "dR_R", "GFP", "RFP", "area", "tracking_judge"])
        draw_graphs(result, name_list[i])
        os.chdir('./Ex-{0}_date_{1}'.format(Ex_num, date))
        result.to_csv("./{}.csv".format(name_list[i]))
        motion_bout_detector(result)
        os.chdir("../")


if __name__ == '__main__':
    main()
