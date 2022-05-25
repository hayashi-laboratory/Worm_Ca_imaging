"""
This Script extract data from
"""

import os
import pandas as pd
import numpy as np

from tkinter import filedialog
import tkinter
import os.path
import re


def select_dir():
    root = tkinter.Tk()
    root.withdraw()
    target_dir = filedialog.askdirectory()
    os.chdir(target_dir)
    os.makedirs("./extracted_data/Motion", exist_ok=True)
    os.makedirs("./extracted_data/Sleep", exist_ok=True)
    return target_dir


def extract_before_and_bout(target_dir):
    before_and_bout_path = []
    # get path
    for curDir, dirs, files in os.walk(target_dir):
        for file in files:
            if re.match("before_and_bout.csv", file):
                before_and_bout_path.append(curDir + "/" + file)
        if not before_and_bout_path:
            continue
    M_index_list = [[], [], [], [], []]
    S_index_list = [[], [], [], [], []]
    for i in range(len(before_and_bout_path)):
        temp_path = os.path.abspath(before_and_bout_path[i])
        temp_data = pd.read_csv(temp_path)
        split_path = [j for j in temp_path.split("\\")]
        # Motion bout
        if not [j for j in split_path if re.match("Sleep_island\d+", j)]:
            Experiment_date = [j for j in split_path if re.match("ExDate\d+", j)][0]
            M_index_list[0].append(Experiment_date)
            worm_num = [j for j in split_path if re.match("worm\d+", j)][0]
            M_index_list[1].append(worm_num)
            Experiment_num = [j for j in split_path if re.match("Ex-\d+", j)][0].split("_")[0]
            M_index_list[2].append(Experiment_num)
            island_num = [j for j in split_path if re.match("island\d+", j)][0]
            M_index_list[3].append(island_num)
            all_index = Experiment_date + worm_num + Experiment_num + island_num
            M_index_list[4].append(all_index)
            if len(M_index_list[0])<2:
                M_bout_loc_data = temp_data["locomotion"]
                M_bout_dR_R_data = temp_data["dR_R"]
                M_bout_tracking_judge = temp_data["tracking_judge"]
            else:
                M_bout_loc_data = pd.concat([M_bout_loc_data, temp_data["locomotion"]], axis=1)
                M_bout_dR_R_data = pd.concat([M_bout_dR_R_data, temp_data["dR_R"]], axis=1)
                M_bout_tracking_judge = pd.concat([M_bout_tracking_judge, temp_data["tracking_judge"]], axis=1)

        # Sleep
        else:
            Experiment_date = [j for j in split_path if re.match("ExDate\d+", j)][0]
            S_index_list[0].append(Experiment_date)
            worm_num = [j for j in split_path if re.match("worm\d+", j)][0]
            S_index_list[1].append(worm_num)
            Experiment_num = [j for j in split_path if re.match("Ex-\d+", j)][0].split("_")[0]
            S_index_list[2].append(Experiment_num)
            island_num = [j for j in split_path if re.match("Sleep_island\d+", j)][0]
            S_index_list[3].append(island_num)
            all_index = "Sleep" + Experiment_date + worm_num + Experiment_num + island_num
            S_index_list[4].append(all_index)
            if len(S_index_list[0])<2:
                S_loc_data = temp_data["locomotion"]
                S_dR_R_data = temp_data["dR_R"]
                S_tracking_judge = temp_data["tracking_judge"]
            else:
                S_loc_data = pd.concat([S_loc_data, temp_data["locomotion"]], axis=1)
                S_dR_R_data = pd.concat([S_dR_R_data, temp_data["dR_R"]], axis=1)
                S_tracking_judge = pd.concat([S_tracking_judge, temp_data["tracking_judge"]], axis=1)

    tuples = list(zip(*M_index_list))
    M_index_list = pd.MultiIndex.from_tuples(tuples, names=['ExDate', 'Worm_num', "Ex", "island", "all_name"])

    M_bout_loc_data = pd.DataFrame(M_bout_loc_data.values.T, index=M_index_list).T
    M_bout_loc_data.to_csv("./extracted_data/Motion/before_and_bout_locomotion.csv")

    M_bout_dR_R_data = pd.DataFrame(M_bout_dR_R_data.values.T, index=M_index_list).T
    M_bout_dR_R_data.to_csv("./extracted_data/Motion/before_and_bout_locomotion_dR_R_data.csv")

    M_bout_tracking_judge = pd.DataFrame(M_bout_tracking_judge.values.T, index=M_index_list).T
    M_bout_tracking_judge.to_csv("./extracted_data/Motion/before_and_bout_locomotion_tracking_judge.csv")

    tuples = list(zip(*S_index_list))
    S_index_list = pd.MultiIndex.from_tuples(tuples, names=['ExDate', 'Worm_num', "Ex", "island", "all_name"])

    S_loc_data = pd.DataFrame(S_loc_data.values.T, index=S_index_list).T
    S_loc_data.to_csv("./extracted_data/Sleep/before_and_bout_locomotion.csv")

    S_dR_R_data = pd.DataFrame(S_dR_R_data.values.T, index=S_index_list).T
    S_dR_R_data.to_csv("./extracted_data/Sleep/before_and_bout_locomotion_dR_R_data.csv")

    S_tracking_judge = pd.DataFrame(S_tracking_judge.values.T, index=S_index_list).T
    S_tracking_judge.to_csv("./extracted_data/Sleep/before_and_bout_locomotion_tracking_judge.csv")

def extract_MS_tr(target_dir):
    MS_tr_path = []
    # get path
    for curDir, dirs, files in os.walk(target_dir):
        for file in files:
            if re.match("MStr.csv", file):
                MS_tr_path.append(curDir + "/" + file)
        if not MS_tr_path:
            continue
    index_list = [[], [], [], [], []]
    for i in range(len(MS_tr_path)):
        temp_path = os.path.abspath(MS_tr_path[i])
        temp_data = pd.read_csv(temp_path)
        split_path = [j for j in temp_path.split("\\")]

        Experiment_date = [j for j in split_path if re.match("ExDate\d+", j)][0]
        index_list[0].append(Experiment_date)
        worm_num = [j for j in split_path if re.match("worm\d+", j)][0]
        index_list[1].append(worm_num)
        Experiment_num = [j for j in split_path if re.match("Ex-\d+", j)][0].split("_")[0]
        index_list[2].append(Experiment_num)
        if not [j for j in split_path if re.match("island\d+", j)]:
            island_or_tr = [j for j in split_path if re.match("only_MStr", j)][0]
            index_list[3].append(island_or_tr)
        else:
            island_or_tr = [j for j in split_path if re.match("island\d+", j)][0]
            index_list[3].append(island_or_tr)
        all_index = Experiment_date + worm_num + Experiment_num + island_or_tr
        index_list[4].append(all_index)

        if i == 0:
            MS_tr_data = temp_data["dR_R"]
        else:
            MS_tr_data = pd.concat([MS_tr_data, temp_data["dR_R"]], axis=1)

    tuples = list(zip(*index_list))
    index_list = pd.MultiIndex.from_tuples(tuples, names=['ExDate', 'Worm_num', "Ex", "island", "all_name"])
    MS_tr_data = pd.DataFrame(MS_tr_data.values.T, index=index_list).T
    MS_tr_data.to_csv("./extracted_data/Motion/MStr_data.csv")


def extract_SM_tr(target_dir):
    SM_tr_path = []
    # get path
    for curDir, dirs, files in os.walk(target_dir):
        for file in files:
            if re.match("SMtr.csv", file):
                SM_tr_path.append(curDir + "/" + file)
        if not SM_tr_path:
            continue
    index_list = [[], [], [], [], []]
    for i in range(len(SM_tr_path)):
        temp_path = os.path.abspath(SM_tr_path[i])
        temp_data = pd.read_csv(temp_path)
        split_path = [j for j in temp_path.split("\\")]

        Experiment_date = [j for j in split_path if re.match("ExDate\d+", j)][0]
        index_list[0].append(Experiment_date)
        worm_num = [j for j in split_path if re.match("worm\d+", j)][0]
        index_list[1].append(worm_num)
        Experiment_num = [j for j in split_path if re.match("Ex-\d+", j)][0].split("_")[0]
        index_list[2].append(Experiment_num)
        if not [j for j in split_path if re.match("island\d+", j)]:
            island_or_tr = [j for j in split_path if re.match("only_SMtr", j)][0]
            index_list[3].append(island_or_tr)
        else:
            island_or_tr = [j for j in split_path if re.match("island\d+", j)][0]
            index_list[3].append(island_or_tr)
        all_index = Experiment_date + worm_num + Experiment_num + island_or_tr
        index_list[4].append(all_index)

        if i == 0:
            SM_tr_data = temp_data["dR_R"]
        else:
            SM_tr_data = pd.concat([SM_tr_data, temp_data["dR_R"]], axis=1)

    tuples = list(zip(*index_list))
    index_list = pd.MultiIndex.from_tuples(tuples, names=['ExDate', 'Worm_num', "Ex", "island", "all_name"])
    SM_tr_data = pd.DataFrame(SM_tr_data.values.T, index=index_list).T
    SM_tr_data.to_csv("./extracted_data/Motion/SMtr_data.csv")

def Motion_bouts_information_extract(target_dir):
    Motion_bouts_info_paths = []
    # get path
    for curDir, dirs, files in os.walk(target_dir):
        for file in files:
            if re.match("Motion_bouts_information.csv", file):
                Motion_bouts_info_paths.append(curDir + "/" + file)
        if not Motion_bouts_info_paths:
            continue
    for i in range(len(Motion_bouts_info_paths)):
        tempdata = pd.read_csv(Motion_bouts_info_paths[i])
        if i == 0:
            Motion_bouts_info_df = tempdata
        else:
            Motion_bouts_info_df = pd.concat([Motion_bouts_info_df, tempdata])
    Motion_bouts_info_df.to_csv("./extracted_data/Motion_bouts_info_df.csv")

def Sleep_bouts_information_extract(target_dir):
    Sleep_bouts_info_paths = []
    # get path
    for curDir, dirs, files in os.walk(target_dir):
        for file in files:
            if re.match("Sleep_bouts_information.csv", file):
                Sleep_bouts_info_paths.append(curDir + "/" + file)
        if not Sleep_bouts_info_paths:
            continue
    for i in range(len(Sleep_bouts_info_paths)):
        tempdata = pd.read_csv(Sleep_bouts_info_paths[i])
        if i == 0:
            Sleep_bouts_info_df = tempdata
        else:
            Sleep_bouts_info_df = pd.concat([Sleep_bouts_info_df, tempdata])
    Sleep_bouts_info_df.to_csv("./extracted_data/Sleep_bouts_info_df.csv")


def main():
    target_dir = select_dir()
    extract_before_and_bout(target_dir)
    extract_MS_tr(target_dir)
    extract_SM_tr(target_dir)
    Motion_bouts_information_extract(target_dir)
    Sleep_bouts_information_extract(target_dir)

if __name__ == '__main__':
    main()
