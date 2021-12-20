"""
ALAのデータはdeltaF/Fという名前にしておくこと
"""

import os
import re
import numpy as np
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd

# parameters
sampling_num = 80


def ask_dir_name():
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showinfo('select directory', 'select analyzing directory')
    directory = filedialog.askdirectory()
    os.chdir(directory)


def search_subdivided_dir_path():
    locomotor_active_path_list = []
    MQtr_path_list = []
    QMtr_path_list = []
    during_path_list = []
    date_list = []
    for curDir, dirs, files in os.walk("./"):
        for file in files:
            # get Ex date
            if len(curDir.split("\\")) != 1:
                Ex_date_list = [i for i in curDir.split("\\") if re.match("Ex-\d+_date_\d+-\d+-\d+", i)]
                if Ex_date_list:
                    Ex_date = Ex_date_list[0].split("_")[-1]
                    if Ex_date in date_list:
                        pass
                    else:
                        date_list.append(Ex_date)
                else:
                    pass
            if re.compile("locomotor_active_df_\d+.csv").search(file):
                locomotor_active_path_list.append(os.path.join(curDir, file))
            elif os.path.basename(file) == "averaged_MQ.csv":
                MQtr_path_list.append(os.path.join(curDir, file))
            elif os.path.basename(file) == "averaged_QM.csv":
                QMtr_path_list.append(os.path.join(curDir, file))
            elif os.path.basename(file) == "during.csv":
                during_path_list.append(os.path.join(curDir, file))
            else:
                pass
    return locomotor_active_path_list, during_path_list, MQtr_path_list, QMtr_path_list, date_list


def locomotor_active_neuron_extraction(data_path_list, date_list):
    # make directories
    os.makedirs("./extracted_data", exist_ok=True)
    # MQ data extraction
    # initiation the date list
    Ex_date_list = []
    df_dict = {}
    for path in data_path_list:
        # check directory type, island? MQ?
        Ex_date = [i for i in path.split("\\") if re.match("Ex-\d+_date_\d+-\d+-\d+", i)][0].split("_")[-1]
        Ex_num = [i for i in path.split("\\") if re.match("Ex-\d+_date_\d+-\d+-\d+", i)][0].split("_")[0]
        os.makedirs("./extracted_data/{}".format(Ex_date), exist_ok=True)
        data = pd.read_csv(path, index_col=0)
        columns_list = list(data.columns)
        columns_list = map(lambda x: x + Ex_num, columns_list)
        meanlist = []
        start = data["time"].iloc[0]
        sampling_bin = (data["time"].max() - data["time"].min()) / sampling_num
        for n in range(sampling_num):
            end = start + sampling_bin
            data_extraction_mask = np.where((data["time"] >= start) \
                                            & (data["time"] <= end), True, False)
            extracted_fluo_data = data[data_extraction_mask]
            meanlist.append(extracted_fluo_data.mean())
            start = end
        mean_array = np.array(meanlist)
        mean_df = pd.DataFrame(mean_array, columns=columns_list)
        # if the first time
        if Ex_date in df_dict:
            df_dict[Ex_date] = pd.concat([df_dict[Ex_date], mean_df], axis=1)
        else:
            df_dict[Ex_date] = mean_df
    for i in list(df_dict.keys()):
        df_dict[i].to_csv("./extracted_data/{}/locomotor_active_summary.csv".format(i))


def MQ_and_QM_data_extraction(MQ_list, QM_list, date_list):
    # initialization
    is_labels_MQ = [[0] for i in range(len(date_list))]
    is_labels_QM = [[0] for i in range(len(date_list))]
    MQ_labels = [[0] for i in range(len(date_list))]
    QM_labels = [[0] for i in range(len(date_list))]

    # MQ data extraction
    # initiation the data list
    island_MQ_data = [[0] for i in range(len(date_list))]
    all_MQ_data = [[0] for i in range(len(date_list))]
    for path in MQ_list:
        # check directory type, island? MQ?
        dir_type = path.split("\\")[-2]
        Ex_date = [i for i in path.split("\\") if re.match("Ex-\d+_date_\d+-\d+-\d+", i)][0].split("_")[-1]
        os.makedirs("./extracted_data/{}".format(Ex_date), exist_ok=True)
        # index of Ex_date
        index_of_date = date_list.index(Ex_date)
        data = pd.read_csv(path)
        if "deltaF/F" in data.columns.values:
            if re.match("island\d+", dir_type):
                temp_array = data["deltaF/F"].values
                if len(island_MQ_data[index_of_date]) == 1:
                    island_MQ_data[index_of_date] = temp_array
                else:
                    island_MQ_data[index_of_date] = np.vstack((island_MQ_data[index_of_date],
                                                               temp_array))
                # island_MQ_data.append(temp_array)
                if len(all_MQ_data[index_of_date]) == 1:
                    all_MQ_data[index_of_date] = temp_array
                else:
                    all_MQ_data[index_of_date] = np.vstack((all_MQ_data[index_of_date], temp_array))
                # all_MQ_data.append(temp_array)
                is_labels_MQ[index_of_date].append("MQ_" + path.split("\\")[1] + "_" + path.split("\\")[-2])
                MQ_labels[index_of_date].append("MQ_" + path.split("\\")[1] + "_" + path.split("\\")[-2])
            # MQ tr data
            else:
                temp_array = data["deltaF/F"].values
                if len(all_MQ_data[index_of_date]) == 1:
                    all_MQ_data[index_of_date] = temp_array
                else:
                    all_MQ_data[index_of_date] = np.vstack((all_MQ_data[index_of_date], temp_array))
                MQ_labels[index_of_date].append("MQ_" + path.split("\\")[1] + "_" + path.split("\\")[-2])
        else:
            pass
    # make dataframes
    for i in range(len(date_list)):
        if type(island_MQ_data[i]) is np.ndarray:
            island_MQ_df = pd.DataFrame(np.array(island_MQ_data[i]).T, columns=is_labels_MQ[i][1:])
            all_MQ_df = pd.DataFrame(np.array(all_MQ_data[i]).T, columns=MQ_labels[i][1:])
            # save
            island_MQ_df.to_csv("./extracted_data/{}/island_MQ.csv".format(date_list[i]))
            all_MQ_df.to_csv("./extracted_data/{}/all_MQ.csv".format(date_list[i]))
        else:
            pass

    # QM data extraction
    island_QM_data = [[0] for i in range(len(date_list))]
    all_QM_data = [[0] for i in range(len(date_list))]
    for path in QM_list:
        dir_type = path.split("\\")[-2]
        Ex_date = [i for i in path.split("\\") if re.match("Ex-\d+_date_\d+-\d+-\d+", i)][0].split("_")[-1]
        os.makedirs("./extracted_data/{}".format(Ex_date), exist_ok=True)
        index_of_date = date_list.index(Ex_date)
        data = pd.read_csv(path)
        if "deltaF/F" in data.columns.values:
            if re.match("island\d+", dir_type):
                temp_array = data["deltaF/F"].values
                if len(island_QM_data[index_of_date]) == 1:
                    island_QM_data[index_of_date] = temp_array
                else:
                    island_QM_data[index_of_date] = np.vstack((island_QM_data[index_of_date], temp_array))
                if len(all_QM_data[index_of_date]) == 1:
                    all_QM_data[index_of_date] = temp_array
                else:
                    all_QM_data[index_of_date] = np.vstack((all_QM_data[index_of_date], temp_array))
                is_labels_QM[index_of_date].append("QM_" + path.split("\\")[1] + "_" + path.split("\\")[-2])
                QM_labels[index_of_date].append("QM_" + path.split("\\")[1] + "_" + path.split("\\")[-2])
            else:
                temp_array = data["deltaF/F"].values
                if len(all_QM_data[index_of_date]) == 1:
                    all_QM_data[index_of_date] = temp_array
                else:
                    all_QM_data[index_of_date] = np.vstack((all_QM_data[index_of_date], temp_array))
                QM_labels[index_of_date].append("QM_" + path.split("\\")[1] + "_" + path.split("\\")[-2])
        else:
            pass
    for i in range(len(date_list)):
        if type(island_QM_data[i]) is np.ndarray:
            island_QM_df = pd.DataFrame(np.array(island_QM_data[i]).T, columns=is_labels_QM[i][1:])
            all_QM_df = pd.DataFrame(np.array(all_QM_data[i]).T, columns=QM_labels[i][1:])
            # save
            island_QM_df.to_csv("./extracted_data/{}/island_QM.csv".format(date_list[i]))
            all_QM_df.to_csv("./extracted_data/{}/all_QM.csv".format(date_list[i]))
        else:
            pass


def normalized_during_activity_extraction(during_path_list, date_list):
    labels_list = [[0] for i in range(len(date_list))]
    # MQ data extraction
    # initiation the data list
    during_data = [[0] for i in range(len(date_list))]
    for path in during_path_list:
        Ex_date = [i for i in path.split("\\") if re.match("Ex-\d+_date_\d+-\d+-\d+", i)][0].split("_")[-1]
        index_of_date = date_list.index(Ex_date)
        os.makedirs("./extracted_data/{}".format(Ex_date), exist_ok=True)
        # extract fluo data
        data = pd.read_csv(path)
        columns = str(list(data.columns))
        fluo_data_name_match = re.findall("deltaF/F+", columns)
        if fluo_data_name_match:
            fluo_data_name = fluo_data_name_match[0]
            fluo_data = data[fluo_data_name]
            meanlist = []
            start = data["time"][0]
            sampling_bin = (data["time"].max() - data["time"].min()) / sampling_num
            for n in range(sampling_num):
                end = start + sampling_bin
                data_extraction_mask = np.where((data["time"] >= start) \
                                                & (data["time"] <= end), True, False)
                extracted_fluo_data = fluo_data[data_extraction_mask]
                meanlist.append(extracted_fluo_data.mean())
                start = end
            mean_array = np.array(meanlist)
            # if the first time
            if len(during_data[index_of_date]) == 1:
                during_data[index_of_date] = mean_array
            else:
                during_data[index_of_date] = np.vstack((during_data[index_of_date],
                                                        mean_array))
            labels_list[index_of_date].append("During_" + path.split("\\")[1] + "_" + path.split("\\")[-2])

    for i in range(len(date_list)):
        if type(during_data[i]) is np.ndarray:
            during_df = pd.DataFrame(np.array(during_data[i].T), columns=labels_list[i][1:])
            # save
            during_df.to_csv("./extracted_data/{}/during_summary.csv".format(date_list[i]))
        else:
            pass

def locomotor_activity_extract(during_path_list,date_list):
    # intialization of lists
    labels_list = [[0] for i in range(len(date_list))]
    during_loc_data = [[0] for i in range(len(date_list))]
    for path in during_path_list:
        Ex_date = [i for i in path.split("\\") if re.match("Ex-\d+_date_\d+-\d+-\d+", i)][0].split("_")[-1]
        index_of_date = date_list.index(Ex_date)
        # extract fluo data
        data = pd.read_csv(path)
        columns = str(list(data.columns))
        loc_data = data["locomotion"]
        meanlist = []
        start = data["time"][0]
        sampling_bin = (data["time"].max() - data["time"].min()) / sampling_num
        for n in range(sampling_num):
            end = start + sampling_bin
            data_extraction_mask = np.where((data["time"] >= start)
                                            & (data["time"] <= end), True, False)
            extracted_fluo_data = loc_data[data_extraction_mask]
            meanlist.append(extracted_fluo_data.mean())
            start = end
        mean_array = np.array(meanlist)
            # if the first time
        if len(during_loc_data[index_of_date]) == 1:
            during_loc_data[index_of_date] = mean_array
        else:
            during_loc_data[index_of_date] = np.vstack((during_loc_data[index_of_date],
                                                    mean_array))
        labels_list[index_of_date].append("During_" + path.split("\\")[1] + "_" + path.split("\\")[-2])

    for i in range(len(date_list)):
        if type(during_loc_data[i]) is np.ndarray:
            during_loc_df = pd.DataFrame(np.array(during_loc_data[i].T), columns=labels_list[i][1:])
            # save
            during_loc_df.to_csv("./extracted_data/{}/during_loc_summary.csv".format(date_list[i]))
        else:
            pass


def locomotor_activity_extract(during_path_list,date_list):
    # intialization of lists
    labels_list = [[0] for i in range(len(date_list))]
    during_loc_data = [[0] for i in range(len(date_list))]
    for path in during_path_list:
        Ex_date = [i for i in path.split("\\") if re.match("Ex-\d+_date_\d+-\d+-\d+", i)][0].split("_")[-1]
        index_of_date = date_list.index(Ex_date)
        # extract fluo data
        data = pd.read_csv(path)
        columns = str(list(data.columns))
        loc_data = data["locomotion"]
        meanlist = []
        start = data["time"][0]
        sampling_bin = (data["time"].max() - data["time"].min()) / sampling_num
        for n in range(sampling_num):
            end = start + sampling_bin
            data_extraction_mask = np.where((data["time"] >= start)
                                            & (data["time"] <= end), True, False)
            extracted_fluo_data = loc_data[data_extraction_mask]
            meanlist.append(extracted_fluo_data.mean())
            start = end
        mean_array = np.array(meanlist)
            # if the first time
        if len(during_loc_data[index_of_date]) == 1:
            during_loc_data[index_of_date] = mean_array
        else:
            during_loc_data[index_of_date] = np.vstack((during_loc_data[index_of_date],
                                                    mean_array))
        labels_list[index_of_date].append("During_" + path.split("\\")[1] + "_" + path.split("\\")[-2])

    for i in range(len(date_list)):
        if type(during_loc_data[i]) is np.ndarray:
            during_loc_df = pd.DataFrame(np.array(during_loc_data[i].T), columns=labels_list[i][1:])
            # save
            during_loc_df.to_csv("./extracted_data/{}/during_loc_summary.csv".format(date_list[i]))
        else:
            pass



def main():
    ask_dir_name()
    locomotor_active_path_list, during_path_list, MQtr_path_list, QMtr_path_list, date_list = search_subdivided_dir_path()
    locomotor_active_neuron_extraction(locomotor_active_path_list, date_list)
    MQ_and_QM_data_extraction(MQtr_path_list, QMtr_path_list, date_list)
    normalized_during_activity_extraction(during_path_list, date_list)
    locomotor_activity_extract(during_path_list, date_list)


if __name__ == '__main__':
    main()
