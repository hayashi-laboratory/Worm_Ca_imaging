# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:29:15 2021

@author: miyas
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import datetime

from tkinter import filedialog
import tkinter 
from tkinter import messagebox


#####functions#####

#get file path
def file_select():
    root = tkinter.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    if path != False:
        pass
    else:
        messagebox.showinfo('quit', 'stop the script')
        sys.exit()
    folderpath = os.path.dirname(path)
    os.chdir(folderpath)
    file_name = os.path.splitext(os.path.basename(path))[0]
    return path, folderpath, file_name
    
#make directories 
def makedirs(filename):
    date = datetime.date.today()
    os.makedirs('./Analyzed_{0}/{1}'.format(date, filename), exist_ok = True)
    os.chdir('./Analyzed_{0}/{1}'.format(date, filename))
    os.makedirs('./datas', exist_ok = True)
    os.makedirs('./datas/rawdata', exist_ok = True)
    os.makedirs('./datas/FFT_data', exist_ok = True)
    os.makedirs('./figures', exist_ok = True)

#get variables
def ask_variables():
    max_datalength = 1800#int(input("max data length: "))
    return max_datalength

#csv file read
def csv_file_read(filepath):
    file_dir, file_name = os.path.split(filepath)
    base, ext = os.path.splitext(file_name)
    if ext == '.csv':
        data = pd.read_csv(filepath)
        return data
    else:
        return messagebox.showinfo('error',
                            'selected file is not csv file')

# extraction of adequate dataframes from data
# (1) contain data length > max_datalength
# (2) data start from Frame0 
# (3) TRACK_ID = 0 is extracted

def data_extraction(filepath, max_datalength):
    data = csv_file_read(filepath)
    dataframe_list = []
    n = 0
    for Track_ID, dataframe in data.groupby('TRACK_ID'):
        if len(dataframe)>max_datalength and dataframe.iloc[0]["FRAME"] == 0 \
            and dataframe.iloc[0]["TRACK_ID"] != "None":
            dataframe_list.append(dataframe[0:max_datalength])
            n += 1
        else:
            pass
            n += 1
    return dataframe_list, n

def Hz_calc(dataframe_list):
    temp = dataframe_list[0]
    Hz = 1/(temp.iloc[1]["POSITION_T"]-temp.iloc[0]["POSITION_T"])
    return Hz

def calc_fluo(dataframe_list):
    Fluo_data = []
    only_Fluo_data = []
    data_for_correlation = []
    ID_list= []
    for i in range(len(dataframe_list)):
        temp_dataframe = dataframe_list[i]
        # obtain ID for temp 
        ID = temp_dataframe.iloc[0]["TRACK_ID"]
        # add ID to list
        ID_list.append(ID)
        mean_GFP_int = temp_dataframe["MEAN_INTENSITY01"].mean()
        temp_dataframe["delta_GFP"] = temp_dataframe["MEAN_INTENSITY01"] - mean_GFP_int
        temp_dataframe["deltaF/F"] = temp_dataframe["delta_GFP"] / temp_dataframe["MEAN_INTENSITY02"]
        temp_dataframe.to_csv('./datas/rawdata/Track_ID{}.csv'.format(ID))
        Fluo_data.append(temp_dataframe[["FRAME", "deltaF/F"]])
        only_Fluo_data.append(temp_dataframe[["deltaF/F"]])
        # add fluo data to correlation list
        data_for_correlation.append(np.squeeze(temp_dataframe[["deltaF/F"]].values))
    # make dataframe
    df_for_correlation = pd.DataFrame(np.array(data_for_correlation).T, columns = ID_list)
    
    return Fluo_data, only_Fluo_data, df_for_correlation


def write_readme(max_datalength, Hz, filepath):
    path_w = './readme.txt'
    contents = 'max_datalength: {0}\nHz: {1}\nFilepath: {2}'.format(max_datalength,Hz,filepath)
    with open(path_w, mode = "a") as f:
        f.write(contents)

def FFT_graph(data, N, Hz, axis):
    F = np.fft.fft(data)
    F_abs = np.abs(F)
    F_abs_amp = F_abs / N 
    fq = np.linspace(0, Hz, N)
    axis.plot(fq[:int(N/2)+1], F_abs_amp[:int(N/2)+1],color = "black")
    FFT_data = F_abs_amp[:int(N/2)+1]
    return FFT_data

def all_data_visualization(dataframe_list, Fluo_data, max_datalength, Hz):
    Num_of_pages = (len(dataframe_list)-1) // 20 +1 
    for i in range(Num_of_pages):
        temp_Fluo_data = Fluo_data[20*i:20*(i+1)]
        fig, ax = plt.subplots(4,5, figsize = (15,12), sharex = True, sharey = True)
        fig.text(0.5, 0.04, 'Frame', ha='center', fontsize = 20)
        fig.text(0.04, 0.5, 'deltaF/F', va='center', rotation='vertical', fontsize = 20)
        for j, axi in enumerate(ax.flat):
            try:
                axi.plot(np.arange(0,max_datalength),
                         temp_Fluo_data[j]["deltaF/F"], color = "black")
                plt.xlim(0,max_datalength)
                plt.ylim(-1, 1)
            except IndexError:
                axi.plot(np.arange(0,max_datalength), 
                         np.zeros(len(range(max_datalength))), color = "black")
                plt.xlim(0, max_datalength)
                plt.ylim(-1, 1)
        plt.savefig("./figures/Intensity_graph{}.png".format(i))
    num = 0
    cumulative_power = np.zeros(int(max_datalength/2) + 1)
    for i in range(Num_of_pages):
        temp_Fluo_data = Fluo_data[20*i:20*(i+1)]
        fig, ax = plt.subplots(4,5, figsize = (15,12), sharex = True, sharey = True)
        fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', fontsize = 20)
        fig.text(0.04, 0.5, 'Amplitude (arb. units)', va='center',
                 rotation='vertical', fontsize = 20)
        for j, axi in enumerate(ax.flat):
            try:
                data = temp_Fluo_data[j]["deltaF/F"]
                FFT_data = FFT_graph(data, len(data), Hz, axi)
                cumulative_power += FFT_data
                np.savetxt('./datas/FFT_data/Track_ID{}.csv'.format(num), FFT_data)
                num += 1
            except IndexError:
                data = np.zeros(len(temp_Fluo_data[0]))
                FFT_data = FFT_graph(data, len(data), 10, axi)
                num += 1
        plt.savefig("./figures/FFT_graph{}.png".format(i))
#FFT axis save 
    FFTaxis = np.linspace(0, Hz, max_datalength)[:int(max_datalength/2)+1]
    np.savetxt('./datas/FFT_data/FFTaxis.csv', FFTaxis)
    return FFTaxis, cumulative_power


## Intensity trace
def intensity_trace(max_datalength, only_Fluo_data, Hz):
    timeaxis = np.arange(max_datalength+1) * (1/Hz)
    NeuroID = np.arange(len(only_Fluo_data)+1)
# dataprep
    meshdata = np.squeeze(np.array([np.array(i) for i in only_Fluo_data]))
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    heatmap = ax1.pcolormesh(timeaxis,NeuroID, meshdata,cmap= plt.cm.jet)
    plt.ylabel("Neuron ID", fontsize = 24)
    plt.xlabel("TIme (s)",  fontsize = 24)
    heatmap.set_clim(-0.2,1.5)
    fig1.colorbar(heatmap, ax=ax1)
    plt.savefig('./figures/Intensity_trace.png')
    plt.show()
    return meshdata, timeaxis, NeuroID

# Distribution of Fluo intensity 
def intensity_histogram(meshdata):
    fig2, ax2 = plt.subplots(figsize=(18, 8))
    flat_Fluo_data = meshdata.ravel()
    plt.hist(flat_Fluo_data, bins=300, density=True, alpha = 0.2,
             histtype='stepfilled', color='r')
    plt.xlim(0, 2)
    plt.xlabel('deltaF/F', fontsize = 24)
    plt.ylabel('Relative probability', fontsize = 24)
    plt.savefig("./figures/Histogram.png")

# Power spectrral densities
def power_spectral_density(FFTaxis,cumulative_power):
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    plt.plot(FFTaxis, cumulative_power, linewidth=2, color="red")
    plt.fill_between(FFTaxis, cumulative_power, color="red", alpha=0.1)
    plt.xlabel('Frequency (Hz)', fontsize = 24)
    plt.ylabel('Cumulative power', fontsize = 24)
    plt.savefig("./figures/Cumulative Power.png")

# Neuro correlation
def correlation(meshdata):
    import seaborn as sns
    Fluo_df = pd.DataFrame(meshdata.T)
    Fluo_df_corr = Fluo_df.corr()
    fig4, ax4 = plt.subplots(figsize=(12, 8)) 
    sns.heatmap(Fluo_df_corr, square=True, vmax=1, vmin=-1, center=0)
    plt.xlabel('NueroID', fontsize = 24)
    plt.ylabel('NeuroID', fontsize = 24)
    plt.savefig('./figures/Correlation.png')
    return Fluo_df_corr

# Correlation distribution 
def correlation_distribution(Fluo_df_corr):
    fig5, ax5 = plt.subplots(figsize=(12, 8)) 
    Fluo_corr_flat = np.array(Fluo_df_corr).ravel() 
    Fluo_corr_flat = np.delete(Fluo_corr_flat, np.where(Fluo_corr_flat == 1))   
    plt.hist(Fluo_corr_flat, bins=25, density=True, alpha = 0.2,
             histtype='stepfilled', color='r')
    np.savetxt('./datas/correlation.csv', Fluo_corr_flat)
    plt.xlabel('Correlation value', fontsize = 24)
    plt.ylabel('Probability', fontsize = 24)
    plt.savefig('./figures/Correlation_hist.png')

# Distance and correlation
# obtain how many neurons are there
def distance_and_correlation_analysis(dataframe_list):
    neuronNum = len(dataframe_list)
    combinations = list(itertools.combinations(np.arange(neuronNum),2))
    distance = []
    correlation = []
    for i in range(len(combinations)):
        neuron1data = dataframe_list[combinations[i][0]]
        neuron2data = dataframe_list[combinations[i][1]]
        # calculate average distance
        neuron1_position = np.array(neuron1data.loc[:,"POSITION_X":"POSITION_Z"])
        neuron2_position = np.array(neuron2data.loc[:,"POSITION_X":"POSITION_Z"])
        tempdistance = np.mean(np.sqrt(np.sum(np.square(neuron1_position-neuron2_position), 1)))
        # calculate correlation
        neuron1_activity = np.array(neuron1data["deltaF/F"])
        neuron2_activity = np.array(neuron2data["deltaF/F"])
        tempcorrelation = np.corrcoef(neuron1_activity, neuron2_activity)[0][1]
        distance.append(tempdistance)
        correlation.append(tempcorrelation)
    # figure 
    fig6, ax6 = plt.subplots(figsize=(12, 8)) 
    ax6.scatter(distance,correlation)
    ax6.set_xlabel('distance between two neurons', fontsize = 24)
    ax6.set_ylabel('activity correlation', fontsize = 24)
    plt.savefig('./figures/Activity correlation and distance.png')
    
def draw_heatmap(a, timeaxis, NeuroID, cmap=plt.cm.jet):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram

    metric = 'correlation'
    method = 'ward'
    fig7, ax7 = plt.subplots(figsize=(18, 8)) 
    main_axes = ax7
    #for adjust colorbar
    divider = make_axes_locatable(main_axes)
    plt.sca(divider.append_axes("left", 1.0, pad=0))
    
    ylinkage = linkage(pdist(a, metric=metric), method=method, metric=metric)
    ydendro = dendrogram(ylinkage, orientation='left', no_labels=True,
                         distance_sort='descending',
                         link_color_func=lambda x: 'black')
    a = a.loc[[a.index[i] for i in ydendro['leaves']]]
    
    plt.gca().set_axis_off()
    
    plt.sca(main_axes)
    heat = ax7.pcolormesh(timeaxis, NeuroID, a,cmap= plt.cm.jet)
    heat.set_clim(-0.2,1.5)
    #imshow(a, aspect='auto', interpolation='none',cmap=cmap, vmin=-0.5, vmax=2.0)
    cbar = fig7.colorbar(heat)
    cbar.set_label("deltaF/F", fontsize = 24)
    #ax7.axes.xaxis.set_ticks(timeaxis)
    ax7.axes.yaxis.set_ticks([])
    plt.xlabel("TIme (s)",  fontsize = 24)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().invert_yaxis()
    plt.savefig('./figures/Activity_and_dendrogram.png')
    
def activity_analysis(dataframe_list, Hz, max_datalength):
    neuronNum = len(dataframe_list)
    activity_list = []
    for i in range(neuronNum):
        temp_activity = []
        position_np = np.asarray(dataframe_list[i].loc[:,"POSITION_X":"POSITION_Z"])
        distance_np = np.sqrt(np.sum(np.square(np.diff(position_np, axis=0)), 1))
        np.asarray(activity_list.append(distance_np))
    activity = np.mean(np.asarray(activity_list), 0)
    #graph
    fig8 = plt.figure(figsize=(10, 3),tight_layout=True)
    ax8 = fig8.add_subplot(111)
    timeaxis = np.linspace(0, (1/Hz)*(max_datalength-2), max_datalength-1)
    ax8.plot(timeaxis, activity)
    ax8.set_xlabel('Time (sec)', fontsize = 32)
    ax8.set_ylabel('Activity', fontsize = 32)
    plt.ylim(0, 2)
    plt.savefig('./figures/Activity.png')
    np.savetxt('./datas/Locomotor_activity.csv', activity)
    return activity, activity_list

def position_of_neurons(dataframe_list, folderpath, filename):
    import random
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    image_path = folderpath + "/" + filename + ".png"
    image_directory = os.path.dirname(image_path)
    os.makedirs("./Neural_positions", exist_ok = True)
    correct_value = 0.614
    for i in range(len(dataframe_list)):
        img = Image.open(image_path)
        tempdataframe = dataframe_list[i]
        position = tempdataframe.iloc[0]["POSITION_X":"POSITION_Y"]
        ID = tempdataframe.iloc[0]["TRACK_ID"]
        text = "ID{}".format(ID)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("C:/Windows/WinSxS/amd64_microsoft"\
                                  "-windows-font-truetype-arial_"\
                                  "31bf3856ad364e35_10.0.18362.1"\
                                  "_none_44e0e02b2a9382cc/arial.ttf", 14)
        #font = ImageFont.truetype("/Library/Fonts/Microsoft/Arial.ttf")
        neuron_position = (int(position[0]/correct_value),
                           int(position[1]/correct_value))
        random_value = np.random.randint(4, 10)
        if neuron_position[1] > img.size[1]/2:
            draw.line(((int(position[0]/correct_value), 
                        int(position[1]/correct_value)),
                       (int(position[0]/correct_value), 
                        int(position[1]/correct_value)+random_value)),
                      fill=(255, 255, 0))
            draw.text((int(position[0]/correct_value),
                       int(position[1]/correct_value) + random_value),
                      text, font=font, fill='#FFF')
        else:
            draw.line(((int(position[0]/correct_value),
                        int(position[1]/correct_value)),
                       (int(position[0]/correct_value),
                        int(position[1]/correct_value)-random_value)),
                      fill=(255, 255, 0))
            draw.text((int(position[0]/correct_value),
                       int(position[1]/correct_value)-random_value - 11),
                      text, font=font, fill='#FFF')
        img.save("./Neural_positions/{0}.png".format(ID), 'PNG',
                 quality=300, optimize=True)
    
def active_bout_analysis(dataframe_list,activity):
    # make dirs 
    os.makedirs("./datas/motion_bout_analysis", exist_ok= True)
    # padding (locomotion data len lost 1 data)
    locomotion_data = np.insert(activity, 0, 0)
    # calculate statistics 
    mean = float(np.mean(locomotion_data))
    std = float(np.std(locomotion_data))
    # locomotion is defined as movement is larger than 0.3 um
    locomotion_bool = np.where(locomotion_data>0.3, True, False)
    np.savetxt('./datas/Locomotor_bool.csv', locomotion_bool)
    # analysis_start_time = 10 means first 10sec data is not used in analysis
    analysis_start_time = 10
    # active_bout_duration = 10 means 10sec before and after start moving (20sec) is analyzed  
    active_bout_duration = 20
    for i in range(len(dataframe_list)):
        # dataframe 
        tempdataframe = dataframe_list[i]
        # obtain ID 
        ID = tempdataframe.iloc[0]["TRACK_ID"]
        # make dir
        os.makedirs("./datas/motion_bout_analysis/ID={}".format(ID),
                    exist_ok= True)
        os.makedirs("./datas/quiescent_bout_analysis/ID={}".format(ID),
                    exist_ok= True)
        #obtain timeaxis (1800)
        time = tempdataframe["POSITION_T"].values
        # mask with bool (extract 3 sd movement timepoints)
        movement_timepoints = time[locomotion_bool]
        #quiescent bout analysis
        if len(movement_timepoints) == 0:
            pass
        elif movement_timepoints[0] > 40:
            q_start = movement_timepoints[0]-30
            q_data_extraction_mask = np.where((time>q_start) & (time<movement_timepoints[0]), True, False)
            q_extracted_data = tempdataframe["deltaF/F"].values[q_data_extraction_mask]
            q_extracted_time = tempdataframe["POSITION_T"].values[q_data_extraction_mask]
            q_temp_extracted_locomotion = locomotion_data[q_data_extraction_mask]
            temp_extracted_df = pd.DataFrame(np.stack([q_extracted_time, q_extracted_data,
                                                       q_temp_extracted_locomotion],
                                                      axis = 1), columns = ["time","deltaF/F", "locomotion"])
            temp_extracted_df.to_csv("./datas/quiescent_bout_analysis/"\
                                     "ID={0}/quiescent_state.csv".format(ID))
        else:
            pass
        
        extracted_timepoints = []
        # the start point of the imaging is judged as motion bout 
        # if the analysis start time is 10, first 10 seconds are 
        # eliminated from analysis
        previous_motion_bout_end = analysis_start_time
        # judgement for the each movement_timepoints are adequate for analysis
        # if the timepoint is within 20sec from previous timepoint, the timepoint 
        # is eliminated 
        for j in range(len(movement_timepoints)):
            temp_pt = movement_timepoints[j]
            if temp_pt > previous_motion_bout_end:
                extracted_timepoints.append(temp_pt)
            else:
                pass
            previous_motion_bout_end = temp_pt + active_bout_duration
            
        # normalized within extracted bout
        for m in range(len(extracted_timepoints)):
            temp_timepoint = extracted_timepoints[m]
            start = temp_timepoint - analysis_start_time
            end = temp_timepoint + active_bout_duration
            # make mask
            data_extraction_mask = np.where((time>start) & (time<end), True, False)
            # mask for before motion bout
            mask_for_norm = np.where((time>start) & (time<temp_timepoint), True, False)
            # calc mean during extracted 
            mean_GFP_extracted = tempdataframe["MEAN_INTENSITY01"].values[mask_for_norm].mean()
            tempdataframe["delta_GFP"] = tempdataframe["MEAN_INTENSITY01"] - mean_GFP_extracted
            tempdataframe["deltaF/F"] = tempdataframe["delta_GFP"] / tempdataframe["MEAN_INTENSITY02"]
            extracted_data = tempdataframe["deltaF/F"].values[data_extraction_mask]
            extracted_time = tempdataframe["POSITION_T"].values[data_extraction_mask]
            temp_extracted_locomotion = locomotion_data[data_extraction_mask]
            temp_extracted_df = pd.DataFrame(np.stack([extracted_time, extracted_data,
                                                       temp_extracted_locomotion],
                                                      axis = 1), columns = ["time","deltaF/F", "locomotion"])
            temp_extracted_df.to_csv("./datas/motion_bout_analysis/"\
                                     "ID={0}/norm_time={1}.csv".format(ID,
                                                                  format(temp_timepoint, '.3f')))
            
            #averaging data with time_interval
            meanlist = []
            stdlist = []
            mean_locomo = []
            std_locomo = []
            time_interval =0.25
            num_of_data =int((analysis_start_time + active_bout_duration) / time_interval)
            start = temp_extracted_df["time"][0]
            for n in range(num_of_data):
                end = start + time_interval
                data_extraction_mask = np.where((extracted_time>=start) & (extracted_time<=end), True, False)
                extracted_data = temp_extracted_df["deltaF/F"].values[data_extraction_mask]
                extracted_locomo_for_av =  temp_extracted_df["locomotion"].values[data_extraction_mask]
                meanlist.append(extracted_data.mean())
                stdlist.append(extracted_data.std())
                mean_locomo.append(extracted_locomo_for_av.mean())
                std_locomo.append(extracted_locomo_for_av.std())
                start = end
            timeaxis = np.arange(0,30,0.25) - 10
            avaraged_activity_df = pd.DataFrame(np.stack([timeaxis, meanlist, mean_locomo],
                                                      axis = 1), 
                                                columns = ["time","mean_deltaF/F", "mean_locomotion"])
            activity_std_df = pd.DataFrame(np.stack([timeaxis, stdlist, std_locomo],
                                                      axis = 1), 
                                                columns = ["time","std_deltaF/F", "std_locomotion"])
            avaraged_activity_df.to_csv("./datas/motion_bout_analysis/ID={0}/norm_time={1}_ave.csv".format(ID,
                                        format(temp_timepoint, '.3f')))
            activity_std_df.to_csv("./datas/motion_bout_analysis/ID={0}/norm_time={1}_std.csv".format(ID,
                                        format(temp_timepoint, '.3f')))
            
def correlation_among_neurons(df_for_correlation, activity):
    import seaborn as sns
    locomotion_data = np.insert(activity, 0, 0)
    # add locomotion column
    df_for_correlation["Loc"] = locomotion_data
    df_corr = df_for_correlation.corr()
    fig, ax = plt.subplots(figsize=(25, 20)) 
    plt.xticks(fontsize= 28)
    plt.yticks(fontsize= 28)
    sns.heatmap(df_corr, square=True, vmax=1, vmin=-1, center=0)
    plt.savefig('./figures/Correlation_among_neurons.png')
    df_corr.to_csv('./datas/rawdata/df_for_correlation.csv')


def execute(filepath):
    #get file path
    folderpath = os.path.dirname(filepath)
    os.chdir(folderpath)
    filename = os.path.splitext(os.path.basename(filepath))[0]
    print("processing: {}".format(filename))
    #makedirectories
    makedirs(filename)
    #get variables
    max_datalength= ask_variables()
    #data extraction
    dataframe_list, n = data_extraction(filepath, max_datalength)
    Hz = Hz_calc(dataframe_list)
    Fluo_data, only_Fluo_data, df_for_correlation = calc_fluo(dataframe_list)
    FFTaxis, cumulative_power = all_data_visualization(dataframe_list, 
                                                       Fluo_data,
                                                       max_datalength, Hz)
    meshdata, timeaxis, NeuroID = intensity_trace(max_datalength, 
                                                  only_Fluo_data, Hz)
    heatmap_data = pd.DataFrame(meshdata,
                                index = np.linspace(0,len(dataframe_list)-1,len(dataframe_list)),
                                columns = np.linspace(0, (1/Hz)*(max_datalength-1), max_datalength))
    intensity_histogram(meshdata)
    draw_heatmap(heatmap_data, timeaxis,NeuroID, cmap=plt.cm.jet)
    power_spectral_density(FFTaxis, cumulative_power)
    Fluo_df_corr = correlation(meshdata)
    correlation_distribution(Fluo_df_corr)
    distance_and_correlation_analysis(dataframe_list)
    activity, activity_list = activity_analysis(dataframe_list, Hz, max_datalength)
    write_readme(max_datalength, Hz, filepath)
    position_of_neurons(dataframe_list, folderpath, filename)  
    active_bout_analysis(dataframe_list,activity)
    correlation_among_neurons(df_for_correlation, activity)
    return meshdata, activity, activity_list


    
def main():
    filelist = []
    folderpath = filedialog.askdirectory()
    all_file_list = os.listdir(folderpath)
    csv_file_list = [i for i in all_file_list if os.path.splitext(i)[1] == '.csv' ]
    for i in range(len(csv_file_list)):
        if csv_file_list[i].split("-")[0] == "Experiment":
            filelist.append(csv_file_list[i])
        else:
            pass
    for i in range(len(filelist)):
        temppath = filelist[i]
        temppath = folderpath +"/" + temppath
        meshdata, activity, activity_list = execute(temppath)
    
############################################################################

####graph params#####
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
############################################################################

if __name__ == '__main__':
    main()