# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:02:33 2021

@author: miyas
"""

import os 
import shutil
import tkinter 
from tkinter import messagebox
from tkinter import filedialog

def select_folder():
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showinfo('select dir', 'select dir')
    directory = filedialog.askdirectory() 
    return directory 

def main():
    dirpath = select_folder()
    os.chdir(dirpath)
    filelist = [os.path.splitext(i)[0] for i in os.listdir(dirpath) if os.path.splitext(i)[1] == '.csv' \
                or os.path.splitext(i)[1] == '.png']
    folder_list = list(set(filelist))
    os.makedirs("./rawdata", exist_ok = True)
    for i in folder_list:
        os.makedirs("./{}".format(i), exist_ok = True)
        os.makedirs("./{}/image_seq".format(i), exist_ok = True)
        shutil.copy("./{}.csv".format(i), "./{0}/{0}.csv".format(i))
        shutil.copy("./{}.png".format(i), "./{0}/{0}.png".format(i))
        shutil.move("./{}.csv".format(i), "./rawdata/{}.csv".format(i))
        shutil.move("./{}.png".format(i), "./rawdata/{}.png".format(i))
if __name__ == '__main__':
    main()