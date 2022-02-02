import PySimpleGUI as sg
import os
import sys
import cv2
from functions import start_message
from experiment import cycle_images
#from calibration_old import calibration

scriptDir = os.path.dirname(__file__)
tracking_folder = os.path.join(scriptDir, '../tracking/')
path = os.path.abspath(tracking_folder)

sys.path.append(path)
from tracking import gaze_tracker

gaze = gaze_tracker

def createFolder():
    # Get New Folder Name
    foldername = sg.popup_get_text("Project Name")
    if foldername is None:
        print("No name was given")
        return
    
    # Locate current folder
    scriptDir = os.path.dirname(__file__)
    dir_folder = os.path.join(scriptDir, '../data')
    data_folder = os.path.abspath(dir_folder)
    print(data_folder)

    final_folder_path = os.path.join(data_folder, foldername)
    
    # Create new folder
    os.mkdir(final_folder_path)
    print("Directory '% s' created" % foldername)
    
    sg.theme('SystemDefaultForReal') # Set Theme for PySimpleGUI
    
    # Add layout
    layout = [
        [sg.Text('Calibratie vereist om te starten!')],
        [sg.Button('Calibreer', key="calibrate")]
    ]
    
    # Create window & event loop
    window = sg.Window(foldername, layout)

    while False:
        event, values = window.read()
        if event == "calibrate":
            window.close()
            # calibration(final_folder_path, foldername)

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            cap.set(cv2.CAP_PROP_FPS,60)

            if (cap.isOpened()== False):
                print("Error opening video stream or file")
                
            gaze.calibration(gaze,final_folder_path,foldername, cap)

            start_message(final_folder_path)
            
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    cycle_images(final_folder_path)

    window.close()

    # Return folder path for later use
    return final_folder_path, foldername
