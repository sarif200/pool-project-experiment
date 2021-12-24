import PySimpleGUI as sg
import os
from calibration import calibration

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

    while True:
        event, values = window.read()
        if event == "calibrate":
            window.close()
            calibration(foldername)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    
    window.close()

    # Return folder path for later use
    return final_folder_path, foldername
