import PySimpleGUI as sg
import os
from experiment import cycle_images

def createFolder():
    foldername = sg.popup_get_text("Project Name")
    if foldername is None:
        print("No name was given")
        return

    scriptDir = os.path.dirname(__file__)
    dir_folder = os.path.join(scriptDir, '../data')
    data_folder = os.path.abspath(dir_folder)
    print(data_folder)

    final_path = os.path.join(data_folder, foldername)

    os.mkdir(final_path)
    print("Directory '% s' created" % foldername)
    startWindow()

    return foldername

def startWindow():

    layout = [
        [sg.Text('Klik op start om te starten!')],
        [sg.Button('Start', key="Start")]
    ]

    window = sg.Window("Window", layout)

    while True:
        event, values = window.read()
        if event == "Start"
            cycle_images()
            time.sleep(1)
            break
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    
    window.close()
