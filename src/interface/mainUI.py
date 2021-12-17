import PySimpleGUI as sg
from results import resultsWindow
from calibration import calibrationWindow
from create_new_project import createFolder
from settings import SettingsWindow

# Main window
def main():
    layout = [
        [sg.Text("Project Eye Tracking"), sg.Button("New Project", key=("new_project")), sg.Button("Show Results", key=("show_results")), sg.Button("Settings", key=("settings"))],
        [sg.HorizontalSeparator()],
    ]

    window = sg.Window("Main Window", layout, size=(800, 600)).Finalize()

    while True:
        event, values = window.read()
        if  event == sg.WIN_CLOSED:
            break
        if event == "show_results":
            resultsWindow()
        if event == "calibrate":
            calibrationWindow()
        if event == "new_project":
            createFolder()
        if event == "settings":
            SettingsWindow()

    window.close()

if __name__ == "__main__":
    main()