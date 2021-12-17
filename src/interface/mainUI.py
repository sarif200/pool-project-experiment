import PySimpleGUI as sg
from results import resultsWindow
from calibration import calibrationWindow
from temp import cycle_images
from settings import SettingsWindow

# Main window
def main():
    layout = [
        [sg.Text("Title"), sg.Button("Calibrate", key=("calibrate")), sg.Button("Start", key=("start_test")), sg.Button("Show Results", key=("show_results")), sg.Button("Settings", key=("settings"))],
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
        if event == "start_test":
            cycle_images()
        if event == "settings":
            SettingsWindow()

    window.close()

if __name__ == "__main__":
    main()