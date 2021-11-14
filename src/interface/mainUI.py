import PySimpleGUI as sg
from results import show_results
from calibration import calibrate
from experiment import experiment

# Main window
def main():
    layout = [
        [sg.Text("Title"), sg.Button("Calibrate", key=("calibrate")), sg.Button("Start", key=("start_test")), sg.Button("Show Results", key=("show_results"))],
    ]

    window = sg.Window("Main Window", layout, size=(800, 600)).Finalize()

    while True:
        event, values = window.read()
        if  event == sg.WIN_CLOSED:
            break
        if event == "show_results":
            show_results()
        if event == "calibrate":
            calibrate()
        if event == "start_test":
            experiment()

    window.close()

if __name__ == "__main__":
    main()