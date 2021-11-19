import PySimpleGUI as sg

def calibrationWindow():

    layout = [
        [sg.Text('Calibration')],
        [sg.Button('Close', key="Exit")]
    ]

    window = sg.Window("Results", layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break