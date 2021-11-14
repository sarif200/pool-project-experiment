import PySimpleGUI as sg

def calibrate():

    layout = [
        [sg.Text('Calibration')],
        [sg.Button('Close', key="Exit")]
    ]

    window = sg.Window("Results", layout, location=(0,0))

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break