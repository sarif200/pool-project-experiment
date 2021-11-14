import PySimpleGUI as sg

def experiment():

    layout = [
        [sg.Text('Experiment')],
        [sg.Button('Close', key="Exit")]
    ]

    window = sg.Window("Results", layout, location=(0,0))

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break