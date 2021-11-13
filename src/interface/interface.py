# Import packages
import PySimpleGUI as sg

# Create Layout
layout =  [
    [sg.Text('Welcome'), sg.Button('Calibrate',size=(7, 2)), sg.Button('Start',size=(7, 2)), sg.Button('Stop',size=(7, 2))],
    [sg.Text('Video')]
]

# Create Window
window = sg.Window("Window Title", layout, size=(800, 600)).Finalize()

while True:
    event, values = window.read()

    if event == "calibrate":
        # Calibration process
        pass
    
    elif event == "start":
        # Start Test
        pass

    elif event == "stop":
        # Stop test & display message
        pass

    if event == sg.WIN_CLOSED:
        break

window.close()
