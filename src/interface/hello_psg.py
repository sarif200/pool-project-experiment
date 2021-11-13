import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import WIN_CLOSED, Window

layout = [
    [sg.Text("Hello")],
    [sg.Button("Ok")]
]

window = sg.Window("Demo", layout)

while True:
    event, values = window.read()

    if event == "Ok" or event == sg.WIN_CLOSED:
        break

window.close()