import PySimpleGUI as sg
import os, io
from PIL import Image

def experiment():

    images = ''

    frame = [
        [
            sg.Image(key="-IMAGE-")
        ]
    ]

    layout = [
        [
            sg.Text('Experiment'), 
            sg.Button('Exit')
        ],
        [
            sg.HorizontalSeparator()
        ],
        [
            sg.Frame('Image Projection', frame)
        ]
    ]

    window = sg.Window("Window", layout).finalize()
    window.Maximize()

    while True:
        for image in images:
            image = Image.open(values[image])
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-IMAGE-"].update(data=bio.getvalue())
            

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()
