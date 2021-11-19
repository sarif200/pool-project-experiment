from tkinter import colorchooser
import PySimpleGUI as sg
import os, io, time
#from PIL import Image



def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1

def experiment():

    #images = ''

    frame = [
        [
            sg.Image('./src/interface/big-pause-button.png')
        ]
    ]

    layout = [
        [
            sg.Text('Experiment'),
        ],
        [
            sg.HorizontalSeparator()
        ],
        [
            sg.Frame('Image Projection', frame)
        ],
        [
            sg.Text(key="-TIME-", text_color="red")
        ]
    ]
    
    window = sg.Window("Window", layout).finalize()
    window.Maximize()
    
    while True:
        # for image in images:
        #     Img = Image.open(values[image])
        #     Img.thumbnail((400, 400))
        #     bio = io.BytesIO()
        #     Img.save(bio, format="PNG")
        #     window["-IMAGE-"].update(data=bio.getvalue())

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()
