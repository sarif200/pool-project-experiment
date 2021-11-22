import PySimpleGUI as sg
import os, io, time
from PIL import Image

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1

def experiment():

    img_folder = os.path.abspath('./src/img')
    images = os.listdir(img_folder)

    frame = [
        [
            # sg.Image('./src/img/big-pause-button.png')
            sg.Image(key="-IMAGE-")
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
        for image in images:
            print(image)
            Img = Image.open(values[image])
            Img.thumbnail((400, 400))
            bio = io.BytesIO()
            Img.save(bio, format="PNG")
            window["-IMAGE-"].update(data=bio.getvalue())
            t=30
            countdown(t)

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()
