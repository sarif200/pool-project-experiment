from sys import maxsize
import PySimpleGUI as sg
import os, io, time
from PIL import Image


def experiment():

    frame = [
        [
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

    maxsize=(1200, 850)
    t = 30

    while True:
        # Get file path
        scriptDir = os.path.dirname(__file__)
        dir_folder = os.path.join(scriptDir, '../img/')
        img_folder = os.path.abspath(dir_folder)
        print(img_folder)

        # Get files from folder
        images = os.listdir(img_folder)
        print(images)

        # Count files in folder
        img_length = len(images)
        print(img_length)

        # Get files separatly
        for image in images:
            img_path = os.path.join(img_folder, image)
            # get_img(image)
            print(img_path)

            img = Image.open(img_path)
            img.thumbnail((400, 400))
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            print(img)
            window['-IMAGE-'].update(img)
            

            while t:
                mins, secs = divmod(t, 60)
                timer = '{:02d}:{:02d}'.format(mins, secs)
                print(timer, end="\r")
                time.sleep(1)
                t -= 1
                window['-TIME-'].update(t)

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()
