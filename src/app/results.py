# Import libraries
import PySimpleGUI as sg
import os, io
import cv2
import numpy as np
from videoPlayer import VideoPlayer

# Open New Window
def resultsWindow():
    folder_location = sg.popup_get_folder('Open Project Folder')
    orgvidname = "original_video.mp4"

    if folder_location is None:
        return
        
    finalfolder = os.path.abspath(folder_location)
    orgvideo_location = os.path.join(finalfolder, orgvidname)

    print(orgvideo_location)

    cap = cv2.VideoCapture(orgvideo_location)

    screen_width = 500
    screen_height = 300
    screenSize = (screen_width, screen_height)

    # App States
    play = False
    delay = 0.023
    frame = 1
    frames = None

    sg.theme('SystemDefaultForReal') # Set Theme for PySimpleGUI

    # Layout
    # Tab group 1
    T1 = sg.Tab("Basic", [
        [
            sg.Text("Open Tracked Video"),
            sg.Button("Open", key="-OPEN TRACKED VID-")
        ],
        [
            sg.Text("Open Tracked Screen"),
            sg.Button("Open", key="-OPEN TRACKED SCRN-")
        ],
        [
            sg.Text("Open Timeline Screen"),
            sg.Button("Open", key="-OPEN TIMELINE SCRN-")
        ],
    ])

    # Tabgroup 2
    T2 = sg.Tab("processing", [
        [
            sg.Checkbox(
                'gray',
                size=(10, 1),
                key='-GRAY-',
                enable_events=True
            )
        ],
    ])

    # Tabgroup 3
    T3 = sg.Tab("mask", [
        [
            sg.Text(
                'hsv',
                size=(10, 1),
                key='-HSV_MASK-',
                enable_events=True
            ),
            sg.Button('Blue', size=(10, 1)),
            sg.Button('Green', size=(10, 1)),
            sg.Button('Red', size=(10, 1))
        ],
        [
            sg.Checkbox(
                'Hue Reverse',
                size=(10, 1),
                key='-Hue Reverse_MASK-',
                enable_events=True
            )
        ],
        [
            sg.Text('Hue', size=(10, 1), key='-Hue_MASK-'),
            sg.Slider(
                (0, 255),
                0,
                1,
                orientation='h',
                size=(19.4, 15),
                key='-H_MIN SLIDER_MASK-',
                enable_events=True
            ),
            sg.Slider(
                (1, 255),
                125,
                1,
                orientation='h',
                size=(19.4, 15),
                key='-H_MAX SLIDER_MASK-',
                enable_events=True
            )
        ],
        [
            sg.Text('Saturation', size=(10, 1), key='-Saturation_MASK-'),
            sg.Slider(
                (0, 255),
                50,
                1,
                orientation='h',
                size=(19.4, 15),
                key='-S_MIN SLIDER_MASK-',
                enable_events=True
            ),
            sg.Slider(
                (1, 255),
                255,
                1,
                orientation='h',
                size=(19.4, 15),
                key='-S_MAX SLIDER_MASK-',
                enable_events=True
            )
        ],
        [
            sg.Text('Value', size=(10, 1), key='-Value_MASK-'),
            sg.Slider(
                (0, 255),
                50,
                1,
                orientation='h',
                size=(19.4, 15),
                key='-V_MIN SLIDER_MASK-',
                enable_events=True
            ),
            sg.Slider(
                (1, 255),
                255,
                1,
                orientation='h',
                size=(19.4, 15),
                key='-V_MAX SLIDER_MASK-',
                enable_events=True
            )
        ]
    ])

    # Tabgroup 4
    T4 = sg.Tab("Save", [
        [
            sg.Button('Write', size=(10, 1)),
            sg.Radio(
                'DIVX',
                "RADIO1",
                key='-DIVX-',
                default=True,
                size=(8, 1)
            ),
            sg.Radio('MJPG', "RADIO1", key='-MJPG-', size=(8, 1)),
            sg.Radio('GIF', "RADIO1", key='-GIF-', size=(8, 1))
        ],
        [
            sg.Text('Caption', size=(10, 1)),
            sg.InputText(
                size=(32, 50),
                key='-CAPTION-',
                enable_events=True
            )
        ]
    ])

    left_col = [
        [
            sg.Text("Timeline", size=(8,1)),
            sg.Slider(
                orientation="horizontal",
                key="-PROGRESS SLIDER-",
                range=(1,100),
                size=(45,15),
                default_value=0
            )
        ],
        [
            sg.Text("Speed", size=(6,1)),
            sg.Slider(
                orientation="horizontal",
                key="-SPEED SLIDER-",
                resolution=0.1,
                range=(0,2),
                default_value=1,
                size=(45, 15)
            )
        ],
        [
            sg.Button('<<<', size=(5, 1)),
            sg.Button('<<', size=(5, 1)),
            sg.Button('<', size=(5, 1)),
            sg.Button('Play / Stop', size=(9, 1)),
            sg.Button('Reset', size=(7, 1)),
            sg.Button('>', size=(5, 1)),
            sg.Button('>>', size=(5, 1)),
            sg.Button('>>>', size=(5, 1))
        ],
        [sg.HorizontalSeparator()],
        [   
            sg.Image(filename='', key='-IMAGE-')
        ],
        [sg.HorizontalSeparator()],
        [
            sg.TabGroup(
                [[T1, T2, T3, T4]],
                tab_background_color="#ccc",
                selected_title_color="#fff",
                selected_background_color="#444",
                tab_location="topleft"
            )
        ],
        [sg.Output(size=(65, 5), key='-OUTPUT-')],
    ]

    vid_col = [
        [
            sg.Text('Original Video')
        ],
        [
            sg.Canvas(size=(500, 300), key="org_canvas", background_color="black")
        ],
                [
            sg.Text('Tracked Video')
        ],
        [
            sg.Canvas(size=(500, 300), key="tracked_canvas", background_color="black")
        ]
    ]

    scrn_col = [
        [
            sg.Text('Tracked Screen')
        ],
        [
            sg.Canvas(size=(500, 300), key="tracked_scrn_canvas", background_color="black")
        ],
                [
            sg.Text('Screen Timeline')
        ],
        [
            sg.Canvas(size=(500, 300), key="scrn_timeline_canvas", background_color="black")
        ]
    ]

    layout = [
        [sg.Column(left_col, element_justification='c'), sg.VSeparator(), sg.Column(vid_col, element_justification='c'), sg.VSeparator(), sg.Column(scrn_col, element_justification='c')],
        [sg.Button('Clear'), sg.Button('Close', key="Exit")]
    ]

    window = sg.Window("Results", layout, resizable=True)
    
    canvas = window.Element("org_canvas")

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        # Get Video Frame
        ret, frameOrg = cap.read()
        frame = cv2.resize(frameOrg, screenSize)

        print("succesfully started")

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["org_canvas"].update(data=imgbytes)

        # org_vid = VideoPlayer(orgvideo_location)
        # org_vid_width = 500
        # org_vid_height = int(org_vid_width * org_vid_height / org_vid_width)

        # frames = int(org_vid.frames)

        # window.Element('-PROGRESS SLIDER-').Update(range=(0, int(frames)), value=0)

        # canvas.config(width=org_vid_width)

        # frame = 0
        # delay = 1 / org_vid.fps

        # if event == "Play / Stop":
        #     if play:
        #         play = False
        #     else:
        #         play = True
        
        # if event == "-PROGRESS SLIDER-":
        #     pass
        

    window.close()