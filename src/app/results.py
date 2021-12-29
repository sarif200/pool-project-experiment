# Import libraries
from functools import total_ordering
import PySimpleGUI as sg
import os, time
import cv2
import numpy as np

# Open New Window
def resultsWindow():

    # Get Folder Location
    folder_location = sg.popup_get_folder('Open Project Folder')
    orgvidname = "original_video.mp4"
    textfile = "offset.txt"

    if folder_location is None:
        return
        
    finalfolder = os.path.abspath(folder_location)
    orgvideo_location = os.path.join(finalfolder, orgvidname)

    file_location = os.path.join(finalfolder, textfile)

    # Read textfile with offset
    document = open(file_location, "r") # Will open & create if file is not found
    offset = document.read()
    print(offset)

    # print(orgvideo_location)

    cap = cv2.VideoCapture(orgvideo_location)

    screen_width = 500
    screen_height = 300
    screenSize = (screen_width, screen_height)

    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_length = int(fps * num_frames)
    total_length_sec = int(total_length / 60)

    sg.theme('SystemDefaultForReal') # Set Theme for PySimpleGUI

    # Layout
    # Tab group 1
    T1 = sg.Tab("Basic", [
        [
            sg.Text(
                'Project Name: '
            ),
            sg.Text(
                '',
                key='-PROJECT NAME-'
            )
        ],
        [
            sg.Text(
                'FPS: '
            ),
            sg.Text(
                '',
                key='-FPS-'
            )
        ],
        [
            sg.Text(
                'Screen Size: '
            ),
            sg.Text(
                '',
                key='-SCREEN SIZE-'
            )
        ],
                [
            sg.Text(
                'Total Time (s): '
            ),
            sg.Text(
                '',
                key='-TIME-'
            )
        ]
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
                range=(0,num_frames),
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
        [sg.Output(size=(65, 15), key='-OUTPUT-')],
    ]

    vid_col = [
        [
            sg.Text('Original Video')
        ],
        [
            sg.Image(filename='', size=(500, 300), key="org_canvas")
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
            sg.Image(size=(500, 300), key="tracked_scrn_canvas")
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

    window = sg.Window("Results: " + folder_location, layout, resizable=True).finalize()
    
    # UI var
    original_video_frame = window.Element("org_canvas")
    screen_video_frame =  window.Element("tracked_scrn_canvas")
    progessSlider = window.Element('-PROGRESS SLIDER-')
    speedSlider = window.Element('-SPEED SLIDER-')

    # Const & presets
    cur_frame = 0
    speed = 1
    video_stop = True
    TIME = 3 # Time between images

    # Update project information
    window.Element('-FPS-').update(fps)
    window.Element('-SCREEN SIZE-').update(screenSize)
    window.Element('-TIME-').update(total_length_sec)

    # Calculate frames
    interval = int(fps * TIME)

    while True: 
        event, values = window.read(timeout=0)
        if event in ('Exit', None):
            break
        
        ret, frame = cap.read()

        frame = cv2.resize(frame, screenSize)

        if not ret:  # if out of data stop looping
            cur_frame = 0

        # Button Events
        if int(values['-PROGRESS SLIDER-']) != cur_frame-1:
            cur_frame = int(values['-PROGRESS SLIDER-'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            progessSlider.update(cur_frame)
            cur_frame += 1 * speed

        if int(values['-SPEED SLIDER-']) != speed:
            speed = int(values['-SPEED SLIDER-'])
            speedSlider.update(speed)
            # print(speed) #ok
        
        if event == 'Play / Stop':
            video_stop = not video_stop

        if video_stop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

        else:
            cur_frame += 1
            progessSlider.update(cur_frame + 1 * speed)

        if event == 'Reset':
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            progessSlider.update(0)
            speed = 1
            speedSlider.update(1)

        if event == '<':
            cur_frame -= 6
            progessSlider.update(cur_frame)
        
        if event == '<<':
            cur_frame -= 11
            progessSlider.update(cur_frame)
        
        if event == '<<<':
            pass

        if event == '>':
            cur_frame += 4
            progessSlider.update(cur_frame)

        if event == '>>':
            cur_frame += 9
            progessSlider.update(cur_frame)

        if event == '>>>':
            pass

        # Screen if between x and y show image 1
        if (cur_frame < interval):
            # Print image 1
            print("1")

        elif (interval < cur_frame < 2 * interval):
            print("2")

        elif (2 * interval < cur_frame < 3 * interval):
            print("3")

        elif (3 * interval < cur_frame < 4 * interval):
            print("4")

        elif (4 * interval < cur_frame < 5 * interval):
            print("5")

        elif (5 * interval < cur_frame < 6 * interval):
            print("6")

        elif (6 * interval < cur_frame < 7 * interval):
            print("7")
        
        elif (7 * interval < cur_frame < 8 * interval):
            print("8")
        
        elif (8 * interval < cur_frame < 9 * interval):
            print("9")
        
        elif (9 * interval < cur_frame < total_length_sec):
            print("10")

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        original_video_frame.update(data=imgbytes)
        
    cap.release()
    # cap1.release()
    window.close()

resultsWindow()