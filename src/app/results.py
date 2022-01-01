# Import libraries
import PySimpleGUI as sg
import os
import cv2
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from tracking import pupil_tracker


# Open New Window
def resultsWindow():

    # Get Folder Location
    folder_location = sg.popup_get_folder('Open Project Folder')
    orgvidname = "original_video.mp4"
    textfile = "offset.txt"

    if folder_location is None:
        return
        
    finalfolder = os.path.abspath(folder_location)
    video_location = os.path.join(finalfolder, orgvidname)

    file_location = os.path.join(finalfolder, textfile)

    # Read textfile with offset
    document = open(file_location, "r") # Will open & create if file is not found
    offset = document.read()
    print(offset)

    scriptDir = os.path.dirname(__file__)
    imgdir_folder = os.path.join(scriptDir, '../img/')
    img_folder = os.path.abspath(imgdir_folder)

    # Get images from folder
    images = os.listdir(img_folder)

    cap = cv2.VideoCapture(video_location)

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
        # [
        #     sg.Text(
        #         'hsv',
        #         size=(10, 1),
        #         key='-HSV_MASK-',
        #         enable_events=True
        #     ),
        #     sg.Button('Blue', size=(10, 1)),
        #     sg.Button('Green', size=(10, 1)),
        #     sg.Button('Red', size=(10, 1))
        # ],
        # [
        #     sg.Checkbox(
        #         'Hue Reverse',
        #         size=(10, 1),
        #         key='-Hue Reverse_MASK-',
        #         enable_events=True
        #     )
        # ],
        # [
        #     sg.Text('Hue', size=(10, 1), key='-Hue_MASK-'),
        #     sg.Slider(
        #         (0, 255),
        #         0,
        #         1,
        #         orientation='h',
        #         size=(19.4, 15),
        #         key='-H_MIN SLIDER_MASK-',
        #         enable_events=True
        #     ),
        #     sg.Slider(
        #         (1, 255),
        #         125,
        #         1,
        #         orientation='h',
        #         size=(19.4, 15),
        #         key='-H_MAX SLIDER_MASK-',
        #         enable_events=True
        #     )
        # ],
        # [
        #     sg.Text('Saturation', size=(10, 1), key='-Saturation_MASK-'),
        #     sg.Slider(
        #         (0, 255),
        #         50,
        #         1,
        #         orientation='h',
        #         size=(19.4, 15),
        #         key='-S_MIN SLIDER_MASK-',
        #         enable_events=True
        #     ),
        #     sg.Slider(
        #         (1, 255),
        #         255,
        #         1,
        #         orientation='h',
        #         size=(19.4, 15),
        #         key='-S_MAX SLIDER_MASK-',
        #         enable_events=True
        #     )
        # ],
        # [
        #     sg.Text('Value', size=(10, 1), key='-Value_MASK-'),
        #     sg.Slider(
        #         (0, 255),
        #         50,
        #         1,
        #         orientation='h',
        #         size=(19.4, 15),
        #         key='-V_MIN SLIDER_MASK-',
        #         enable_events=True
        #     ),
        #     sg.Slider(
        #         (1, 255),
        #         255,
        #         1,
        #         orientation='h',
        #         size=(19.4, 15),
        #         key='-V_MAX SLIDER_MASK-',
        #         enable_events=True
        #     )
        # ]
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


    layout = [
        [
            sg.Text("Timeline", size=(8,1)),
            sg.Slider(
                orientation="horizontal",
                key="-PROGRESS SLIDER-",
                range=(0,num_frames),
                size=(44,15),
                default_value=0
            )
        ],
        [
            sg.Text("Speed: ", size=(8,1)),
            sg.Slider(
                orientation="horizontal",
                key="-SPEED SLIDER-",
                resolution=1,
                range=(0,20),
                default_value=10,
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
        [
            sg.Output(size=(65, 10), key='-OUTPUT-')
        ],
        [
            sg.Button('Clear'), sg.Button('Close', key="Exit")
        ]
    ]

    window = sg.Window("Results: " + folder_location, layout, resizable=True).finalize()
    
    # UI var
    progessSlider = window.Element('-PROGRESS SLIDER-')
    speedSlider = window.Element('-SPEED SLIDER-')

    # Const & presets
    cur_frame = 0
    speed = 1
    video_stop = True
    TIME = 3 # Time between images
    n = 0

    # Update project information
    window.Element('-FPS-').update(fps)
    window.Element('-SCREEN SIZE-').update(screenSize)
    window.Element('-TIME-').update(total_length_sec)

    # Calculate frames
    interval = int(fps * TIME)

    # tracker = pupil_tracker

    while True: 
        event, values = window.read(timeout=0)
        if event == ('Exit', None):
            break
        
        ret, frame = cap.read()
        ret1, frame_tracked = cap.read()
        # frame = cv2.resize(frame, (960, 540))
        # frame_tracked = cv2.resize(frame, (640, 400))

        if not ret and not ret1:  # if out of data stop looping
            cur_frame = 0

        # Button Events
        if int(values['-PROGRESS SLIDER-']) != cur_frame-1:
            cur_frame = int(values['-PROGRESS SLIDER-'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            progessSlider.update(cur_frame)
            cur_frame += 1 * (speed-1)

        if int(values['-SPEED SLIDER-']) != 10:
            speed = int(values['-SPEED SLIDER-']) / 10
            speedSlider.update(speed * 10)
        
        if event == 'Play / Stop':
            video_stop = not video_stop

        if video_stop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

        else:
            cur_frame += 1
            progessSlider.update(cur_frame + 1 * (speed-1))

        if event == 'Reset':
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            progessSlider.update(0)
            speed = 1
            speedSlider.update(10)

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
        
        # Show original video
        # cv2.imshow("Original video", frame)

        # Show tracked video
        # pupils = tracker.detect_in_frame(tracker,frame_tracked)
        # #print(pupils)
        # cv2.circle(frame_tracked,(int(pupils[0][0]),int(pupils[0][1])),10,(0,255,0),3)
        # cv2.circle(frame_tracked,(int(pupils[1][0]),int(pupils[1][1])),10,(0,255,0),3)

        # cv2.imshow('Tracked video', frame_tracked)

        # Combined video
        combined = cv2.vconcat([frame, frame_tracked])
        combined = cv2.resize(combined, (1000, 960))
        cv2.imshow("Video Stream", combined)

        # Screen projection
        # Loop true n
        # Get the current frame, if between frame x & y show image n
        for n in range(10):
            if (n * interval < cur_frame < (n+1) * interval):
                current_image = images[n]
                img_path = os.path.join(img_folder, current_image)
                img = cv2.imread(img_path)
                # img = cv2.resizeWindow(img, (960, 540))
                cv2.imshow("Image", img)
              
        
    cap.release()
    cv2.destroyAllWindows()
    window.close()

resultsWindow()