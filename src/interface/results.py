# Import libraries
import PySimpleGUI as sg
import os, io
import cv2 as cv
import numpy as np

file_types = [("GIF (*.gif)", "*.gif"),
                ("MP4 (*.mp4)", "*.mp4"),
              ("All files (*.*)", "*.*")]

# Open New Window
def resultsWindow():
    filename = sg.popup_get_file('Filename to play')

    if filename is None:
        return
    vidFile = cv.VideoCapture(filename)
    
    num_frames = vidFile.get(cv.CAP_PROP_FRAME_COUNT)
    fps = vidFile.get(cv.CAP_PROP_FPS)

    e_frame = num_frames
    s_frame = 0

    stop_flg = False

    # Layout
    # Tab group 1
    T1 = sg.Tab("Basic", [
        [
            sg.Text("Resize     ", size=(13, 1)),
            sg.Slider(
                (0.1, 4),
                1,
                0.01,
                orientation='h',
                size=(40, 15),
                key='-RESIZE SLIDER-',
                enable_events=True
            )
        ],
        [
            sg.Checkbox(
                'blur',
                size=(10, 1),
                key='-BLUR-',
                enable_events=True
            ),
            sg.Slider(
                (1, 10),
                1,
                1,
                orientation='h',
                size=(40, 15),
                key='-BLUR SLIDER-',
                enable_events=True
            )
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
                'Show Original Video'
            ),
            sg.Checkbox(
                key='-ORIGINAL-',
                size=(8, 1)
            )
        ],
        [
            sg.Text(
                'Show Mask Video'
            ),
            sg.Checkbox(
                key='-MASKING-',
                size=(8, 1),
              default=True
            )
        ],
        [
            sg.Text(
                'Show Display'
            ),
            sg.Checkbox(
                key='-DISPLAY-',
                size=(8, 1)
            )
        ],
        [
            sg.Checkbox(
                "Blue",
                size=(10, 1),
                default=True,
                key='-BLUE_MASK-',
                enable_events=True
            ),
            sg.Checkbox(
                "Green",
                size=(10, 1),
                default=True,
                key='-GREEN_MASK-',
                enable_events=True
            ),
            sg.Checkbox(
                "Red",
                size=(10, 1),
                default=True,
                key='-RED_MASK-',
                enable_events=True
            )
        ],
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

    
    layout = [
        [
            sg.Text("Timeline", size=(8,1)),
            sg.Slider(
                orientation="horizontal",
                key="-PROGRESS SLIDER-",
                range=(1,num_frames),
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
        [sg.Button('Clear'), sg.Button('Close', key="Exit")]
    ]

    window = sg.Window("Results", layout)


    try:
        while True:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            
            ret, frame = vidFile.read()
            if not ret:  # if out of data stop looping
                break

            # Move Slider
            if event == '-PROGRESS SLIDER-':
                #Set the frame count to the progress bar
                frame_count = int(values['-PROGRESS SLIDER-'])
                vidFile.set(cv.CAP_PROP_POS_FRAMES, frame_count)
                if values['-PROGRESS SLIDER-'] > values['-END FRAME SLIDER-']:
                    window['-END FRAME SLIDER-'].update(
                        values['-PROGRESS SLIDER-'])

            if event == '<<<':
                    frame_count = np.maximum(0, frame_count - 150)
                    window['-PROGRESS SLIDER-'].update(frame_count)
                    vidFile.set(cv.CAP_PROP_POS_FRAMES, frame_count)

            if event == '<<':
                frame_count = np.maximum(0, frame_count - 30)
                window['-PROGRESS SLIDER-'].update(frame_count)
                vidFile.set(cv.CAP_PROP_POS_FRAMES, frame_count)

            if event == '<':
                frame_count = np.maximum(0, frame_count - 1)
                window['-PROGRESS SLIDER-'].update(frame_count)
                vidFile.set(cv.CAP_PROP_POS_FRAMES, frame_count)

            if event == '>':
                frame_count = frame_count + 1
                window['-PROGRESS SLIDER-'].update(frame_count)
                vidFile.set(cv.CAP_PROP_POS_FRAMES, frame_count)

            if event == '>>':
                frame_count = frame_count + 30
                window['-PROGRESS SLIDER-'].update(frame_count)
                vidFile.set(cv.CAP_PROP_POS_FRAMES, frame_count)

            if event == '>>>':
                frame_count = frame_count + 150
                window['-PROGRESS SLIDER-'].update(frame_count)
                vidFile.set(cv.CAP_PROP_POS_FRAMES, frame_count)

            #If the counter exceeds the end frame, restart from the start frame
            if frame_count >= e_frame:
                vidFile.set(cv.CAP_PROP_POS_FRAMES, s_frame)
                frame_count = s_frame
                window['-PROGRESS SLIDER-'].update(frame_count)
                continue
        
            if event == 'Play / Stop':
                    stop_flg = not stop_flg

            # if(
            #     (
            #         stop_flg
            #         and event == "__TIMEOUT__"
            #         and mouse_flg is False
            #     )
            # ):
            #     window['-PROGRESS SLIDER-'].update(frame_count)
            #     continue

            #Load frame
            ret, frame = vidFile.read()
            valid_frame = int(frame_count - s_frame)
            #Self when the last frame is over.s_Resume from frame
            if not ret:
                vidFile.set(cv.CAP_PROP_POS_FRAMES, s_frame)
                frame_count = s_frame
                continue

            # if values['-MASKING-']:
            #     # Masks
            #     #Display image
            #         cv.imshow("Movie", frame)
            #         if values['-MASKING-']:
            #             cv.imshow("Mask", cv.cvtColor(mask, cv.COLOR_GRAY2BGR))
            #             cv.setWindowProperty("Mask", cv.WND_PROP_VISIBLE, 0)
            #         elif not values['-MASKING-'] and cv.getWindowProperty("Mask", cv.WND_PROP_VISIBLE):
            #             cv.destroyWindow("Mask")

            #         if stop_flg:
            #             vidFile.set(cv.CAP_PROP_POS_FRAMES, frame_count)

            #         else:
            #             frame_count += 1
            #             window['-PROGRESS SLIDER-'].update(frame_count + 1)

            #         #Other processing###############################################
            #         #Clear log window
            #         if event == 'Clear':
            #             window['-OUTPUT-'].update('')
            
            # if values['-DISPLAY-']:
            #   # Display screen recording with projection
            #   pass
            
            # if values['-ORIGINAL']:
            #   pass

    finally:
        cv.destroyWindow("Movie")
        cv.destroyWindow("Mask")
        vidFile.release()
        window.close()
