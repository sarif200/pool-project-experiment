# Import libraries
import PySimpleGUI as sg

# Open New Window
def show_results():
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
            sg.Radio(
                'Rectangle',
                "RADIO2",
                key='-RECTANGLE_MASK-',
                default=True,
                size=(8, 1)
            ),
            sg.Radio(
                'Masking',
                "RADIO2",
                key='-MASKING-',
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
            sg.Text("Start", size=(8,1)),
            sg.Slider(
                orientation="horizontal",
                key="timeSlider",
                range=(1,100),
                size=(45,15),
                default_value=0
            )
        ],
        [
            sg.Text("Speed", size=(6,1)),
            sg.Slider(
                orientation="horizontal",
                key="speedSlider",
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

    window = sg.Window("Results", layout, location=(0,0))

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break