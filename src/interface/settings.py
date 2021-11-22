import PySimpleGUI as sg

def SettingsWindow():

    camera_resolution = ['480p', '720p', '1080p', '4K']
    camera_frame_rate = ['10 FPS','30 FPS', '60 FPS', '120 FPS']

    layout = [
        [
            sg.Text('Settings'), 
            sg.Button('Close', key=("Exit"))
        ],
        [
            sg.HorizontalSeparator()
        ],
        [
           sg.Text('Camera Resolution:'), sg.Combo(values = camera_resolution)
        ],
        [
            sg.Text('Camera Frame Rate:'), sg.Combo(values = camera_frame_rate)
        ],
        [
            sg.Text('Open Live Tracking Window (Dev mode)'), sg.Checkbox('Dev_mode', key='-dev_mode-')
        ]
    ]

    window = sg.Window("Results", layout)

    while True:
        event, values = window.read()
        print(event, values)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()