import PySimpleGUI as sg
from results import resultsWindow
from calibration import calibration
from new_project import createFolder
from settings import SettingsWindow

# Main window
def main():
    sg.theme('SystemDefaultForReal') # Set Theme for PySimpleGUI
    
    # Add Layout
    layout = [
        [
            sg.Text("Project Eye Tracking"), 
            sg.Button("New Project", key=("new_project")), 
            sg.Button("Open Project", key=("open_project")), 
            sg.Button("Settings", key=("settings"))
        ],
        [
            sg.HorizontalSeparator()
        ],
        [
            sg.Text("Instructions:")
        ],
        [
            sg.Multiline("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.", size=(450,500), key='textbox')
        ]
    ]
    
    # Create window & evenloop
    window = sg.Window("Main Window", layout, size=(500, 550)).Finalize()

    while True:
        event, values = window.read()
        if  event == sg.WIN_CLOSED:
            break
        if event == "open_project":
            resultsWindow()
        if event == "new_project":
            createFolder()
        if event == "settings":
            SettingsWindow()

    window.close()

if __name__ == "__main__":
    main()