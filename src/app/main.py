import PySimpleGUI as sg
from results import resultsWindow
from new_project import createFolder

# Main window
def main():
    sg.theme('SystemDefaultForReal') # Set Theme for PySimpleGUI
    
    # Add Layout
    layout = [
        [
            sg.Text("Project Eye Tracking"), 
            sg.Button("New Project", key=("new_project")), 
            sg.Button("Open Project", key=("open_project")), 
        ],
        [
            sg.HorizontalSeparator()
        ],
        [
            sg.Text("Instructions:")
        ],
        [
            sg.Multiline("Stap 1: klik op maak nieuw project. \nStap 2: Vul de gewenste project naam in. \nStap 3: Klik op calibreer, kijk naar de groene cirkels. Druk op de toets a wanneer je ernaar kijkt (meerdere klik soms nodig). \nStap 4: Wanneer de calibratie voltooid is druk op het de start knop om het experiment te starten. \nStap 5: Kijk naar de foto's zonder je blik te forceren.", size=(450,500), key='textbox', font="Helvetica 14")
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

    window.close()

if __name__ == "__main__":
    main()