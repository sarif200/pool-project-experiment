import PySimpleGUI as sg
from results import show_results

# Main window
def main():
    layout = [
        [sg.Text('This window is empty'), sg.Button("Show Results", key=("show_results"))],
    ]
    window = sg.Window("Main Window", layout, size=(800, 600)).Finalize()

    while True:
        event, values = window.read()
        if  event == sg.WIN_CLOSED:
            break
        if event == "show_results":
            show_results()

    window.close()

if __name__ == "__main__":
    main()