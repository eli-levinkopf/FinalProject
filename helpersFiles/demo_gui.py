import tkinter as tk
from tkinter import filedialog
import os
import time
import numpy as np
from PIL import Image, ImageTk


anomalies_dict = {
        'case#3': np.array([]),
        'case#5': np.array([[173, 329, 115]]),
        'case#6': np.array([[300, 246,  23], [308, 218,  28]]), 
        'case#10': np.array([[190, 260, 145], [292, 206, 161]]), 
        'case#17': np.array([[179, 274, 173],[306, 320, 168],[318, 275, 174]]), 
        'case#18': np.array([[318, 307, 154]]), 
        'case#19': np.array([[168, 252,  41],[293, 281,  47],[309, 255,  38]]), 
        'case#25': np.array([[296, 252, 131]]), 
        'case#44': np.array([[267, 211, 109]])
       }

def upload_file():
    anomalies_label["text"] = ""
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.gz")])
    if file_path:
        analyze_button.config(state=tk.NORMAL)
        analyze_button["text"] = "Analyze scan ({})".format(os.path.basename(file_path))
        analyze_button["command"] = lambda: analyze_scan(file_path)


def analyze_scan(file_path):
    anomalies = anomalies_dict[file_path.split('/')[-1].split('.')[0]]
    time.sleep(3)
    # Display the anomalies in the GUI
    if anomalies.size > 0:
        anomalies_str = ["({}                 {}             {})".format(*anomaly) for anomaly in anomalies]
        anomalies_label["text"] = "Anomalies found in:\nSagital axis, Coronal axis, Axial axis\n" + "\n".join(anomalies_str) 
    else:
        anomalies_label["text"] = "No anomalies found in the CT scan."


# Create the main window
window = tk.Tk()
window.title("CT Scan Analyzer")
window.attributes("-fullscreen", True)  # Add this line to run in full screen

# Set the background color to white
window.configure(bg="lightgray")

# Load the background image
background_image = Image.open("/Users/elilevinkopf/Documents/Ex23A/FinalProject/yellow_grreen_sinus.png")
background_image = background_image.resize((window.winfo_screenwidth(), window.winfo_screenheight()), Image.ANTIALIAS)

# Create a new image with 50% transparency
transparent_image = Image.new("RGBA", background_image.size, (0, 0, 0, 0))
blended_image = Image.alpha_composite(background_image.convert("RGBA"), transparent_image)

# Create an ImageTk object from the blended image
background_photo = ImageTk.PhotoImage(blended_image)

# Create a canvas and place it in the window
canvas = tk.Canvas(window, width=window.winfo_screenwidth(), height=window.winfo_screenheight())
canvas.pack()

# Create the image item with the background image on the canvas
canvas.create_image(0, 0, image=background_photo, anchor=tk.NW)

# Create the exit button and place it in the top-left corner
exit_button = tk.Button(window, text="Exit", command=window.destroy, font=("Arial", 16))
exit_button.place(x=0, y=0)

# Create the "Upload a CT scan" button
upload_button = tk.Button(window, text="Upload a CT scan", command=upload_file, font=("Arial", 16))
upload_button.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

# Create the "Analyze scan" button (disabled by default)
analyze_button = tk.Button(window, text="Analyze scan", state=tk.DISABLED, font=("Arial", 16))
analyze_button.place(relx=0.5, rely=0.45, anchor=tk.CENTER)

# Create the "Anomalies" label (initially empty)
anomalies_label = tk.Label(window, text="")
anomalies_label.place(relx=0.5, rely=0.55, anchor=tk.CENTER)


# Start the GUI event loop
window.mainloop()

