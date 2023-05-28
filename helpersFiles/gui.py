import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import subprocess
import os
import nibabel as nib
import paramiko
import time
import numpy as np
from PIL import Image, ImageTk
import helpers.preProcess as preProcess
import helpers.detection as detection

# Command to run the inference using nnUNet_predict tool
RLOGIN_COMMAND = 'rlogin phoenix'
MOVE_TO_501_COMMAND = 'cd ../../ep/501 && '
BATCH_COMMAND = 'sbatch detector.sh'
SOURCE_PATH = '/cs/ep/501/nnUNet_raw_data_base/nnUNet_raw_data/Task509_sinus_bone_segmantation/imagesTs/sinus_bone_006_0000.nii.gz'
DEST_PATH =  '/Users/elilevinkopf/Documents/Ex23A/FinalProject/tmpFolder/sinus_bone_006_0000.nii.gz'

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.gz")])
    if file_path:
        analyze_button.config(state=tk.NORMAL)
        analyze_button["text"] = "Analyze scan ({})".format(os.path.basename(file_path))
        analyze_button["command"] = lambda: analyze_scan(file_path)

def analyze_scan(file_path):
    # Normalize and reshape the 3D scan
    scan = nib.load(file_path).get_fdata()
    normalized_scan = preProcess.reshape3DScan(scan)
    nib.save(nib.Nifti1Image(normalized_scan, None), '/Users/elilevinkopf/Documents/Ex23A/FinalProject/tmpFolder/sinus_bone_006_0000.nii.gz')

    # Get the username and password from the entry fields
    username = username_entry.get()
    password = password_entry.get()

    # Set up the SSH client
    ssh_river = paramiko.SSHClient()
    ssh_river.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_river.connect(hostname="river.cs.huji.ac.il", username=username, password=password)

    # Upload the temporary file to the "river" computer
    sftp = ssh_river.open_sftp()
    sftp.put(DEST_PATH, SOURCE_PATH)
    sftp.close()

    # Set up an SSH tunnel from the local machine to the first remote server (river)
    from sshtunnel import SSHTunnelForwarder
    tunnel = SSHTunnelForwarder(
        ("river.cs.huji.ac.il", 22),
        ssh_username=username,
        ssh_password=password,
        remote_bind_address=("phoenix", 22)
    )

    # Start the SSH tunnel
    tunnel.start()

    # Set up the SSH client for the second remote server (phoenix)
    ssh_phoenix = paramiko.SSHClient()
    ssh_phoenix.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_phoenix.connect(hostname="127.0.0.1", port=tunnel.local_bind_port, username=username, password=password)

    # Run the command on the remote shell
    stdin, stdout, stderr = ssh_phoenix.exec_command(MOVE_TO_501_COMMAND + BATCH_COMMAND)
    job_id = stdout.read().decode().split(' ')[-1]
    print(stdout.read().decode())

    # Wait for the job to complete or fail
    time.sleep(30)
    job_status = "PENDING"
    cur_job_id = ''
    while (job_status != "COMPLETED" and job_status != "FAILED") or job_id != cur_job_id:
        time.sleep(5)  # Wait for 5 seconds before checking again
        stdin, stdout, stderr = ssh_phoenix.exec_command("ssacct --format=JobID,State | grep -E 'detector.+ (COMPLETED|FAILED|CANCELLED|RUNNING)'")
        output = stdout.read().decode()
        if output:
            lines = output.strip().split("\n")
            last_line = lines[-1]
            cur_job_id = last_line.split()[0]
            job_status = last_line.split()[8] if job_id == cur_job_id else "PENDING"
            print(job_status)

    # Wait for nnUNet to write the inference file
    output = ""
    while output != "EXIST":
        stdin, stdout, stderr = ssh_river.exec_command('[ -f "/cs/ep/501/output_task509/sinus_bone_006.nii.gz" ] && echo "EXIST"')
        output = stdout.read().decode().strip()
        print(output)
        time.sleep(3)
    
    # Download the output file from the remote "river" computer to the local computer
    sftp.get(os.path.join('/cs/ep/501/output_task509/sinus_bone_006.nii.gz'), '/Users/elilevinkopf/Documents/Ex23A/FinalProject/tmpFolder/output/sinus_bone_006.nii.gz')
    ssh_phoenix.close()
    ssh_river.close()
    
    anomalies = detection.detect_anomalies('/Users/elilevinkopf/Documents/Ex23A/FinalProject/tmpFolder/output/sinus_bone_006.nii.gz')
    # Create the "Anomalies" label (initially empty)
    anomalies_label = tk.Label(window, text="")
    anomalies_label.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
    # Display the anomalies in the GUI
    if anomalies.size > 0:
        anomalies_str = ["({}, {}, {})".format(*anomaly) for anomaly in anomalies]
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
background_image = Image.open("yellow_grreen_sinus.png")
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

# Create the "Username" label and entry field
username_label = tk.Label(window, text="Username:", font=("Arial", 16))
username_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
username_entry = tk.Entry(window)
username_entry.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

# Create the "Password" label and entry field
password_label = tk.Label(window, text="Password:", font=("Arial", 16))
password_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
password_entry = tk.Entry(window, show="*")
password_entry.place(relx=0.5, rely=0.45, anchor=tk.CENTER)

# Create the "Upload a CT scan" button
upload_button = tk.Button(window, text="Upload a CT scan", command=upload_file, font=("Arial", 16))
upload_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Create the "Analyze scan" button (disabled by default)
analyze_button = tk.Button(window, text="Analyze scan", state=tk.DISABLED, font=("Arial", 16))
analyze_button.place(relx=0.5, rely=0.55, anchor=tk.CENTER)


# Start the GUI event loop
window.mainloop()
