# Import the library tkinter
from tkinter import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
import torch
from skimage import transform
from imageio import imread
import numpy as np
import joblib
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Create a GUI app
root = tk.Tk()
# Adjust size
root.geometry("900x636")
# Give title to your GUI app
root.title("Pneumonia Detection App")
root.config(bg='white')
# Add image file
bg = PhotoImage(file="background.png")
img = PhotoImage(file="background.png")
label = Label(
    root,
    image=img
)
label.place(x=0, y=0)
# Create Canvas
canvas1 = Canvas(root, width=960,
                 height=480)

canvas1.pack(fill="both", expand=True)
# Display image
canvas1.create_image(0, 0, image=bg,
                     anchor="nw")
# Heading of project
canvas1.create_text(
    460, 80, text="PNEUMONIA DETECTOR", font=("Arial", 36), fill="#39FF14")
canvas1.create_text(
    460, 120, text="A ML BASED APPLICATION", font=("Arial", 18), fill="#39FF14")

label_training_acc = Label(root,
                           text="Selected Image Path",
                           width=20, height=2,
                           fg="blue")
label_testing_acc = Label(root,
                          text="Selected Image Path",
                          width=20, height=2,
                          fg="blue")
canvas1.create_text(
    140, 215, text="Training accuracy:", font=("Arial", 16), fill="#39FF14")
label_training_acc.configure(text="86.37%")
canvas1.create_text(
    140, 255, text="Testing accuracy: ", font=("Arial", 16), fill="#39FF14")
label_testing_acc.configure(text="73.12%")

# function to open windows explorer

label_file_explorer = Label(root,
                            text="Selected Image Path",
                            width=60, height=2,
                            fg="blue")


def open():
    global selectedFile
    f_types = [('JPEG Files', '.jpeg'),
               ('PNG Files', '.png'), ('JPG Files', '.jpg')]
    selectedFile = filedialog.askopenfilename(
        initialdir='/', title="Select A File", filetypes=f_types)
    label_file_explorer.configure(text="File Opened: "+selectedFile)


# Creating button to select image
button = Button(
    root,
    text='Select Your Sample Here',
    relief=RAISED,
    font=('Arial', 14),
    command=lambda: open()
)

# positioning the button
button.place(x=50, y=300)
# positioning the path descriptor
label_file_explorer.place(x=300, y=300)
label_training_acc.place(x=240, y=200)
label_testing_acc.place(x=240, y=240)
# Creating button to convert into raw
label_predicted = Label(root,
                        text="Selected Image Path",
                        width=20, height=2,
                        fg="blue")
label_predicted.place(x=50, y=420)
label_predicted.configure(text="normal/pneumonia")
# predict function here


def predicted_case():
    global selectedFile

    def img_resizer(img, target_shape):
        resized = transform.resize(
            np.float32(img) / 255, target_shape,
            order=2,
            anti_aliasing=True,
            clip=True
        )
        return np.float32(resized)
    device = torch.device('cpu')
    model = torch.load('weights/cnn.pth',
                       map_location=device).feature_extractor.to(device) #enter path and filename of saved weights from train_cnn
    with torch.no_grad():
        model.eval()
        dataset = []
        print(selectedFile)
        img = imread(selectedFile)
        resized = img_resizer(img, (256, 256, 3))
        resized = resized.reshape((1, 3, 256, 256))
        inp = torch.from_numpy(resized).to(device)
        features = model.forward(inp).cpu().numpy()
        features = np.concatenate((features[0], np.array([0])), axis=0)
        dataset.append(features)
    dataset = np.asarray(dataset)
    model_path = 'weights\svc.sav' #enter path and filename of saved weights from training_svm
    X, y = dataset[:, :-1], np.uint8(dataset[:, -1])
    loaded_model = joblib.load(model_path)
    y_pred = loaded_model.predict(X)
    if(y_pred >= 0.5):
        label_predicted.configure(text="Predicted as normal")
    else:
        label_predicted.configure(text="Predicted as pneumonia")


button1 = Button(
    root,
    text='DETECT',
    relief=RAISED,
    font=('Arial', 14),
    command=lambda: predicted_case()
)
# positioning the button
button1.place(x=50, y=360)

# Make the loop for displaying app
root.mainloop()
