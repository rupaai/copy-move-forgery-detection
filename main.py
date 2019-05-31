from tkinter import *
from PIL import ImageTk
from PIL import Image
root = Tk()
from tkinter import filedialog
from cmf_testing.cmf_testing import LocalBinaryPatterns
import numpy as np
# global panelA, panelB
import cv2
import matplotlib.pyplot as plt
# open a file chooser dialog and allow the user to select an input
# image

def lbp():
    path = filedialog.askopenfilename()
    edged = None
    image = None
    panelA = None
    panelB = None
    multi = LocalBinaryPatterns(24, 8)
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        print(image.shape)
        his = multi.describe(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        plt.plot(his)
        plt.show()
        his = np.reshape(his, (-1, 2))
        hist = Image.fromarray(his)
        hist = ImageTk.PhotoImage(hist)


        panelA = Label(image=image)
        panelB = Label(image=hist)
    if panelA is None or panelB is None:
        # the first panel will store our original image

        panelA.image = image
        panelA.pack(side="left", padx=10, pady=10)

        # while the second panel will store the edge map

        panelB.image = edged
        panelB.pack(side="right", padx=10, pady=10)

        # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image)
        panelB.configure(image=edged)
        panelA.image = image
        panelB.image = edged
    panelA.pack(side=LEFT)
    panelB.pack(side=RIGHT)
def select_image():
    path = filedialog.askopenfilename()
    edged = None
    image = None
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)

        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)
    panelA = Label(image=image)
    panelB = Label(image=edged)
    if panelA is None or panelB is None:
        # the first panel will store our original image

        panelA.image = image
        panelA.pack(side="left", padx=10, pady=10)

        # while the second panel will store the edge map

        panelB.image = edged
        panelB.pack(side="right", padx=10, pady=10)

        # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image)
        panelB.configure(image=edged)
        panelA.image = image
        panelB.image = edged
    panelA.pack(side=LEFT)
    panelB.pack(side=RIGHT)
btn = Button(root, text="Select an image", command=lbp)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")


root.mainloop()