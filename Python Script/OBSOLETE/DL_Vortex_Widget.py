import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from tensorflow.keras.models import load_model
import skimage as ski
import math
from PIL import Image
import pandas as pd

root = tk.Tk()
root.geometry('800x600')
root.title('Deep Learning Segmentation Widget')

# *** creating title for the widget
label = tk.Label(root, text='Deep Learning Segmentation Widget', font=('Arial', 15))


# *** fuction that disables and enables button
def disable(button):
    button['state'] = 'disabled'
def enable(button):
    button['state'] = 'normal'

# *** loads the models
def loadModel():
    global ringModel
    global fixModel
    ringModel = load_model("W:\Advanced Engineering\Vortex Ballistics\~ACTIVE\Python Script\models\Ring Mask- 30 Epochs- 561 img.h5")
    fixModel = load_model("W:\Advanced Engineering\Vortex Ballistics\~ACTIVE\Python Script\models\model 10-27_Fixture Mask-105 epochs-582 images\95.h5")
loadModel()

# *** function that controls action on select file button
def selectImage(label):
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select an Image File",
                                          filetypes = (("Image files",
                                                        "*.jpg*"),
                                                       ("all files",
                                                        "*.*")))
    # Change label contents
    label.configure(text="File Opened: "+filename.split('/')[-1])
    global dirSelect
    dirSelect = False

# *** function that controls action on select directory button
def selectDir(label):
    global filename
    filename = filedialog.askdirectory(initialdir = "/",
                                          title = "Select a Directory to Analyze All Files In")
    # Change label contents
    label.configure(text="Directory Selected: "+filename.split('/')[-1])
    global dirSelect
    dirSelect = True

# *** creates label and button relevant to image file Exploration
label_img_explorer = tk.Label(root, 
                            text = "Select an Image File for Segmentation",
                            #width = 100, height = 4, 
                            fg = "blue")
button_explore_img = tk.Button(root, 
                        text = "Browse Files",
                        command = lambda: [selectImage(label_in_text), enable(button_outDir)]) 

# *** create label and button relevant to full directory segmentation
label_dir_explorer = tk.Label(root, 
                            text = "Select a Directory for Segmentation",
                            #width = 100, height = 4, 
                            fg = "blue")
button_explore_dir = tk.Button(root, 
                        text = "Browse Files",
                        command = lambda: [selectDir(label_in_text), enable(button_outDir)]) 
#button_exit = tk.Button(root, 
#                     text = "Exit",
#                     command = exit)


#button_exit.grid(column = 1,row = 3)
label_in_text = tk.Label(root, 
                            #width = 100, height = 4, 
                            fg = "blue")
label_out_text = tk.Label(root, 
                            #width = 100, height = 4, 
                            fg = "blue")




# *** creates analyze function
def analyze():
    if not os.path.exists(outDir + '/Full Thickness Data'):
        os.makedirs(outDir + '/Full Thickness Data')
    if not os.path.exists(outDir + '/Segmented Image'):
        os.makedirs(outDir + '/Segmented Image')
    if not os.path.exists(outDir + '/Radial Thickness Plot'):
        os.makedirs(outDir + '/Radial Thickness Plot')
    if not os.path.exists(outDir + '/Radial Thickness Overlay'):
        os.makedirs(outDir + '/Radial Thickness Overlay')
    if not os.path.exists(outDir + '/Thickness Overlay Statistics'):
        os.makedirs(outDir + '/Thickness Overlay Statistics')
    if dirSelect:
        dirPath = filename
        try:
            images = os.listdir(dirPath)
            for image in images:
                try:
                    Image.open(dirPath + '/' + image)
                    imageAnalyzer(dirPath + '/' + image)
                except Exception as e:
                    continue
        except Exception as e:
            error_label.configure(text= e)
    else:
        imagePath = filename
        imageAnalyzer(imagePath)

# *** helper method for analysis
def imageAnalyzer(path):
    #***
    imgName = (path.split('/')[-1]).split('.')[0]
    imgColor = cv2.imread(path)
    imgColor = cv2.resize(imgColor, (576, 384))
    #addGraph(imgColor, 5, 0, title='Original Image')
    #***
    image = cv2.imread(path, 0)
    image = cv2.resize(image, (576, 384))
    image = np.expand_dims(image, axis=0)
    image = image/255
    ringMask, fixMask = predict(image)
    ringMask = np.bitwise_and(ringMask, np.bitwise_xor(ringMask, fixMask))
    openingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    ringMask = cv2.morphologyEx(ringMask, cv2.MORPH_OPEN, openingKernel, iterations = 3)
    greenBack = np.full(imgColor.shape, 0, dtype=np.uint8)
    greenBack[:,:,1] = 255
    freezeGreen = cv2.bitwise_or(greenBack, greenBack, mask=ringMask)
    #***
    redBack = np.full(imgColor.shape, 0, dtype=np.uint8)
    redBack[:,:,0] = 255
    fixRed = cv2.bitwise_or(redBack, redBack, mask=fixMask)
    overlay = cv2.addWeighted(imgColor, 2/3, freezeGreen, 1/3, gamma = 0)
    segment = cv2.addWeighted(overlay, 0.75, fixRed, 0.25, gamma = 0)
    #addGraph(segment, 5, 1, title='Segmented Image')
    cv2.imwrite(outDir + '/Segmented Image/' + imgName + '.jpg', segment)
    #***
    pixWidth = max(fixMask.sum(axis=1)) / 255
    mm2pix = 34/pixWidth
    radPlotPic = radialGraphPic(ringMask, mm2pix, 3, imgName)
    addGraph(radPlotPic, 6, 0, title='Radial Plot')
    cv2.imwrite(outDir + '/Radial Thickness Plot/' + imgName + '.jpg', radPlotPic)
    full, section = plot_overlay(overlay.copy(), radPlotPic.copy(), ringMask)
    addGraph(full, 6, 1, title='Radial Plot Overlay')
    cv2.imwrite(outDir + '/Radial Thickness Overlay/' + imgName + '.jpg', full)

# *** uses the models to develop the ring and fixture segmentations
def predict(img):
    ring = ringModel.predict(img)
    ring = ring.reshape(img.shape[1:3])
    ringMask = ring
    fix = fixModel.predict(img)
    fix = fix.reshape(img.shape[1:3])
    openingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fix = cv2.morphologyEx(fix, cv2.MORPH_OPEN, openingKernel, iterations = 3)
    fixMask = fix
    ringMask = maskThresh(ringMask)
    fixMask = maskThresh(fixMask)
    return ringMask, fixMask

# *** uses masks and histograms to threshold and convert between 0-1
def maskThresh(mask):
    histoGray = ski.util.img_as_float(mask)
    vals, counts = np.unique(histoGray, return_counts = True)
    colors = vals[counts > 50]
    thresh = cv2.inRange(mask*255, np.round(colors[-1]*255-50), 255)
    return thresh

# *** uses y-values to create polar plot as image
def radialGraphPic(mask, mm2pix, samplingRate, imgName):
    thicknessProfile, xShort, maxRad = thicknessGraph(mask, samplingRate)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.tight_layout(pad=0)
    x, y = zip(*thicknessProfile)
    x = np.array(x)
    y = np.array(y)*mm2pix
    x = 360-x
    rads = x.astype('float') * np.pi/180
    ax.plot(rads, y, color='red')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    if xShort:
        ratio = maxRad*2/w
        graphPic = cv2.resize(im, (maxRad*2, int(ratio*int(h))))
    else:
        ratio = maxRad*2/h
        graphPic = cv2.resize(im, (int(ratio*w), maxRad*2))

    thickVals = list(list(zip(*thicknessProfile))[1])
    big = max(thickVals)*mm2pix
    tiny = min(thickVals)*mm2pix
    rng = big-tiny
    avg = np.mean(thickVals)*mm2pix
    std = np.std(thickVals)*mm2pix
    med = np.median(thickVals)*mm2pix
    statsArr = ['Max Thickness: ' + str(big), 
                'Min Thickness: ' + str(tiny),
                'Range of Thicknesses: ' + str(rng),
                'Median Thickness: '+ str(med),
                'Average Thickness: ' + str(avg),
                'Standard Deviation of Thicknesses: ' + str(std)]
    with open(outDir + '/Thickness Overlay Statistics/' + imgName + '.txt', 'w') as f:
        for line in statsArr:
            f.write(line)
            f.write('\n')

    df = pd.DataFrame(thicknessProfile)
    df = df.rename(columns={0: 'Radians(Counterclockwise from positive x-axis)', 1: 'Thickness(mm)'})
    df.to_csv(outDir + '/Full Thickness Data/' + imgName + '.csv', index=False)

    return graphPic 

# *** creates the y-values in the thickness 
def thicknessGraph(mask, samplingRate = 3):
    cX, cY = findCenter(mask)
    print(cX, cY)
    distCenter = [cX, cY, mask.shape[0] - cY, mask.shape[1] - cX]
    maxRad = min(distCenter)

    if distCenter.index(maxRad) == 0 or distCenter.index(maxRad) == 3:
        xShort = True
    else:
        xShort = False

    startX = cX
    startY = cY
    lineLength = maxRad

    thicknessProfile = []

    for j in range(360*samplingRate):
        i = j/samplingRate
        lineMask = np.zeros(mask.shape)
        endX = math.floor(cX + math.cos(math.radians(i))*lineLength)
        endY = math.floor(cY + math.sin(math.radians(i))*lineLength)
        cv2.line(lineMask, (startX, startY), (endX, endY), 255, 10)
        currSlice = np.bitwise_and((np.ceil(mask/255)).astype(int), (np.ceil(lineMask/255)).astype(int))
        count = (np.count_nonzero(currSlice) / np.count_nonzero(lineMask)) * lineLength
        thicknessProfile.append((i,count))

    return thicknessProfile, xShort, maxRad

#rakes ringMask and does centroid calculations to find the center
def findCenter(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inCircle = np.zeros(mask.shape)

    outerRingCount = 0
    innerRingCount = 0
    innerRingIdx = -1
    for i in range(len(contours)):
        if (len(contours[i]) > outerRingCount):
            outerRingCount = len(contours[i])
        elif (len(contours[i]) > innerRingCount):
            innerRingCount = len(contours[i])
            innerRingIdx = i

    cv2.drawContours(inCircle, contours,innerRingIdx,255,-1)        
    M = cv2.moments(inCircle)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

# *** plots anything over the original image
def plot_overlay(back, fore, mask):
    #calculates the section of the background where the foreground image is going to be placed
    #this is done with the centroid and the length and width of the foreground image
    cX, cY = findCenter(mask)
    xOff = cX - int(fore.shape[1]/2)
    yOff = cY - int(fore.shape[0]/2)
    roi = back[yOff:yOff+fore.shape[0], xOff:xOff+fore.shape[1]]

    #converts the graph to a black and white copy. This part is hard coded because we know that the background is white
    #and anything that is non-white must belong to the graph
    #for other images, this threshold needs to be changed especially if this code is used for something than what is stated above
    back2gray = cv2.cvtColor(fore, cv2.COLOR_RGB2GRAY)
    back2gray = cv2.threshold(back2gray, 245, 255, cv2.THRESH_BINARY)[1]
    maskBack = cv2.bitwise_not(back2gray)
    whiteBack = np.full(fore.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(whiteBack, whiteBack, mask = maskBack)

    #uses the mask from above to place the foreground over hte background in the region of interest
    fg = cv2.bitwise_or(fore, fore, mask=maskBack)
    finalRoi =  cv2.bitwise_or(roi, fg)

    #replaces the region of interest in the original image with the updated one 
    back[yOff:yOff+fore.shape[0], xOff:xOff+fore.shape[1]] = finalRoi
    back = cv2.resize(back, (back.shape[1]*2, back.shape[0]*2))
    return back, finalRoi






# *** method to add images to the screen
def addGraph(img, r, c, title):
    f = Figure(figsize=(3, 5))
    a = f.add_subplot(111)
    a.set_title(title)
    a.imshow(img)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(column = c, row = r)
    canvas._tkcanvas.grid(column = c, row = r)

# *** label to display errors
error_label = tk.Label(root, fg = "red")


# *** create analysis button
button_analyze = tk.Button(root, 
                        text = "Analyze",
                        command = lambda: [analyze(), enable(button_explore_img), enable(button_explore_dir), disable(button_analyze)],
                        state = 'disabled') 


def selectOutDir(label):
    global outDir
    filename = filedialog.askdirectory(initialdir = "/",
                                          title = "Select a Directory to Analyze All Files In")
    # Change label contents
    label.configure(text="Directory Selected: "+filename.split('/')[-1])
    outDir = filename


label_outDir_explorer = tk.Label(root, 
                            text = "Select a Directory for Segmentation Outputs",
                            #width = 100, height = 4, 
                            fg = "blue")
# *** create button for the output directory
button_outDir = tk.Button(root,
                          text = 'Browse',
                          command = lambda:[selectOutDir(label_out_text), enable(button_analyze)],
                          state = 'disabled')



label.grid(row=0, column=1, padx= 10, pady=10)
label_img_explorer.grid(row = 1, column = 1)
button_explore_img.grid(row = 2, column = 1)
label_dir_explorer.grid(row = 1, column = 2)
button_explore_dir.grid(row = 2, column = 2)
label_outDir_explorer.grid(row = 3, column = 1)
button_outDir.grid(row = 4, column = 1)
button_analyze.grid(row = 3, column=2)
label_in_text.grid(row=5, column=1)
label_out_text.grid(row=5, column=2)
error_label.grid(row = 4, column =2)


root.mainloop()