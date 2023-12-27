from PyQt6 import QtCore
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QFont
import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from tensorflow.keras.models import load_model
import skimage as ski
import math
from PIL import Image
import pandas as pd

def main():
    app = QApplication([])
    window = QWidget()
    window.setGeometry(100, 100, 500, 500)
    window.setWindowTitle("Deep Learning GUI")

    global outerLayout
    outerLayout = QVBoxLayout()

    global ringModel
    global fixModel
    ringModel = load_model("W:\Advanced Engineering\Vortex Ballistics\~ACTIVE\Python Script\models\Ring Mask- 30 Epochs- 561 img.h5")
    fixModel = load_model("W:\Advanced Engineering\Vortex Ballistics\~ACTIVE\Python Script\models\model 10-27_Fixture Mask-105 epochs-582 images\95.h5")

    title = QLabel("Deep Learning Segmentation for Images")
    title.setFont(QFont("Arial", 16))
    outerLayout.addWidget(title, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)


    dirLayout = QGridLayout()
    dirLabel = QLabel("Select a Directory for Segmentation of All Images (Analyze Multiple images in a directory at once)")
    dirButton = QPushButton("Select a Directory")
    dirButton.clicked.connect(lambda: [selectDir(dirLabel), enable(outButton)])

    fileLabel = QLabel("Select an Image (Analyze a Single Image)")
    fileButton = QPushButton("Select an Image(s)")
    fileButton.clicked.connect(lambda: [selectImg(fileLabel), enable(outButton)])

    dirLayout.addWidget(dirLabel, 0,0, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
    dirLayout.addWidget(fileLabel, 0,1, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
    dirLayout.addWidget(dirButton, 1,0, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
    dirLayout.addWidget(fileButton, 1,1, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
    dirLayout.setHorizontalSpacing(25)
    outerLayout.addLayout(dirLayout)

    outLabel = QLabel("Select a Directory to Save Outputs")
    outButton = QPushButton("Select an Output Directory")
    outButton.clicked.connect(lambda: [selectOutDir(outLabel), enable(analyzeButton)])
    outerLayout.addWidget(outLabel, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
    outerLayout.addWidget(outButton, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

    analyzeButton = QPushButton("Analyze")
    analyzeButton.clicked.connect(lambda: [analyze(outerLayout), enable(dirButton), enable(fileButton), disable(outButton)])
    outerLayout.addWidget(analyzeButton, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

    window.setLayout(outerLayout)
    window.show()
    app.exec()



def disable(button: QPushButton):
    button.setEnabled(False)
def enable(button: QPushButton):
    button.setEnabled(True)

def selectDir(label: QLabel):
    global filename
    filename = QFileDialog.getExistingDirectory(caption = "Select Directory")
    label.setText("Directory Selected: " + filename.split('/')[-1])
    global dirSelect
    dirSelect = True

def selectOutDir(label: QLabel):
    global outDir
    filename = QFileDialog.getExistingDirectory(caption = "Select Directory")
    # Change label contents
    label.setText("Directory Selected: "+filename.split('/')[-1])
    outDir = filename
    global dirSelect
    dirSelect = False


def selectImg(label: QLabel):
    global filename
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    dialog.setNameFilter("(*.png *.jpg)")
    dialog.setViewMode(QFileDialog.ViewMode.List)
    if dialog.exec():
        filename = dialog.selectedFiles()
    label.setText("Image Selected: " + filename[0].split('/')[-1])

def analyze(layout):
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
            error_label = QLabel()
            error_label.setText(e)
            layout.addWidget(error_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

    else:
        for image in filename:
            imageAnalyzer(image)

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
    canvas = FigureCanvasQTAgg(f)
    canvas.draw()
    outerLayout.addWidget(canvas)
    #canvas._tkcanvas.grid(column = c, row = r)


if __name__ == '__main__':
    main()