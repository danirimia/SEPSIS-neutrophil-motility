#!/usr/bin/env python
#---------------------------------------
# Analysis of unstained neutrophils
# Author: Julianne
#---------------------------------------

### IMPORT ALL THE THINGS
#%matplotlib inline

import cv2 as cv2
from matplotlib import pyplot as plt
import numpy as np
import time as t
import pandas as pd
import pims
import trackpy as tp
import ipdb
import sys
import os
import av
from tqdm import tqdm

print "OpenCV Version : %s " % cv2.__version__

### ASSIGN VARIABLES
cellThresh = 0
netThresh = 0
redThresh = 0
cells = []
nets = []
ros = []
image_num = 0
history = 10
frames = []


video = sys.argv[1]
# videos = [os.path.join(video, s) for s in os.listdir(video)]
# videos = sorted(videos)
# videos = videos[0:24]

def half_show(name, frame):
    cv2.imshow(name, cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2)))

def pixel2micron(area):
    # pixels**2  * (microns* / pixels)**2 = microns**2
    area = area*((50./109)**2)
    return area

def calculate_cell_count(frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    blur = cv2.medianBlur(frame,7)
    blurdst = cv2.fastNlMeansDenoising(blur,10,10,7,21)
    # half_show('dst', dst)
    # half_show('blur', blur)
    # half_show('dstblur', dstblur)
    # half_show('blurdst', blurdst)
    # cv2.waitKey(0)
    grey = cv2.cvtColor(blurdst, cv2.COLOR_RGB2GRAY);
    circles = cv2.HoughCircles(grey,cv2.cv.CV_HOUGH_GRADIENT,1,20,param1=15,param2=5,minRadius=2,maxRadius=10)

    if circles is None:
        return (0,None)

    filteredcircles = []

    for i in circles[0,:]:
        if i[0]>150:
            filteredcircles.append(i)

    for i in filteredcircles:
       cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),3) # draw the outer circle
       cv2.circle(blurdst,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
       cv2.circle(blurdst,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

    numCells = len(filteredcircles)
    # half_show("preview", frame)
    # half_show("preview-blue", blurdst)
    # cv2.waitKey(100)
    return (numCells,circles)

def tracking(video):
    print "running tracking"
    pimsFrames = pims.Video(video, as_grey = True)
    cells = []
    track = []
    for frame in pimsFrames[:]:
        f = tp.locate(frame, 301, invert=False, minmass = 2000)
        t = tp.link_df(f, 5) #remember cells after they left frame
        tp.annotate(f, frame)
        cells += f
        track += t
        print t.head()
    tp.plot_traj(t)
    return t.head()

def heatmap(circles,image):
    print "running heatmap"
    #max values: [179,255,255]
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    r = 15
    for point in circles:
        tl = [point[0]-r,point[1]-r]
        br = [point[0]+r,point[1]+r]
        print "point value is: %s" %(point)
        for i in range(tl[0],br[0]):
            for j in range(tl[1],br[1]):
                value = hsv[i][j]
                if value == [0,100,0]:
                    value = [102,235,26]
                    #hsv = cv2.cvtColor(value,cv2.COLOR_RGB2HSV)
                else:
                    hsv[0]-= n #some number

    return hsv


def process_video(video):
    cap = cv2.VideoCapture(video)
    cap.open(video)
    cur_frame_idx = 0
    cellcounts = []
    cellareacounts = []
    circs = []
    height = 550
    width = 1200
    blank = np.zeros((height, width, 3), np.uint8) #change for size of video
    fgbg = cv2.BackgroundSubtractorMOG()
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        fgmask = fgbg.apply(frame, learningRate=1.0/history)
        mask_rbg = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
        # half_show('frame', frame)
        # half_show('background subtractor',fgmask)
        # cv2.waitKey(100)
        cellcount,circ = calculate_cell_count(mask_rbg)
        if circ != None:
            for value in circ:
                circs.append(value)
        cellcounts.append(cellcount)
        print "running"
        # import ipdb; ipdb.set_trace()

    track, traj = tracking(video)
    hm = heatmap(circs, blank)
    plt.plot(hm)
    times = np.linspace(0, 6, len(video))
    cellcounts = np.array(cellcounts)
    times = np.array(times)
    mask = np.arange(0, len(cellcounts))[10:]

    # plt.plot(times, cellcounts[mask])
    # plt.figure()
    # plt.plot(times, cellareacounts[mask])
    # plt.show()

    frame = pd.DataFrame()
    frame['times'] = times
    frame['cell_count'] = cellcounts
    frame['cell_count_area'] = cellareacounts
    frame['track'] = track
    frame['traj'] = traj

    import os
    base_name = os.path.basename(video)
    frame.to_csv(base_name+'_export.csv', indexGG=False)

# for v in tqdm(video):
process_video(video)
# tracking(video)
#        # with open('cell_count.csv', 'w') as csvfile:
#    #     fieldnames = ["image","count"]
#    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    #     writer.writerow({'image': image_num, 'count': len(circles[0])})
#    image_num += 1
