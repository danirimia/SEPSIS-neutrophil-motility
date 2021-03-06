#!/usr/bin/env python
#---------------------------------------
# Analysis of Neutrophils - unstained with gaussian filter
# Author: Julianne
#---------------------------------------

### IMPORT ALL THE THINGS
#%matplotlib inline
import cv2 as cv2
from matplotlib import pyplot as plt
import numpy as np
import time as t
import pandas as pd
import cPickle
# import pims
# import trackpy as tp
# import ipdb
import sys
import os
# import av
from tqdm import tqdm, trange
from scipy.ndimage.filters import gaussian_filter

# http://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
import numpy as np
import scipy.stats as st

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

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


#video = os.path.realpath("LeoVideo/20130919C5a10nM+LTB410nMxy41.avi")
# videos = sorted(videos)
# videos = videos[0:24]

def half_show(name, frame):
    cv2.imshow(name, cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2)))

def pixel2micron(area):
    # pixels**2  * (microns* / pixels)**2 = microns**2
    area = area*((50./109)**2)
    return area

def calculate_cell_count(frame):
    # print "calculating"
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
    # print "circles: ", filteredcircles
    return (numCells,filteredcircles)

def tracking(video):
    # print "running tracking"
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
    # print "running heatmap"
    #max values: [179,255,255]
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    hsv[:,:,0] = 51
    hsv[:,:,1] = 235
    offset = 400

    res = 3
    #jcount = np.zeros((hsv.shape[0] * res + offset*2, hsv.shape[1] * res + offset*2))
    count = np.zeros((hsv.shape[1] * res + offset*2, hsv.shape[0] * res + offset*2))
    s = count.shape

    r = 10
    circ_temp = np.zeros((r*2*res,r*2*res))
    for i in range(r*2*res):
        for j in range(r*2*res):
            dx = (i - (r*res))
            dy = (j - (r*res))
            dist = np.sqrt(dx**2 + dy**2)
            if dist < r*res:
                circ_temp[i,j] = 1

    #circ_temp = gkern(r*2*res, nsig=1)
    #plt.imshow(circ_temp)
    #plt.show()

    for i, point in tqdm(list(enumerate(circles))):
        tl = (max(0, point[0]-r),max(point[1]-r, 0)) #make sure on min or max
        br = (min(s[0], point[0]+r),min(s[1], point[1]+r)) #make sure on min or max
        tl = [int(x) for x in tl]
        br = [int(x) for x in br]

        dx = tl[0] - br[0]
        dy = tl[1] - br[1]
        #count[tl[0]:br[0], tl[1]:br[1]] += 1
        try:
            #count[tl[0]*res+offset:br[0]*res+offset, tl[1]*res+offset:br[1]*res+offset] += circ_temp
            count[tl[0]*res+offset:br[0]*res+offset, tl[1]*res+offset:br[1]*res+offset] += 1
        except:
            print "error"

    plt.imshow(count)
    plt.colorbar()
    plt.show()

    return count


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

    if True:
        rets, frames = [], []
        while(cap.isOpened()):
            # Capture frame-by-frame
            print "running while loop"
            ret, frame = cap.read()
            if frame is None:
                break
            rets.append(ret)
            frames.append(frame)

        for ret, frame in tqdm(list(zip(rets, frames))):
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

        cPickle.dump(circs, open("circs.pkl", "w"))
    circs = cPickle.load(open("circs.pkl"))

    hm = heatmap(circs, blank)
    cv2.imshow('heatmap', hm)
    cv2.waitKey(1000)

    track, traj = tracking(video)
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

video = sys.argv[1]
videos = [os.path.join(video, s) for s in os.listdir(video)]
for v in tqdm(video):
    process_video(video)

