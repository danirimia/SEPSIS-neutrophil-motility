#!/usr/bin/env python
#---------------------------------------
# Analysis of Neutrophils, NETs and ROS production with Fungus
# Author: Julianne
#---------------------------------------

### IMPORT ALL THE THINGS
#%matplotlib inline

import cv2 as cv2
from matplotlib import pyplot as plt
import numpy as np
import time as t
import ipdb
import sys
import pandas as pd
import os
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
#frames = []
# video = '/media/julianne/Julianne/Imaging/BATTLEDOME AVI/20150804  Neuts DPI NEi BattleDome + CellROX021.avi'
# video = '/media/julianne/Julianne/Imaging/BATTLEDOME AVI/20150804  Neuts DPI NEi BattleDome + CellROX005.avi'
video = sys.argv[1]
videos = [os.path.join(video, s) for s in os.listdir(video)]
videos = sorted(videos)
videos = videos[0:24]

def half_show(name, frame):
    cv2.imshow(name, cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2)))

def pixel2micron(area):
    # pixels**2  * (microns* / pixels)**2 = microns**2
    area = area*((50./109)**2)
    return area

# def area2cells(area):
#     # pixels**2  * (microns* / pixels)**2 = microns**2
#     cellnum = area*((50./109)**2)
#     return cellnum

def calculate_red_area(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # red_lower = np.array([120,60,30])
    # red_upper = np.array([180,100,70])
    red_lower = np.array([100,10,10])
    red_upper = np.array([180,240,240])

    red_mask = cv2.inRange(hsv,red_lower,red_upper)
    red_filtered = cv2.bitwise_and(frame,frame,mask=red_mask)  
    #half_show("red mask",red_mask)
    blur_mask = cv2.medianBlur(red_mask, 3)
    half_show("red blur", blur_mask)
    red_area = np.sum(blur_mask)
    red_area = pixel2micron(red_area)
    cv2.waitKey(0)

    if red_area > 0:
        red_area = red_area/53
    return red_area

def calculate_net_area(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    #blue_lower = np.array([0,150,50])
    #blue_upper = np.array([20,255,150])
    blue_lower = np.array([0,150,10])
    blue_upper = np.array([20,255,50])
    blue_mask = cv2.inRange(hsv,blue_lower, blue_upper)
    blue_filtered = cv2.bitwise_and(frame,frame,mask=blue_mask)

    mask = cv2.inRange(hsv,blue_lower, blue_upper)
    filtered = cv2.bitwise_and(frame,frame,mask=mask)  
    #half_show("blue mask",blue_mask)
    #half_show("fame",frame)
    mask = cv2.medianBlur(mask, 3)
    #half_show("median", mask)
    area = np.sum(mask)
    area = pixel2micron(area)
    cv2.waitKey(0)
    return area

def calculate_cell_area(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    blur = cv2.medianBlur(frame,5)
    blue_grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    blue_lower = np.array([0,150,50])
    blue_upper = np.array([20,255,150])
    blue_mask = cv2.inRange(hsv,blue_lower, blue_upper)
    blue_filtered = cv2.bitwise_and(frame,frame,mask=blue_mask)

    mask = cv2.inRange(hsv,blue_lower, blue_upper)
    filtered = cv2.bitwise_and(frame,frame,mask=mask)  
    half_show("blue mask",blue_mask)
    # half_show("frame",frame)
    mask = cv2.medianBlur(mask, 3)
    # half_show("median", mask)
    area = np.sum(mask)
    area = pixel2micron(area)
    cellnum = area/5
    # cv2.waitKey(0)
    return area

def calculate_cell_count(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    blur = cv2.medianBlur(frame,5)
    blue_grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    # half_show("blue grey",blue_grey)
    half_show("original", frame)
    
    blue_lower = np.array([0,150,50])
    blue_upper = np.array([20,255,150])
    blue_mask = cv2.inRange(hsv,blue_lower, blue_upper)
    blue_filtered = cv2.bitwise_and(frame,frame,mask=blue_mask)
    pre_circle = np.asarray(255 * ((blue_grey / 255.0) * (blue_mask / 255.0)), dtype=np.uint8)
    #half_show("bluemask", blue_mask)
    #half_show("pre_circle", pre_circle)
    circles = cv2.HoughCircles(pre_circle,cv2.cv.CV_HOUGH_GRADIENT,1,3,param1=10,param2=5,minRadius=5,maxRadius=10)
    if circles is None:
        #half_show("preview", frame)
        #cv2.waitKey(0)
        return 0

    for i in circles[0,:]:
       cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
       # cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

    numCells = len(circles[0])

    half_show("preview", frame)
    cv2.waitKey(0)
    return numCells

    # ##VIDEO PROCESSING
    # cap = cv2.VideoCapture(video)
    # cap.open(video)
    # # import ipdb
    # # ipdb.set_trace()
    # if not cap.isOpened():
    #     raise Exception("Path not found")
    # # cur_frame_idx = 0
    # while(cap.isOpened()):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     #frames.append(frame)
    #     # cv2.waitKey(1000)
    #     # cv2.imshow("raw", cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2)))
    #     #cv2.waitKey(0)
    #     print cur_frame_idx
    #     if cur_frame_idx == 10:
    #       cur_frame = frame
    #         break
    #     cur_frame_idx += 1
    #     continue


    # calculate_cell_count(cur_frame)

def process_video(video):
    cap = cv2.VideoCapture(video)
    cap.open(video)
    # import ipdb
    # ipdb.set_trace()
    cur_frame_idx = 0
    redareas = []
    netareas = []
    cellcounts = []
    cellareacounts = []
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break
        redarea = calculate_red_area(frame)
        redareas.append(redarea)
        netarea = calculate_net_area(frame)
        netareas.append(netarea)
        cellcount = calculate_cell_count(frame)
        cellcounts.append(cellcount)
        cellareacount = calculate_cell_area(frame)
        cellareacounts.append(cellareacount)
        # import ipdb; ipdb.set_trace()

    times = np.linspace(0, 6, len(netareas))
    netareas = np.array(netareas)
    redareas = np.array(redareas)
    cellcounts = np.array(cellcounts)
    cellareacounts = np.array(cellareacounts)
    times = np.array(times)
    mask = np.arange(0, len(netareas))[10:]

    plt.plot(times[mask], netareas[mask])
    plt.figure()
    plt.plot(times[mask], redareas[mask])
    plt.figure()
    plt.plot(times[mask], cellcounts[mask])
    plt.figure()
    plt.plot(times[mask], cellareacounts[mask])
    plt.show()


    frame = pd.DataFrame()
    frame['times'] = times
    frame['net_areas'] = netareas
    frame['ros_areas'] = redareas
    frame['cell_count'] = cellcounts
    frame['cell_count_area'] = cellareacounts
    frame['normalized_net'] = netareas / cellcounts
    frame['normalized_ros'] = redareas / cellcounts

    base_name = os.path.basename(video)
    frame.to_csv(base_name+'_export.csv', indexGG=False)

for v in tqdm(videos):
    process_video(v)
#        # with open('cell_count.csv', 'w') as csvfile:
#    #     fieldnames = ["image","count"]
#    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    #     writer.writerow({'image': image_num, 'count': len(circles[0])})
#    image_num += 1
