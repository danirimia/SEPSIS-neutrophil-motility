import numpy as np
import sys
import pandas as pd
import matplotlib.pylab as plt
from tqdm import tqdm

MAZE_SIZE = 1600, 588

SEG = 1190
MAZE_AREA = SEG*11*8 + (SEG*7*12)
RES = 3

def calculate_total_area(xs, ys, diam):
    n = 40
    circle_mask = np.zeros((n*2+1, n*2+1))
    cell_size = 7*RES
    for rx in range(n*2+1):
        for ry in range(n*2+1):
            x = rx - n
            y = ry - n
            circle_mask[rx,ry] = (cell_size > np.sqrt(x**2 + y**2))
    #plt.imshow(circle_mask)
    #plt.show()
    off = 300
    #mask = np.zeros((off+1600 * RES, off+588 * RES))
    mask = np.zeros((2*off+1600 * RES, 2*off+588 * RES))
    mask_1 = np.zeros((2*off+1600 * RES, 2*off+588 * RES))
    mask_2 = np.zeros((2*off+1600 * RES, 2*off+588 * RES))

    offset = 2600 / 3 * RES

    for x,y,d in zip(xs, ys, diam):
        try:
            mask[off+x*RES-n:off+x*RES+n+1, off+y*RES-n:off+y*RES+n+1] += circle_mask
            if x >= offset / 2 / RES:
                mask_1[off+x*RES-n:off+x*RES+n+1, off+y*RES-n:off+y*RES+n+1] += circle_mask
            else:
                mask_2[off+x*RES-n:off+x*RES+n+1, off+y*RES-n:off+y*RES+n+1] += circle_mask
        except ValueError:
            print "coulnt set wrong shapes"
    mask = np.minimum(mask, 1)
    mask = mask[:offset]
    mask_1 = np.minimum(mask_1, 1)
    mask_1 = mask_1[:offset]
    mask_2 = np.minimum(mask_2, 2)
    mask_2 = mask_2[:offset]
    #plt.figure()
    #plt.imshow(mask)
    #plt.figure()
    #plt.imshow(mask_1)
    #plt.figure()
    #plt.imshow(mask_2)
    #plt.show()
    return mask, mask_1, mask_2

from glob import glob

if __name__ == "__main__":
    files = glob(sys.argv[1] + "/*.csv")
    outs = []
    #files = files[0:2]
    for f in tqdm(files):
        frame = pd.read_csv(f)
        xs = frame['x'].values
        ys = frame['y'].values
        ds = frame['size'].values
        m1, m2, m3 = calculate_total_area(xs, ys, ds)
        area_total = np.sum(m1) / (RES*RES) / MAZE_AREA
        #import ipdb; ipdb.set_trace()
        percentA = np.sum(m3) / np.sum(m2+m3)
        percentB = np.sum(m2) / np.sum(m2+m3)
        outs.append((percentA, percentB, area_total))
    frame = dict()
    frame['percent_1'] = zip(*outs)[0]
    frame['percent_2'] = zip(*outs)[1]
    frame['total_area'] = zip(*outs)[2]
    frame['files'] = files
    frame = pd.DataFrame.from_dict(frame)
    frame.to_csv("output.csv")
