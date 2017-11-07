import pandas as pd
import glob
import numpy as np
from matplotlib import pylab as plt
from tqdm import trange
import os

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def load_raw(patient_name, draw_num):
    path = "converted_data/%s/control_%d.csv"%(patient_name, draw_num)
    if not os.path.exists(path):
        print path, "Not found"
        return None
    frame = pd.read_csv(path)
    return frame

def features(frame, draw_num):
    features = []
    has_cells = (len(frame['x']) != 0)
    features.append(has_cells)
    # Maybe remove, possibly data leakage....
    features.append(draw_num)
    if has_cells == False:
        return features + [0]*8
        return features + [0]*17

    num_cells = np.max(frame['track_n'])
    features.append(num_cells > 10)
    features.append(num_cells > 20)
    features.append(num_cells)

    #Parameters: N, O, P, R, MD, AD, V, A, Delta V, Delta A

    dx = np.diff(frame['x'])
    ddx = np.diff(dx)
    features.append(np.mean(dx))
    features.append(np.mean(np.abs(dx)))
    features.append(np.var(dx))
    features.append(np.sum(np.abs(np.sign(ddx))))

    dy = np.diff(frame['y'])
    ddy = np.diff(dy)
    features.append(np.mean(dy))
    features.append(np.mean(np.abs(dy)))
    features.append(np.var(dy))
    features.append(np.sum(np.abs(np.sign(ddy))))

    features.append(np.mean(frame['velocity']))
    features.append(np.var(frame['velocity']))

    pixel_value = frame['pixel_value']
    features.append(np.mean(pixel_value))
    features.append(np.var(pixel_value))

    distance = frame['distance']
    features.append(np.mean(distance))
    features.append(np.var(distance))

    return features

def load_labels():
    ret = {}
    for l in open("converted_data/labels.txt").readlines():
        name, draws = l.strip().split(":")
        print draws
        draws = [int(x) for x in draws.split(",") if x!='']
        ret[name] = draws
    return ret

def gather_feats_labels(patients, name_to_feats, name_to_labels):
    X = []
    Y = []
    for p in patients:
        X.extend(name_to_feats[p])
        Y.extend(name_to_labels[p])
    X,Y = zip(*[(x,y) for x,y in zip(X, Y) if x is not None])
    return np.asarray(X), np.asarray(Y)

def shuffle_train(X, Y, teX):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    sX = X[idx]
    sY = Y[idx]
    scalar = StandardScaler()
    lr = LogisticRegression(C=0.7)
    pf = PolynomialFeatures()
    sX = scalar.fit_transform(sX)
    #sX = pf.fit_transform(sX)
    lr.fit(sX, sY)

    teX = scalar.transform(teX)
    #teX = pf.fit_transform(teX)

    X = scalar.transform(X)
    #X = pf.fit_transform(X)


    tePY = lr.predict_proba(teX)[:, 1]
    trPY = lr.predict_proba(X)[:, 1]
    return trPY, tePY

def main():
    name_to_labels = load_labels()
    for k,v in name_to_labels.items():
        name_to_labels[k] = [0 if i==0 else 1 for i in v]
    name_to_feats = {}
    for k,v in name_to_labels.items():
        name_to_feats[k] = []
        for i,_ in enumerate(v):
            # draws 1 index
            frame = load_raw(k, i+1)
            if frame is not None:
                feat = features(frame, draw_num=i)
            else:
                feat = None
            name_to_feats[k].append(feat)

    #max_feats = max([len(x) for x in name_to_feats.values()[0] if x is not None])
    #for k,v in name_to_feats.items():
        #for i in range(len(v)):
            #v[i] = v[i] + [0]*(max_feats - len(v[i]))


    # fix the features where can't make with zeros

    accum = []
    for x in trange(10000):
        patients = name_to_labels.keys()
        n_train = int(len(patients) * .8)
        train_pat = np.random.choice(patients, size=n_train).tolist()
        test_pat = list(set(patients) - set(train_pat))

        trX, trY = gather_feats_labels(train_pat, name_to_feats, name_to_labels)
        teX, teY = gather_feats_labels(test_pat, name_to_feats, name_to_labels)
        if len(np.unique(teY)) == 1:
            continue
        if len(np.unique(trY)) == 1:
            continue
        X, Y = gather_feats_labels(patients, name_to_feats, name_to_labels)
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        spt = int(len(X)*.8)
        trX = X[idx[:spt]]
        teX = X[idx[spt:]]
        trY = Y[idx[:spt]]
        teY = Y[idx[spt:]]


        trPY, tePY = shuffle_train(trX, trY, teX)

        heldout_auc = roc_auc_score(teY, tePY)
        train_auc = roc_auc_score(trY, trPY)

        fpr, tpr, _ = roc_curve(teY, tePY)
        #plt.plot(fpr, tpr)
        accum.append([heldout_auc, train_auc])
    print np.mean(accum, axis=0), np.var(accum, axis=0)
    accum = np.asarray(accum)
    plt.subplot(2,1,1)
    plt.title("Histogram of AUC from ML based model")
    plt.hist(accum[:, 0], bins=100, normed=True, range=[0,1])
    plt.xlabel("Validation AUC")
    plt.subplot(2,1,2)
    plt.hist(accum[:, 1], bins=100, normed=True, range=[0,1])
    plt.xlim([0, 1])
    plt.xlabel("Train AUC")
    plt.show()
main()
