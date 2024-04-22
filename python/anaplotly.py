import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import timedelta
from PIL import Image
import cv2

dataFileName = 'yolo/results.csv'
csvFileName = 'yolo/birdstatus.csv'
flagFigScatter = False
flagFigTime = False
flagMixedFig= False
flagFig3d = True
flagMakeCSV = True

figSizeX = 1920
figSizeY = 1080
figReduceRatioX = 15
figReduceRatioY = 15

def distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # d = ds.cityblock(p1, p2)
    return d

def cutFirstFrame(videoFielName):
    cap = cv2.VideoCapture(videoFielName)
    if not cap.isOpened():
        exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('{}_{}.{}'.format('yolo/frame', '001', 'png'), frame)
    else:
        exit(1)

def readData(fileName):
    org_data = []
    with open(fileName) as csvf:
        videoFileName = csvf.readline()
        reader = csv.reader(csvf)
        org_data = [row for row in reader]

    fvideo = videoFileName.split()
    print(fvideo[2] + ",")
    cutFirstFrame(fvideo[2])
    df_data = pd.DataFrame(org_data)
    data = df_data.to_numpy()
    m = int(data[-1,0])
    return m, data

def putFigScatter(Bird1, Bird2):
    plt.figure(2)
    #plt.plot(B[:, 0], 1 - B[:, 1], '.', label="bird1")
    #plt.plot(C[:, 0], 1 - C[:, 1], '.', label="bird2")
    plt.plot(Bird1[:, 0], 1024 - Bird1[:, 1], '.', label="Bird1")
    plt.plot(Bird2[:, 0], 1024 - Bird2[:, 1], '.', label="Bird2")
    plt.legend()
    plt.axis([0, figSizeX, 0, figSizeY])
    plt.xlabel('Width (pixel)')
    plt.ylabel('Height (pixel)')
    plt.savefig('yolo/resultsFigScatter.png')

def putFigTime(L, numFrame, ratio1, ratio2, ratioTotal):
    if numFrame/600 < 10:
        xticks = np.arange(0, numFrame, 300)
    else:
        xticks = np.arange(0, numFrame, 600*5)
    xlabels = [f'{x:.1f}' for x in xticks/600]

    plt.figure(3)
    ax1 = plt.subplot(3,1,1)
    plt.plot(L[:, 0], L[:, 1], '-', label="Bird1: " + "{:.2f}".format(ratio1) + "%", linewidth=2.0, color='tab:blue')
    plt.axis([0, numFrame, 0, 1.2])
    plt.yticks(np.arange(0, 2, 1))
    plt.ylabel("Bird 1")
    plt.tick_params('x', labelbottom=False)
    plt.legend()
    plt.grid()

    ax2 = plt.subplot(3,1,2, sharex=ax1, sharey=ax1)
    plt.plot(L[:, 0], L[:, 2], '-', label="Bird2: " + "{:.2f}".format(ratio2) + "%", linewidth=2.0, color='tab:red')
    plt.xticks(xticks, labels=xlabels)
    plt.tick_params('x', labelbottom=False)
    plt.ylabel("Bird 2")
    plt.legend()
    plt.grid()

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    plt.plot(L[:, 0], L[:, 3], '-', label="Total: " + "{:.2f}".format(ratioTotal) + "%", linewidth=2.0, color='tab:gray')
    plt.legend()
    plt.axis([0, numFrame, 0, 2.2])
    plt.yticks(np.arange(0, 3, 1))
    plt.xticks(xticks, labels=xlabels)
    plt.grid()
    plt.xlabel("Time [min]")
    plt.ylabel("Total")
    plt.savefig('yolo/resultsFigTime.png')

def putMixedFig(xy1, xy2):
    xy1[0, 0] = 0
    xy2[0, 0] = 0
    x00, y00 = np.arange(0,int(figSizeX/figReduceRatioX), 1), np.arange(0,int(figSizeY/figReduceRatioY), 1)
    exy1 = np.where(xy1 > 0.0, 250, 0)
    exy2 = np.where(xy2 > 0.0, 250, 0)
    location1 = np.uint8(exy1).transpose()
    location2 = np.uint8(exy2).transpose()

    extent = np.min(x00), np.max(x00), np.min(y00), np.max(y00)
    fig = plt.figure(9, frameon=False)
    img = Image.open('yolo/frame_001.png')
    im1 = plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest', extent=extent)
    im3 = plt.imshow(location2, cmap=plt.cm.Reds, alpha=0.8, interpolation='nearest', extent=extent)
    im2 = plt.imshow(location1, cmap=plt.cm.GnBu, alpha=0.5, interpolation='nearest', extent=extent)
    plt.savefig('yolo/resultsSuper.jpg')

def calcXY100(Bird1, Bird2):
    sizeX = int(figSizeX/figReduceRatioX)
    sizeY = int(figSizeY/figReduceRatioY)
    xy1 = np.zeros((sizeX, sizeY), int)
    xy2 = np.zeros((sizeX, sizeY), int)
    blen = len(Bird1)
    for i in range(blen):
        x0 = int(Bird1[i, 0]/figReduceRatioX)
        y0 = int(Bird1[i, 1]/figReduceRatioY)
        xy1[x0, y0] += 1
        x1 = int(Bird2[i, 0]/figReduceRatioX)
        y1 = int(Bird2[i, 1]/figReduceRatioY)
        xy2[x1, y1] += 1
    return xy1, xy2

def calcXX(Bird1, Bird2):
    blen = len(Bird1) 
    mm1 = np.zeros([blen + 2, 3])
    mm2 = np.zeros([blen + 2, 3])
    mm3 = np.zeros([2 * blen + 2, 3])
    for i in range(0, blen, 10):
        x0 = int(Bird1[i, 0]/figReduceRatioX)
        y0 = figSizeY/figReduceRatioY - int(Bird1[i, 1]/figReduceRatioY)
        x1 = int(Bird2[i, 0]/figReduceRatioX)
        y1 = figSizeY/figReduceRatioY - int(Bird2[i, 1]/figReduceRatioY)
        time = i/600
        mm1[i, :] = [time, x0, y0]
        mm2[i, :] = [time, x1, y1]
        mm3[2*i,:] = [time, x0, y0]
        mm3[2*i + 1,:] = [time, x1, y1]
    mm1[blen,:] = [blen/600, 0, 0]
    mm1[blen+1,:] = [(blen+1)/600, figSizeX/figReduceRatioX, figSizeY/figReduceRatioY]
    mm2[blen,:] = [blen/600, 0, 0]
    mm2[blen+1,:] = [(blen+1)/600, figSizeX/figReduceRatioX, figSizeY/figReduceRatioY]
    mm3[-2,:] = [blen/600, 0, 0]
    mm3[-1,:] = [(blen+1)/600, figSizeX/figReduceRatioX, figSizeY/figReduceRatioY]
    return mm3

def putFig3d(mm):
    df = pd.DataFrame(mm, columns = ['Time (min)', 'Width', 'Height'] )
    fig = px.scatter_3d(df, x = 'Time (min)', y = 'Width', z = 'Height',
                        color='Time (min)', size='Time (min)', size_max=10)
    fig.show()

def calcRatios(L):
    mm = len(L)
    tlen = mm /(10 * 60)
    ex1 = np.sum(L[:,1])
    ex2 = np.sum(L[:,2])
    ex3 = np.where(L[:,3] > 0)[0]
    len_ex3 = len(ex3)

    ratio1 =  ex1/mm*100
    ratio2 =  ex2/mm*100
    ratioTotal =  len_ex3/mm*100
    return mm, tlen, ratio1, ratio2, ratioTotal

def printStats(mm, tlen, ratio1, ratio2, ratioTotal):
    print('Number of frames: ' + str(mm) + '(' + "{:.3f}".format(tlen) + 'm)' )
    print("Bird1: " + "{:.2f}".format(ratio1) + '%')
    print("Bird2: " + "{:.2f}".format(ratio2) + '%')
    print("Total: " + "{:.2f}".format(ratioTotal) + '%')

def countBirds(m, npdata):
    J = np.zeros(shape=(m, 7))
    B = np.zeros(shape=(m, 2))
    C = np.zeros(shape=(m, 2))

    for i in range(m):
        float_i = float(i + 1)
        np_data = npdata[npdata[:, 0] == str(i+1)]
        if len(np_data) != 0:
            J[i, 0] = len(np_data) # number of birds
            len(np_data)

        if len(np_data) == 1:
            J[i, 1] = np_data[0, 1]
            J[i, 2] = np_data[0, 2]
            J[i, 3] = np_data[0, 3]

        elif len(np_data) == 2:
            if int(np_data[0, 1]) < int(np_data[1, 1]):
                J[i, 1] = np_data[0, 1]
                J[i, 2] = np_data[0, 2]
                J[i, 3] = np_data[0, 3]

                J[i, 4] = np_data[1, 1]
                J[i, 5] = np_data[1, 2]
                J[i, 6] = np_data[1, 3]
            else:
                J[i, 1] = np_data[1, 1]
                J[i, 2] = np_data[1, 2]
                J[i, 3] = np_data[1, 3]

                J[i, 4] = np_data[0, 1]
                J[i, 5] = np_data[0, 2]
                J[i, 6] = np_data[0, 3]

        m2 = len(J)
        s1 = J[0, 1:4]
        s2 = J[0, 4:7]

        B[0, 0] = J[0, 2]
        B[0, 1] = J[0, 3]
        C[0, 0] = J[0, 5]
        C[0, 1] = J[0, 6]

    for i in range(1, m2):
        L = []
        d1 = distance(s1[1:3], J[i, 2: 4])
        d2 = distance(s1[1:3], J[i, 5: 7])
        d3 = distance(s2[1:3], J[i, 2: 4])
        d4 = distance(s2[1:3], J[i, 5: 7])
        L.append(d1)
        L.append(d2)
        L.append(d3)
        L.append(d4)
        loc = L.index(np.min(L))

        if loc == 0 or loc == 3:
            s1 = J[i, 1:4]
            s2 = J[i, 4:7]
            if np.any(J[:, 5]) > 0:
                B[i, 0] = J[i, 2]
                B[i, 1] = J[i, 3]
                C[i, 0] = J[i, 5]
                C[i, 1] = J[i, 6]
            else:
                B[i, 0] = J[i, 2]
                B[i, 1] = J[i, 3]
        else:
            s1 = J[i, 4:7]
            s2 = J[i, 1:4]

            if np.any(J[:, 5]) > 0:
                B[i, 0] = J[i, 5]
                B[i, 1] = J[i, 6]
                C[i, 0] = J[i, 2]
                C[i, 1] = J[i, 3]
            else:
                B[i, 0] = J[i, 2]
                B[i, 1] = J[i, 3]
    return B, C


def calcBirds(m, B, C):
    L3 = np.zeros(shape=(m, 4))
    t2 = np.arange(1, m + 1)
    L3[:, 0] = t2
    for i in range(len(t2)):
        if B[i, 0] > 0:
            L3[i, 1] = 1
        else:
            L3[i, 1] = 0

        if C[i, 0] > 0:
            L3[i, 2] = 1
        else:
            L3[i, 2] = 0

    L3[:, 3] = L3[:, 1] + L3[:, 2]
    return L3


def smoothData(m, B, C):
    flag1 = 0
    b = {}
    for i in range(m):
        if B[i, 0] > 0:
            b[i] = 1
            flag1 = 1
        elif (i > 0 and B[i - 1, 0] > 0 and B[i, 0] == 0 and B[i - 1, 0] < 0.6 and B[i - 2, 1] < 0.5) and flag1 == 1:
            b[i] = 1
            flag1 = 1
            B[i, :] = B[i - 1, :]
        else:
            b[i] = 0
            flag1 = 0

    flag2 = 0
    c = {}
    for i in range(m):
        if C[i, 0] > 0:
            c[i] = 1
            flag2 = 1
        elif (i > 0 and C[i - 1, 0] > 0 and C[i, 0] == 0 and C[i - 1, 0] < 0.2 and C[i - 2, 1] < 0.2) and flag2 == 1:
            c[i] = 1
            flag2 = 1
            C[i, :] = C[i - 1, :]
        else:
            c[i] = 0
            flag2 = 0
    return B, C

def makeTable(m, B, C):
    L = np.zeros(shape=(m, 4))
    t = np.arange(1, m + 1)

    L[:, 0] = t

    for i in range(len(t)):
        if B[i, 0] > 0:
            L[i, 1] = 1
        else:
            L[i, 1] = 0

        if C[i, 0] > 0:
            L[i, 2] = 1
        else:
            L[i, 2] = 0

    L[:, 3] = L[:, 1] + L[:, 2]
    return L

def makeCSV(fileName, m, L):
    L1 = np.zeros(shape=(m, 4), dtype=str)
    t = np.arange(1, m + 1)
    t1 = []
    t1_str = []
    for i in range(len(t)):
        second = t[i] * 0.1
        td = timedelta(seconds=second)
        t_str = str(td)
        t1.append(t_str)
        t1_str.append(t_str)

    L1[:, 0] = t1
    L1[:, 1:4] = L[:, 1: 4]

    t1_str_pd = pd.DataFrame(t1_str)
    L1_pd = pd.DataFrame(L1[:, 1:4])
    L1_data_pd = pd.concat([t1_str_pd, L1_pd], axis=1)
    L1_data_pd.to_csv(fileName, header=False, index=False)

m, npdata = readData(dataFileName)
B,C = countBirds(m, npdata)
#L3 = calcBirds(m, B, C)
#B, C = smoothData(m, B, C)
BirdTable = makeTable(m, B, C)

if flagFigScatter == True:
    putFigScatter(B, C)

if flagMixedFig:
    xy1, xy2 = calcXY100(B, C)
    putMixedFig(xy1, xy2)

if flagFig3d:
    mm = calcXX(B, C)
    putFig3d(mm)

if flagFigTime == True:
    mm, tlen, ratio1, ratio2, ratioTotal = calcRatios(BirdTable)
    printStats(mm, tlen, ratio1, ratio2, ratioTotal)
    putFigTime(BirdTable, mm, ratio1, ratio2, ratioTotal)

if flagMakeCSV == True:
    makeCSV(csvFileName, m, BirdTable)