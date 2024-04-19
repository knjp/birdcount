import csv
import pandas as pd
import numpy as np
from scipy.signal import medfilt 
import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib import cm
from matplotlib import colormaps as cm2
from PIL import Image
import cv2

dataFileName = 'yolo/results.csv'
csvFileName = 'yolo/birdstatus.csv'
figflag1 = False
figflag2 = True
figflag3 = True
figflag4 = True
figflag5 = False
figflag6 = False
figheatmap = True
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

def putFig1(allBirds):
    plt.figure(1)
    plt.plot(allBirds[:, 0], allBirds[:, 3], '.-', label="Total")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of Birds in video")
    plt.axis([0, m, 0, 3])
    plt.savefig('yolo/resultsFig1.png')

def putFig2(Bird1, Bird2):
    plt.figure(2)
    #plt.plot(B[:, 0], 1 - B[:, 1], '.', label="bird1")
    #plt.plot(C[:, 0], 1 - C[:, 1], '.', label="bird2")
    plt.plot(Bird1[:, 0], 1024 - Bird1[:, 1], '.', label="Bird1")
    plt.plot(Bird2[:, 0], 1024 - Bird2[:, 1], '.', label="Bird2")
    plt.legend()
    plt.axis([0, figSizeX, 0, figSizeY])
    plt.xlabel('Width (pixel)')
    plt.ylabel('Height (pixel)')
    plt.savefig('yolo/resultsFig2.png')

def putFig3(L, numFrame, ratio1, ratio2, ratioTotal):
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
    plt.savefig('yolo/resultsFig3.png')

def putFig5(m, L):
    t = np.arange(1, m + 1)
    MF = np.zeros(shape=(m, 4))
    MF[:, 0] = t
    MF[:,1] = medfilt(L[:,1], kernel_size=9)
    MF[:,2] = medfilt(L[:,2], kernel_size=9)
    MF[:,3] = medfilt(L[:,3], kernel_size=9)
    plt.figure(5)
    plt.plot(MF[:, 0], MF[:, 1] + 0.03, '.-', label="bird1")
    plt.plot(MF[:, 0], MF[:, 2], '.-', label="bird2")
    plt.plot(MF[:, 0], MF[:, 3], '.-', label="Total")
    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("Bird Count")
    plt.axis([0, m, 0, 3])
    plt.savefig('yolo/resultsFig5.png')

def putFig6(xy):
    xy[0, 0] = 0
    x00, y00 = np.arange(0,figSizeX, 1), np.arange(0,figSizeY, 1)
    mx, my = np.meshgrid(x00, y00)
    z = xy[mx,my]
    #xpos = mx.ravel()
    #ypos = my.ravel()
    #zpos = 0
    #dx = dy = np.ones_like(zpos)
    #dz = z.ravel()
    plt.figure(5)
    plt.scatter(mx, my, s=1, c=z)
    plt.savefig('yolo/resultsFig6.png')

def putHeatmap(xy):
    xy[0, 0] = 0
    x00, y00 = np.arange(0,figSizeX, 1), np.arange(0,figSizeY, 1)
    mx, my = np.meshgrid(x00, y00)
    maxz = np.max(xy) 
    #xy = (xy/maxz) * 255
    #amax  = np.unravel_index(np.argmax(xy, axis=None), xy.shape) 
    z = xy[mx,my]
    z2 = np.where(z > 0.1, 100, 0)

    ax = plt.figure(4).add_subplot(projection='3d')
    #ax.plot_surface(mx, 1080-my, z, cmap= cm.coolwarm)
    ax.plot_surface(mx, 1080-my, z2, cmap=cm.binary)
    plt.savefig('yolo/resultsFig4.png')
    #ax.set(zlim=(0,1))

    xy2 = np.where(xy > 0.0, 200, 0)
    #heatmap0 = np.uint8(xy/np.max(xy) * 255)
    heatmap0 = np.uint8(xy2)
    heatmap = heatmap0.transpose()
    #heatmap[:,1] = heatmap[:,1]
    hmimage = Image.fromarray(heatmap)
    hmimage.save('yolo/heatmap.png')

    img = Image.open('yolo/frame_001.png')
    jet = cm2.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:,:3]
    jet_heatmap=jet_colors[heatmap]
    hmmax = np.max(jet_heatmap)
    jhm = jet_heatmap/hmmax * 255
    jet_heatmap = Image.fromarray(np.uint8(jhm))
    jet_heatmap = jet_heatmap.resize((figSizeX, figSizeY))
    jet_heatmap = np.asarray(jet_heatmap)

    superimposed = jet_heatmap * .2 + img
    superimposed = Image.fromarray(np.uint8(superimposed))
    superimposed.save('yolo/superimposed.png')

def putMixedFig(xy1, xy2):
    xy1[0, 0] = 0
    xy2[0, 0] = 0
    x00, y00 = np.arange(0,int(figSizeX/figReduceRatioX), 1), np.arange(0,int(figSizeY/figReduceRatioY), 1)
    exy1 = np.where(xy1 > 0.0, 250, 0)
    exy2 = np.where(xy2 > 0.0, 250, 0)
    location1 = np.uint8(exy1).transpose()
#    location1 = location1.transpose()
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

def calcXY(Bird1, Bird2):
    xy = np.zeros((figSizeX, figSizeY), int)
    blen = len(Bird1)
    h5 = 1
    w5 = h5 * 3
    for i in range(blen):
        x0 = int(Bird1[i, 0])
        y0 = int(Bird1[i, 1])
        #xy[x0-w5:x0+w5, y0-h5:y0+h5] += 1
        xy[x0, y0] += 1
        x1 = int(Bird2[i, 0])
        y1 = int(Bird2[i, 1])
        #xy[x1-w5:x1+w5, y1-h5:y1+h5] += 1
        xy[x1, y1] += 1
    return xy

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
L3 = calcBirds(m, B, C)
xy = calcXY(B, C)
xy1, xy2 = calcXY100(B, C)

if figflag1 == True:
    putFig1(L3)

if figflag2 == True:
    putFig2(B, C)


if figflag6 == True:
    putFig6(xy)

if figheatmap:
    putMixedFig(xy1, xy2)
    #putHeatmap(xy)


B, C = smoothData(m, B, C)
BirdTable = makeTable(m, B, C)
makeCSV(csvFileName, m, BirdTable)
mm, tlen, ratio1, ratio2, ratioTotal = calcRatios(BirdTable)
printStats(mm, tlen, ratio1, ratio2, ratioTotal)

if figflag3 == 1:
    putFig3(BirdTable, mm, ratio1, ratio2, ratioTotal)

if figflag5 == True:
    putFig5(m, BirdTable)