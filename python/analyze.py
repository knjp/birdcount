import csv
import pandas as pd
import numpy as np
from scipy.signal import medfilt 
import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib import cm
from matplotlib import colormaps as cm2
from PIL import Image

def distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # d = ds.cityblock(p1, p2)
    return d


org_data = []
with open('yolo/results2.csv') as csvf:
    reader = csv.reader(csvf)
    org_data = [row for row in reader]

df_data = pd.DataFrame(org_data)
npdata = df_data.to_numpy()
m = int(npdata[-1,0])
#print(m)

J = np.zeros(shape=(m, 7))
J1 = np.zeros(shape=(m, 7))
B = np.zeros(shape=(m, 2))
C = np.zeros(shape=(m, 2))

#print(npdata[npdata[:,0] == '387'])
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

figflag1 = 1
figflag2 = 1
figflag3 = 1

m2 = len(J)
s1 = J[0, 1:4]
s2 = J[0, 4:7]
#print('s1: ' + str(s1))
#print('s2: ' + str(s2))

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
    #print('L: ' + str(L))
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
    #print('B[[' + str(i) + ',]: ' + str(B[i, ]) + '   C[' + str(i) + ',]: ' + str(C[i,]))

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
    #print('L3[[' + str(i) + ',]: ' + str(L3[i, ]))



L3[:, 3] = L3[:, 1] + L3[:, 2]

if figflag1 == 5:
    plt.figure(1)
    plt.plot(L3[:, 0], L3[:, 3], '.-', label="Total")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of Birds in video")
    plt.axis([0, m, 0, 3])

xy = np.zeros((1920, 1080), int)
blen = len(B)
h5 = 2
w5 = h5 * 3
for i in range(blen):
    x0 = int(B[i, 0])
    y0 = int(B[i, 1])
    xy[x0-w5:x0+w5, y0-h5:y0+h5] += 1
    x1 = int(C[i, 0])
    y1 = int(C[i, 1])
    xy[x1-w5:x1+w5, y1-h5:y1+h5] += 1

xy[0, 0] = 0

x00, y00 = np.arange(0,1920, 1), np.arange(0,1080, 1)
mx, my = np.meshgrid(x00, y00)
z = xy[mx,my]

#xpos = mx.ravel()
#ypos = my.ravel()
#zpos = 0
#dx = dy = np.ones_like(zpos)
#dz = z.ravel()

plt.figure(5)
plt.scatter(mx, my, s=500, c=z, cmap='Blues')
plt.show()
#print(mx.shape)

exit(0)

ax = plt.figure(4).add_subplot(projection='3d')
ax.plot_surface(mx, 1080-my, z, cmap= cm.coolwarm)
#ax.set(zlim=(0,30))

heatmap0 = np.uint8(xy/np.max(xy) * 255)
heatmap = heatmap0.transpose()
heatmap[:,1] = heatmap[:,1]
hh = Image.fromarray(heatmap)
hh.save('test50.png')


img = Image.open('yolo/frame_001.png')
jet = cm2.get_cmap("jet")
jet_colors = jet(np.arange(256))[:,:3]
jet_heatmap=jet_colors[heatmap]

jet_heatmap = Image.fromarray(np.uint8(jet_heatmap))

jet_heatmap = jet_heatmap.resize((1920, 1080))
jet_heatmap = np.asarray(jet_heatmap)

superimposed = jet_heatmap * 200 + img
superimposed = Image.fromarray(np.uint8(superimposed))

superimposed.save('test.png')



if figflag2 == 1:
    plt.figure(2)
    #plt.plot(B[:, 0], 1 - B[:, 1], '.', label="bird1")
    #plt.plot(C[:, 0], 1 - C[:, 1], '.', label="bird2")
    plt.plot(B[:, 0], 1024 - B[:, 1], '.', label="bird1")
    plt.plot(C[:, 0], 1024 - C[:, 1], '.', label="bird2")
    plt.legend()
    plt.axis([0, 1920, 0, 1024])

plt.show()
exit(0)

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

L = np.zeros(shape=(m, 4))
L1 = np.zeros(shape=(m, 4), dtype=str)
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



t1 = []
t1_str = []
for i in range(len(t)):
    second = t[i] * 0.1
    td = timedelta(seconds=second)
    t_str = str(td)
    t1.append(t_str)
    t1_str.append(t_str)

L1[:, 0] = t1
#print(L1)
L1[:, 1:4] = L[:, 1: 4]
#print(L1)

t1_str_pd = pd.DataFrame(t1_str)
L1_pd = pd.DataFrame(L1[:, 1:4])
L1_data_pd = pd.concat([t1_str_pd, L1_pd], axis=1)
L1_data_pd.to_csv('yolo/birdstatus.csv', header=False, index=False)

MF = np.zeros(shape=(m, 4))
MF[:, 0] = t
MF[:,1] = medfilt(L[:,1], kernel_size=9)
MF[:,2] = medfilt(L[:,2], kernel_size=9)
MF[:,3] = medfilt(L[:,3], kernel_size=9)

mm = len(L)
tlen = mm /(10 * 60)
print('Number of frames: ' + str(mm) + '(' + "{:.3f}".format(tlen) + 'm) <br>' )

#print(  'Exec  Time: ' + "{:.3f}".format(dtime) + ' sec' + '<br>')

ex1 = np.sum(L[:,1])
ex2 = np.sum(L[:,2])
ex3 = np.sum(L[:,3])
print('')

print("Bird1: " + "{:.3f}".format(ex1/mm*100) + '%<br>')
print("Bird2: " + "{:.3f}".format(ex2/mm*100) + '%<br>')
print("Total: " + "{:.3f}".format(ex3/mm*100) + '%')
#      ,ex2/mm,ex3/mm)

if figflag3 == 1:
    plt.figure(3)
    plt.plot(L[:, 0], L[:, 1] + 0.03, '.-', label="bird1")
    plt.plot(L[:, 0], L[:, 2], '.-', label="bird2")
    plt.plot(L[:, 0], L[:, 3], '.-', label="Total")
    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("Bird Count")
    plt.axis([0, m, 0, 3])
    plt.show()

#plt.figure(4)
#plt.plot(MF[:, 0], MF[:, 1] + 0.03, '.-', label="bird1")
#plt.plot(MF[:, 0], MF[:, 2], '.-', label="bird2")
#plt.plot(MF[:, 0], MF[:, 3], '.-', label="Total")
#plt.legend()
#plt.xlabel("frame")
#plt.ylabel("Bird Count")
#plt.axis([0, m, 0, 3])
#plt.show()
#plt.pause(1)